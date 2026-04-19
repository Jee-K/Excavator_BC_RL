"""
VRAM-first trainer for the delta-target sequence MLP policy.

Design:
- split train/val at the segment level
- materialize every supervised sample for each split up front
- normalize states/goals once during materialization
- train on cumulative chunk deltas relative to the current/start action
- move the full train/val splits to VRAM before training
- batch by GPU-side indexing with tqdm progress bars

Target convention:
- For anchor t, start_action = normalized trajectory[t]
- Absolute future chunk:
      future_abs = normalized trajectory[t+1 : t+1+action_horizon]
- Supervised target chunk:
      future_delta = future_abs - start_action
  so each step is a cumulative delta from the current pose, not an incremental delta
  from the immediately previous future step.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Tuple

import h5py
import numpy as np
import torch
from torch import Tensor
from torchvision.transforms import Normalize
from tqdm.auto import tqdm

Batch = Dict[str, Tensor]


class MLPBlock(torch.nn.Module):
    def __init__(self, dims: Iterable[int], dropout: float = 0.0) -> None:
        super().__init__()
        dims = list(dims)
        if len(dims) < 2:
            raise ValueError("dims must contain at least input and output sizes")

        layers: list[torch.nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(torch.nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(torch.nn.ReLU())
                if dropout > 0.0:
                    layers.append(torch.nn.Dropout(dropout))
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class DeltaGoalConditionedMLPPolicy(torch.nn.Module):
    """
    Goal-conditioned MLP baseline that predicts a chunk of cumulative deltas.

    Batch convention:
        batch = {
            "state_history": Tensor[B, T, state_dim],
            "goal": Tensor[B, goal_dim],
            "action_targets": Tensor[B, K, action_dim],  # optional delta targets
            "start_action": Tensor[B, action_dim],       # optional, not required by the model
            "history_mask": BoolTensor[B, T],            # optional, ignored by this model
        }
    """

    def __init__(
        self,
        state_dim: int,
        goal_dim: int,
        action_dim: int,
        history_len: int,
        action_horizon: int,
        hidden_dims: Iterable[int] = (512, 512),
        dropout: float = 0.0,
        loss_type: str = "smooth_l1",
    ) -> None:
        super().__init__()
        self.state_dim = int(state_dim)
        self.goal_dim = int(goal_dim)
        self.action_dim = int(action_dim)
        self.history_len = int(history_len)
        self.action_horizon = int(action_horizon)
        self.hidden_dims = list(hidden_dims)
        self.dropout = float(dropout)
        self.loss_type = str(loss_type)

        self.loss_fn = self._get_loss_fn(self.loss_type)

        input_dim = self.history_len * self.state_dim + self.goal_dim
        output_dim = self.action_horizon * self.action_dim
        self.net = MLPBlock([input_dim, *self.hidden_dims, output_dim], dropout=self.dropout)

        self.config = dict(
            policy_type=self.__class__.__name__,
            state_dim=self.state_dim,
            goal_dim=self.goal_dim,
            action_dim=self.action_dim,
            history_len=self.history_len,
            action_horizon=self.action_horizon,
            hidden_dims=list(self.hidden_dims),
            dropout=self.dropout,
            loss_type=self.loss_type,
            target_representation="cumulative_delta_from_start_action_in_normalized_joint_space",
        )

    @staticmethod
    def _get_loss_fn(loss_type: str):
        loss_type = loss_type.lower()
        if loss_type == "mse":
            return torch.nn.functional.mse_loss
        if loss_type in {"smooth_l1", "huber"}:
            return torch.nn.functional.smooth_l1_loss
        raise ValueError(f"Unsupported loss_type={loss_type!r}")

    def forward(self, batch: Batch) -> Dict[str, Tensor]:
        state_history = batch["state_history"]
        goal = batch["goal"]

        if state_history.ndim != 3:
            raise ValueError(
                f"state_history must have shape [B, T, state_dim], got {tuple(state_history.shape)}"
            )
        if state_history.shape[1] != self.history_len:
            raise ValueError(
                f"Expected state history length {self.history_len}, got {state_history.shape[1]}"
            )

        flat_history = state_history.reshape(state_history.shape[0], self.history_len * self.state_dim)
        x = torch.cat([flat_history, goal], dim=-1)
        predicted_deltas = self.net(x).view(-1, self.action_horizon, self.action_dim)

        out: Dict[str, Tensor] = {
            "predicted_deltas": predicted_deltas,
        }

        action_targets = batch.get("action_targets")
        if action_targets is not None:
            if action_targets.shape[1] != self.action_horizon:
                raise ValueError(
                    f"Expected action horizon {self.action_horizon}, got {action_targets.shape[1]}"
                )
            out["loss"] = self.loss_fn(predicted_deltas, action_targets)

        return out



JOINT_STATE_RAW_MIN = torch.tensor([-1.5 * np.pi, -0.45, -0.9, -1.222], dtype=torch.float32)
JOINT_STATE_RAW_MAX = torch.tensor([1.5 * np.pi, 1.0, 0.3, 0.873], dtype=torch.float32)
LOSS_TYPE_CHOICES = ("mse", "smooth_l1", "huber")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a delta-target sequence MLP behavioral cloning policy.")
    parser.add_argument("--data", type=str, required=True, help="Path to the HDF5 segment store.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./bc_component/outputs/sequence_policy_mlp_delta",
        help="Directory for checkpoints and logs.",
    )
    parser.add_argument("--train-ratio", type=float, default=0.9, help="Segment-level train split ratio.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=512, help="GPU-side batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-5, help="AdamW weight decay.")
    parser.add_argument(
        "--clip-grad-norm",
        type=float,
        default=1.0,
        help="Gradient clipping max norm. Set <= 0 to disable.",
    )
    parser.add_argument("--num-workers", type=int, default=0, help="Ignored: data is preloaded and batched on GPU.")
    parser.add_argument("--history-len", type=int, default=16, help="Length of the fixed history window.")
    parser.add_argument("--action-horizon", type=int, default=8, help="Number of future targets to predict.")
    parser.add_argument(
        "--mlp-hidden-dims",
        type=int,
        nargs="+",
        default=[128, 128, 128],
        help="Hidden layer sizes for the delta MLP policy.",
    )
    parser.add_argument("--mlp-dropout", type=float, default=0.0, help="Dropout for the delta MLP policy.")
    parser.add_argument(
        "--mlp-loss-type",
        type=str,
        default="smooth_l1",
        choices=LOSS_TYPE_CHOICES,
        help="Loss type for the delta MLP policy.",
    )
    return parser.parse_args()


@dataclass
class SplitSummary:
    total_segments: int
    accepted_segments: int
    rejected_segments: int
    train_segments: int
    val_segments: int
    train_samples: int
    val_samples: int


@dataclass
class MaterializedSequenceSplit:
    split_name: str
    state_histories: Tensor
    history_masks: Tensor
    goals: Tensor
    action_targets: Tensor
    start_actions: Tensor
    absolute_targets: Tensor
    segment_ids: Tensor
    timesteps: Tensor
    rejected_segments: List[Tuple[int, int]]
    state_dim: int
    goal_dim: int
    action_dim: int
    history_len: int
    action_horizon: int

    def __len__(self) -> int:
        return int(self.state_histories.shape[0])

    @property
    def bytes(self) -> int:
        return (
            self.state_histories.numel() * self.state_histories.element_size()
            + self.history_masks.numel() * self.history_masks.element_size()
            + self.goals.numel() * self.goals.element_size()
            + self.action_targets.numel() * self.action_targets.element_size()
            + self.start_actions.numel() * self.start_actions.element_size()
            + self.absolute_targets.numel() * self.absolute_targets.element_size()
            + self.segment_ids.numel() * self.segment_ids.element_size()
            + self.timesteps.numel() * self.timesteps.element_size()
        )


class DeviceSequenceSplit:
    def __init__(self, split: MaterializedSequenceSplit, device: torch.device) -> None:
        self.split_name = split.split_name
        self.state_histories = split.state_histories.to(device=device, non_blocking=False)
        self.history_masks = split.history_masks.to(device=device, non_blocking=False)
        self.goals = split.goals.to(device=device, non_blocking=False)
        self.action_targets = split.action_targets.to(device=device, non_blocking=False)
        self.start_actions = split.start_actions.to(device=device, non_blocking=False)
        self.absolute_targets = split.absolute_targets.to(device=device, non_blocking=False)
        self.segment_ids = split.segment_ids.to(device=device, non_blocking=False)
        self.timesteps = split.timesteps.to(device=device, non_blocking=False)
        self.rejected_segments = split.rejected_segments
        self.state_dim = split.state_dim
        self.goal_dim = split.goal_dim
        self.action_dim = split.action_dim
        self.history_len = split.history_len
        self.action_horizon = split.action_horizon
        self.device = device

    def __len__(self) -> int:
        return int(self.state_histories.shape[0])

    @property
    def bytes(self) -> int:
        return (
            self.state_histories.numel() * self.state_histories.element_size()
            + self.history_masks.numel() * self.history_masks.element_size()
            + self.goals.numel() * self.goals.element_size()
            + self.action_targets.numel() * self.action_targets.element_size()
            + self.start_actions.numel() * self.start_actions.element_size()
            + self.absolute_targets.numel() * self.absolute_targets.element_size()
            + self.segment_ids.numel() * self.segment_ids.element_size()
            + self.timesteps.numel() * self.timesteps.element_size()
        )

    def iter_batches(self, batch_size: int, shuffle: bool) -> Iterator[Dict[str, Tensor]]:
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        if len(self) == 0:
            return

        if shuffle:
            indices = torch.randperm(len(self), device=self.device)
        else:
            indices = torch.arange(len(self), device=self.device)

        for start in range(0, len(self), batch_size):
            batch_idx = indices[start : start + batch_size]
            yield {
                "state_history": self.state_histories.index_select(0, batch_idx),
                "history_mask": self.history_masks.index_select(0, batch_idx),
                "goal": self.goals.index_select(0, batch_idx),
                "action_targets": self.action_targets.index_select(0, batch_idx),
                "start_action": self.start_actions.index_select(0, batch_idx),
                "absolute_targets": self.absolute_targets.index_select(0, batch_idx),
                "segment_idx": self.segment_ids.index_select(0, batch_idx),
                "t": self.timesteps.index_select(0, batch_idx),
            }


def format_bytes(num_bytes: int) -> str:
    return f"{num_bytes / 1024**3:.3f} GiB"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_joint_tensor(x: Tensor, joint_raw_min: Tensor, joint_raw_max: Tensor) -> Tensor:
    x = x.float()
    return 2.0 * (x - joint_raw_min) / (joint_raw_max - joint_raw_min) - 1.0


def materialize_sequence_split(
    path: str | Path,
    segment_indices: Iterable[int],
    split_name: str,
    history_len: int,
    action_horizon: int,
    joint_raw_min: Tensor,
    joint_raw_max: Tensor,
    goal_norm: Normalize | None = None,
) -> MaterializedSequenceSplit:
    segment_indices = list(segment_indices)
    joint_raw_min = joint_raw_min.detach().clone().float()
    joint_raw_max = joint_raw_max.detach().clone().float()
    rejected_segments: List[Tuple[int, int]] = []

    if history_len <= 0:
        raise ValueError(f"history_len must be positive, got {history_len}")
    if action_horizon <= 0:
        raise ValueError(f"action_horizon must be positive, got {action_horizon}")
    if joint_raw_min.ndim != 1 or joint_raw_max.ndim != 1:
        raise ValueError("joint_raw_min and joint_raw_max must be rank-1 tensors")
    if joint_raw_min.shape != joint_raw_max.shape:
        raise ValueError("joint_raw_min and joint_raw_max must have the same shape")
    if not torch.all(joint_raw_max > joint_raw_min):
        raise ValueError("Every entry in joint_raw_max must be strictly greater than joint_raw_min.")

    history_rows: List[Tensor] = []
    history_mask_rows: List[Tensor] = []
    delta_target_rows: List[Tensor] = []
    absolute_target_rows: List[Tensor] = []
    goal_rows: List[Tensor] = []
    start_action_rows: List[Tensor] = []
    segment_id_rows: List[Tensor] = []
    timestep_rows: List[Tensor] = []

    with h5py.File(path, "r") as f:
        goals = f["goals"]
        trajectories = f["trajectories"]
        lengths = f["lengths"]

        if trajectories.ndim != 3:
            raise ValueError(
                f"Expected trajectories with shape (S, N, state_dim), got {tuple(trajectories.shape)}"
            )

        state_dim = int(trajectories.shape[2])
        action_dim = state_dim
        if joint_raw_min.shape != (state_dim,) or joint_raw_max.shape != (state_dim,):
            raise ValueError(
                "Normalization constants must match trajectory feature dimension: "
                f"expected {(state_dim,)}, got {tuple(joint_raw_min.shape)}"
            )

        sample_goal = torch.as_tensor(goals[0])
        goal_dim = int(sample_goal.numel())
        if goal_dim != 9:
            raise ValueError(
                f"Expected goals to flatten to size 9, got stored shape {tuple(goals.shape[1:])}"
            )

        for seg_idx in segment_indices:
            seg_len = int(lengths[seg_idx])
            sample_count = max(0, seg_len - action_horizon)
            if sample_count <= 0:
                rejected_segments.append((seg_idx, seg_len))
                warnings.warn(
                    f"[{split_name}] rejecting segment {seg_idx} with length {seg_len}: "
                    f"need at least {action_horizon + 1} steps for a full future target chunk.",
                    stacklevel=2,
                )
                continue

            traj = torch.from_numpy(trajectories[seg_idx, :seg_len]).float()
            traj = normalize_joint_tensor(traj, joint_raw_min=joint_raw_min, joint_raw_max=joint_raw_max)

            goal = torch.from_numpy(goals[seg_idx]).float().reshape(3, 3)
            if goal_norm is not None:
                goal = goal_norm(goal.t().unsqueeze(-1)).squeeze(-1).t()
            goal_row = goal.reshape(-1)

            for t in range(sample_count):
                start_idx = max(0, t - history_len + 1)
                history_valid = traj[start_idx : t + 1]
                valid_len = int(history_valid.shape[0])

                history = torch.zeros((history_len, state_dim), dtype=torch.float32)
                history_mask = torch.zeros((history_len,), dtype=torch.bool)
                history[-valid_len:] = history_valid
                history_mask[-valid_len:] = True

                start_action = traj[t]
                future_abs = traj[t + 1 : t + 1 + action_horizon]
                future_delta = future_abs - start_action.unsqueeze(0)

                history_rows.append(history.unsqueeze(0))
                history_mask_rows.append(history_mask.unsqueeze(0))
                delta_target_rows.append(future_delta.unsqueeze(0))
                absolute_target_rows.append(future_abs.unsqueeze(0))
                goal_rows.append(goal_row.unsqueeze(0))
                start_action_rows.append(start_action.unsqueeze(0))
                segment_id_rows.append(torch.tensor([seg_idx], dtype=torch.long))
                timestep_rows.append(torch.tensor([t], dtype=torch.long))

    if history_rows:
        state_histories = torch.cat(history_rows, dim=0).contiguous()
        history_masks = torch.cat(history_mask_rows, dim=0).contiguous()
        action_targets = torch.cat(delta_target_rows, dim=0).contiguous()
        absolute_targets = torch.cat(absolute_target_rows, dim=0).contiguous()
        goals_tensor = torch.cat(goal_rows, dim=0).contiguous()
        start_actions = torch.cat(start_action_rows, dim=0).contiguous()
        segment_ids = torch.cat(segment_id_rows, dim=0).contiguous()
        timesteps = torch.cat(timestep_rows, dim=0).contiguous()
    else:
        state_histories = torch.empty((0, history_len, state_dim), dtype=torch.float32)
        history_masks = torch.empty((0, history_len), dtype=torch.bool)
        action_targets = torch.empty((0, action_horizon, action_dim), dtype=torch.float32)
        absolute_targets = torch.empty((0, action_horizon, action_dim), dtype=torch.float32)
        goals_tensor = torch.empty((0, goal_dim), dtype=torch.float32)
        start_actions = torch.empty((0, action_dim), dtype=torch.float32)
        segment_ids = torch.empty((0,), dtype=torch.long)
        timesteps = torch.empty((0,), dtype=torch.long)

    split = MaterializedSequenceSplit(
        split_name=split_name,
        state_histories=state_histories,
        history_masks=history_masks,
        goals=goals_tensor,
        action_targets=action_targets,
        start_actions=start_actions,
        absolute_targets=absolute_targets,
        segment_ids=segment_ids,
        timesteps=timesteps,
        rejected_segments=rejected_segments,
        state_dim=state_dim,
        goal_dim=goal_dim,
        action_dim=action_dim,
        history_len=history_len,
        action_horizon=action_horizon,
    )
    print(f"[{split_name}] loaded {len(split)} samples into RAM ({format_bytes(split.bytes)})")
    return split


def build_segment_split(
    path: str | Path,
    train_ratio: float,
    seed: int,
    action_horizon: int,
) -> Tuple[List[int], List[int], SplitSummary]:
    if not 0.0 < train_ratio < 1.0:
        raise ValueError(f"train_ratio must be in (0, 1), got {train_ratio}")

    with h5py.File(path, "r") as f:
        num_segments = int(f["goals"].shape[0])
        lengths = f["lengths"][:]

    rejected_segments = [idx for idx, seg_len in enumerate(lengths) if int(seg_len) <= action_horizon]
    accepted_segments = [idx for idx, seg_len in enumerate(lengths) if int(seg_len) > action_horizon]

    rng = random.Random(seed)
    rng.shuffle(accepted_segments)

    if len(accepted_segments) == 0:
        raise RuntimeError(
            f"No usable segments found: every segment has length <= action_horizon ({action_horizon})."
        )

    train_count = int(math.floor(len(accepted_segments) * train_ratio))
    if train_count <= 0:
        train_count = 1
    if train_count >= len(accepted_segments) and len(accepted_segments) > 1:
        train_count = len(accepted_segments) - 1

    train_segments = accepted_segments[:train_count]
    val_segments = accepted_segments[train_count:]

    if not val_segments:
        warnings.warn(
            "Validation split is empty after the segment-level split. "
            "Consider using more data or a different train_ratio.",
            stacklevel=2,
        )

    train_samples = int(sum(max(0, int(lengths[idx]) - action_horizon) for idx in train_segments))
    val_samples = int(sum(max(0, int(lengths[idx]) - action_horizon) for idx in val_segments))

    summary = SplitSummary(
        total_segments=num_segments,
        accepted_segments=len(accepted_segments),
        rejected_segments=len(rejected_segments),
        train_segments=len(train_segments),
        val_segments=len(val_segments),
        train_samples=train_samples,
        val_samples=val_samples,
    )
    return train_segments, val_segments, summary


def compute_goal_normalization_stats(
    path: str | Path,
    train_segments: Iterable[int],
    action_horizon: int,
    eps: float = 1e-6,
) -> Dict[str, Tensor]:
    goal_rows: List[Tensor] = []

    with h5py.File(path, "r") as f:
        goals = f["goals"]
        lengths = f["lengths"]

        for seg_idx in train_segments:
            seg_len = int(lengths[seg_idx])
            if seg_len <= action_horizon:
                continue

            goal = torch.from_numpy(goals[seg_idx]).float().reshape(3, 3)
            goal_rows.append(goal)

    if not goal_rows:
        raise RuntimeError("No usable train segments for goal normalization stats.")

    goal_all = torch.stack(goal_rows, dim=0)
    goal_mean = goal_all.mean(dim=(0, 1))
    goal_std = goal_all.std(dim=(0, 1), unbiased=False).clamp_min(eps)

    return {
        "goal_mean": goal_mean,
        "goal_std": goal_std,
    }


def build_model(args: argparse.Namespace, split: MaterializedSequenceSplit) -> DeltaGoalConditionedMLPPolicy:
    return DeltaGoalConditionedMLPPolicy(
        state_dim=split.state_dim,
        goal_dim=split.goal_dim,
        action_dim=split.action_dim,
        history_len=args.history_len,
        action_horizon=args.action_horizon,
        hidden_dims=args.mlp_hidden_dims,
        dropout=args.mlp_dropout,
        loss_type=args.mlp_loss_type,
    )


@torch.no_grad()
def evaluate(
    model: DeltaGoalConditionedMLPPolicy,
    split: DeviceSequenceSplit,
    batch_size: int,
    epoch: int,
    total_epochs: int,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_abs_mae = 0.0
    total_count = 0

    total_batches = math.ceil(len(split) / batch_size) if len(split) > 0 else 0
    progress = tqdm(
        split.iter_batches(batch_size=batch_size, shuffle=True),
        total=total_batches,
        desc=f"val {epoch:03d}/{total_epochs:03d}",
        leave=False,
    )

    for batch in progress:
        out = model(batch)
        loss = out["loss"]
        predicted_deltas = out["predicted_deltas"]
        predicted_abs = batch["start_action"].unsqueeze(1) + predicted_deltas
        abs_mae = torch.mean(torch.abs(predicted_abs - batch["absolute_targets"]))

        batch_size_actual = int(batch["action_targets"].shape[0])
        total_loss += float(loss.detach().item()) * batch_size_actual
        total_abs_mae += float(abs_mae.detach().item()) * batch_size_actual
        total_count += batch_size_actual

        progress.set_postfix(loss=f"{float(loss.detach().item()):.5f}", abs_mae=f"{float(abs_mae.detach().item()):.5f}")

    if total_count == 0:
        return {"loss": float("nan"), "abs_mae": float("nan")}

    return {
        "loss": total_loss / total_count,
        "abs_mae": total_abs_mae / total_count,
    }


def train_one_epoch(
    model: DeltaGoalConditionedMLPPolicy,
    split: DeviceSequenceSplit,
    batch_size: int,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    total_epochs: int,
    clip_grad_norm: float | None,
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    total_abs_mae = 0.0
    total_count = 0

    total_batches = math.ceil(len(split) / batch_size) if len(split) > 0 else 0
    progress = tqdm(
        split.iter_batches(batch_size=batch_size, shuffle=True),
        total=total_batches,
        desc=f"train {epoch:03d}/{total_epochs:03d}",
        leave=False,
    )

    for batch in progress:
        optimizer.zero_grad(set_to_none=True)
        out = model(batch)
        loss = out["loss"]
        predicted_deltas = out["predicted_deltas"]
        predicted_abs = batch["start_action"].unsqueeze(1) + predicted_deltas
        abs_mae = torch.mean(torch.abs(predicted_abs - batch["absolute_targets"]))

        loss.backward()
        if clip_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
        optimizer.step()

        batch_size_actual = int(batch["action_targets"].shape[0])
        total_loss += float(loss.detach().item()) * batch_size_actual
        total_abs_mae += float(abs_mae.detach().item()) * batch_size_actual
        total_count += batch_size_actual

        progress.set_postfix(loss=f"{float(loss.detach().item()):.5f}", abs_mae=f"{float(abs_mae.detach().item()):.5f}")

    if total_count == 0:
        raise RuntimeError("Training split produced zero samples.")

    return {
        "loss": total_loss / total_count,
        "abs_mae": total_abs_mae / total_count,
    }


def save_checkpoint(
    path: Path,
    model: DeltaGoalConditionedMLPPolicy,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    history: List[Dict[str, float]],
    config: Dict[str, object],
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "history": history,
            "config": config,
        },
        path,
    )


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    seed_everything(args.seed)

    if not torch.cuda.is_available():
        raise RuntimeError("This VRAM-first trainer requires CUDA, but CUDA is not available.")
    device = torch.device("cuda")

    train_segments, val_segments, split_summary = build_segment_split(
        path=args.data,
        train_ratio=args.train_ratio,
        seed=args.seed,
        action_horizon=args.action_horizon,
    )

    goal_stats = compute_goal_normalization_stats(
        path=args.data,
        train_segments=train_segments,
        action_horizon=args.action_horizon,
    )
    goal_norm = Normalize(
        mean=goal_stats["goal_mean"].tolist(),
        std=goal_stats["goal_std"].tolist(),
    )

    train_split = materialize_sequence_split(
        path=args.data,
        segment_indices=train_segments,
        split_name="train",
        history_len=args.history_len,
        action_horizon=args.action_horizon,
        joint_raw_min=JOINT_STATE_RAW_MIN,
        joint_raw_max=JOINT_STATE_RAW_MAX,
        goal_norm=goal_norm,
    )
    val_split = materialize_sequence_split(
        path=args.data,
        segment_indices=val_segments,
        split_name="val",
        history_len=args.history_len,
        action_horizon=args.action_horizon,
        joint_raw_min=JOINT_STATE_RAW_MIN,
        joint_raw_max=JOINT_STATE_RAW_MAX,
        goal_norm=goal_norm,
    )

    print("Segment split summary:")
    print(json.dumps(asdict(split_summary), indent=2))
    print("Goal normalization stats (train split only):")
    print(
        json.dumps(
            {
                "goal_mean_grouped": goal_stats["goal_mean"].tolist(),
                "goal_std_grouped": goal_stats["goal_std"].tolist(),
            },
            indent=2,
        )
    )
    print("Joint scaling constants (raw -> [-1, 1]):")
    print(
        json.dumps(
            {
                "joint_raw_min": JOINT_STATE_RAW_MIN.tolist(),
                "joint_raw_max": JOINT_STATE_RAW_MAX.tolist(),
            },
            indent=2,
        )
    )
    print(
        f"Constructed datasets: train_samples={len(train_split)}, val_samples={len(val_split)}, "
        f"train_rejected={len(train_split.rejected_segments)}, val_rejected={len(val_split.rejected_segments)}"
    )
    if args.num_workers != 0:
        print(f"Ignoring --num-workers={args.num_workers} because batching is done directly on the GPU.")

    print(f"[train] moving full split to {device} ({format_bytes(train_split.bytes)})")
    train_split_dev = DeviceSequenceSplit(train_split, device=device)
    print(f"[val] moving full split to {device} ({format_bytes(val_split.bytes)})")
    val_split_dev = DeviceSequenceSplit(val_split, device=device)
    print(f"[train] on-device footprint: {format_bytes(train_split_dev.bytes)}")
    print(f"[val] on-device footprint: {format_bytes(val_split_dev.bytes)}")

    model = build_model(args, train_split).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    clip_grad_norm = None if args.clip_grad_norm <= 0 else float(args.clip_grad_norm)

    config: Dict[str, object] = {
        "data": str(args.data),
        "output_dir": str(output_dir),
        "model_choice": "mlp_delta",
        "seed": args.seed,
        "train_ratio": args.train_ratio,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "clip_grad_norm": clip_grad_norm,
        "device": str(device),
        "history_len": args.history_len,
        "action_horizon": args.action_horizon,
        "model": dict(model.config),
        "normalization": {
            "joint_mode": "raw_joint_limits_to_-1_1",
            "joint_raw_min": JOINT_STATE_RAW_MIN.tolist(),
            "joint_raw_max": JOINT_STATE_RAW_MAX.tolist(),
            "goal_enabled": True,
            "goal_library": "torchvision.transforms.Normalize",
            "goal_mean_grouped": goal_stats["goal_mean"].tolist(),
            "goal_std_grouped": goal_stats["goal_std"].tolist(),
            "goal_mode": "grouped_by_coordinate_across_3_entries",
            "target_uses_joint_scaling": True,
            "delta_reference_space": "normalized_joint_space",
        },
        "sample_formation": {
            "anchor_definition": "history ends at t, targets are t+1..t+action_horizon",
            "history_padding": "left_zero_pad",
            "history_mask": True,
            "future_padding": False,
            "future_requirement": "full_action_horizon_required",
            "start_action": "trajectory[t]",
            "target_representation": "cumulative_delta_from_start_action",
        },
        "split_summary": asdict(split_summary),
        "cli_args": vars(args),
    }

    print(f"Using device: {device}")
    print(f"CUDA device: {torch.cuda.get_device_name(device)}")
    print(f"Model config: {json.dumps(model.config, indent=2)}")

    history: List[Dict[str, float]] = []
    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(
            model=model,
            split=train_split_dev,
            batch_size=args.batch_size,
            optimizer=optimizer,
            epoch=epoch,
            total_epochs=args.epochs,
            clip_grad_norm=clip_grad_norm,
        )
        val_metrics = (
            evaluate(
                model=model,
                split=val_split_dev,
                batch_size=args.batch_size,
                epoch=epoch,
                total_epochs=args.epochs,
            )
            if len(val_split_dev) > 0
            else {"loss": float("nan"), "abs_mae": float("nan")}
        )

        epoch_record = {
            "epoch": float(epoch),
            "train_loss": train_metrics["loss"],
            "train_abs_mae": train_metrics["abs_mae"],
            "val_loss": val_metrics["loss"],
            "val_abs_mae": val_metrics["abs_mae"],
        }
        history.append(epoch_record)

        print(
            f"epoch {epoch:03d} | "
            f"train_loss={train_metrics['loss']:.6f} | train_abs_mae={train_metrics['abs_mae']:.6f} | "
            f"val_loss={val_metrics['loss']:.6f} | val_abs_mae={val_metrics['abs_mae']:.6f}"
        )

        save_checkpoint(output_dir / "latest.pt", model, optimizer, epoch, history, config)
        if not math.isnan(val_metrics["loss"]) and val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            save_checkpoint(output_dir / "best.pt", model, optimizer, epoch, history, config)

    with open(output_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    print(f"Finished. Wrote outputs to: {output_dir}")
    print("Saved: latest.pt, best.pt, history.json, summary.json")


if __name__ == "__main__":
    main()
