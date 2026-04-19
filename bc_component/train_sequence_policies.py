"""
Sequence goal-conditioned behavioral cloning trainer.

This trainer supports the three sequence policies defined in models.py:
    - GoalConditionedMLPPolicy
    - LSTMSeq2SeqPolicy
    - ACTStylePolicy

Assumptions carried over from the single-step trainer:
- Input data lives in an HDF5 segment store with datasets:
    goals:        (S, 9) or (S, 3, 3)
    trajectories: (S, N_max, 4)
    lengths:      (S,)
- Trajectories serve as both the state-history source and the future target source.
- Train/val split is performed by segment, before sample extraction.
- Joint states/targets are normalized to [-1, 1] using explicit raw min/max constants.
- Goals are normalized using torchvision.transforms.Normalize with grouped statistics
  computed on the train split only.
- CUDA is assumed to be available.

Supervised sample formation chosen here:
- One sample is anchored at time index t.
- Input history is the fixed-length window ending at t:
      state_history = trajectory[max(0, t-history_len+1) : t+1]
  This is left-padded with zeros when the segment prefix is shorter than history_len.
- history_mask marks which entries in the fixed history window are valid.
- Target chunk is the next action_horizon steps:
      action_targets = trajectory[t+1 : t+1+action_horizon]
- Only anchors with a full future chunk are kept.
- start_action is set to trajectory[t], which is a reasonable decoder warm-start in
  this setting because trajectories provide both the observed state stream and the
  supervised future targets in the same joint space.

Notes:
- ACT consumes history_mask directly.
- The provided MLP/LSTM model definitions do not consume history_mask. They therefore
  see zero-padded prefix tokens. If that becomes a limitation, the model definitions
  should be updated rather than adding hidden trainer-side branching.
- For the LSTM, teacher forcing is scheduled linearly from --teacher-forcing-start to
  --teacher-forcing-end across training. This is a pragmatic compromise: strong guidance
  early for optimization stability, then reduced dependence on teacher forcing later.
  See teacher_forcing_ratio_for_epoch(...) if you want to change the schedule.
"""

import argparse
import json
import math
import random
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import h5py
import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Normalize

from models import ACTStylePolicy, GoalConditionedMLPPolicy, LSTMSeq2SeqPolicy


# Prenormalization values carried over from the single-step trainer.
JOINT_STATE_RAW_MIN = torch.tensor([-1.5 * np.pi, -0.45, -0.9, -1.222], dtype=torch.float32)
JOINT_STATE_RAW_MAX = torch.tensor([1.5 * np.pi, 1.0, 0.3, 0.873], dtype=torch.float32)

MODEL_CHOICES = ("mlp", "lstm", "act")
LOSS_TYPE_CHOICES = ("mse", "smooth_l1", "huber")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train goal-conditioned sequence BC policies.")
    parser.add_argument("--data", type=str, required=True, help="Path to the HDF5 segment store.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./bc_component/outputs/sequence_policy",
        help="Directory for checkpoints and logs.",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=MODEL_CHOICES,
        help="Which policy architecture to train.",
    )
    parser.add_argument("--train-ratio", type=float, default=0.9, help="Segment-level train split ratio.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")

    parser.add_argument("--epochs", type=int, default=200, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=512, help="Batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-5, help="AdamW weight decay.")
    parser.add_argument(
        "--clip-grad-norm",
        type=float,
        default=1.0,
        help="Gradient clipping max norm. Set <= 0 to disable.",
    )
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers.")

    parser.add_argument("--history-len", type=int, default=16, help="Length of the fixed history window.")
    parser.add_argument("--action-horizon", type=int, default=8, help="Number of future targets to predict.")

    parser.add_argument(
        "--mlp-hidden-dims",
        type=int,
        nargs="+",
        default=[512, 512],
        help="Hidden layer sizes for GoalConditionedMLPPolicy.",
    )
    parser.add_argument("--mlp-dropout", type=float, default=0.0, help="Dropout for GoalConditionedMLPPolicy.")
    parser.add_argument(
        "--mlp-loss-type",
        type=str,
        default="smooth_l1",
        choices=LOSS_TYPE_CHOICES,
        help="Loss type for GoalConditionedMLPPolicy.",
    )

    parser.add_argument(
        "--lstm-encoder-hidden-dim",
        type=int,
        default=256,
        help="Encoder hidden size for LSTMSeq2SeqPolicy.",
    )
    parser.add_argument(
        "--lstm-decoder-hidden-dim",
        type=int,
        default=256,
        help="Decoder hidden size for LSTMSeq2SeqPolicy.",
    )
    parser.add_argument("--lstm-num-layers", type=int, default=2, help="Number of LSTM layers.")
    parser.add_argument("--lstm-dropout", type=float, default=0.1, help="Dropout for LSTMSeq2SeqPolicy.")
    parser.add_argument(
        "--lstm-loss-type",
        type=str,
        default="smooth_l1",
        choices=LOSS_TYPE_CHOICES,
        help="Loss type for LSTMSeq2SeqPolicy.",
    )
    parser.add_argument(
        "--teacher-forcing-start",
        type=float,
        default=1.0,
        help="Teacher forcing ratio at the start of training for LSTMSeq2SeqPolicy.",
    )
    parser.add_argument(
        "--teacher-forcing-end",
        type=float,
        default=0.2,
        help="Teacher forcing ratio at the end of training for LSTMSeq2SeqPolicy.",
    )

    parser.add_argument("--act-d-model", type=int, default=256, help="Transformer width for ACTStylePolicy.")
    parser.add_argument("--act-latent-dim", type=int, default=32, help="Latent size for ACTStylePolicy.")
    parser.add_argument("--act-nhead", type=int, default=8, help="Number of attention heads for ACTStylePolicy.")
    parser.add_argument(
        "--act-num-encoder-layers",
        type=int,
        default=4,
        help="Number of encoder layers for ACTStylePolicy.",
    )
    parser.add_argument(
        "--act-num-decoder-layers",
        type=int,
        default=3,
        help="Number of decoder layers for ACTStylePolicy.",
    )
    parser.add_argument("--act-ff-dim", type=int, default=512, help="FFN width for ACTStylePolicy.")
    parser.add_argument("--act-dropout", type=float, default=0.1, help="Dropout for ACTStylePolicy.")
    parser.add_argument("--act-kl-weight", type=float, default=1e-4, help="KL loss weight for ACTStylePolicy.")
    parser.add_argument(
        "--act-recon-loss-type",
        type=str,
        default="smooth_l1",
        choices=LOSS_TYPE_CHOICES,
        help="Reconstruction loss type for ACTStylePolicy.",
    )
    parser.add_argument(
        "--act-sample-prior-at-inference",
        action="store_true",
        help="Sample from the latent prior at inference time instead of using zeros.",
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


class SequenceSegmentDataset(Dataset):
    """
    In-memory segment dataset for sequence-conditioned supervised learning.

    Each supervised sample contains:
        state_history:   [history_len, state_dim]
        history_mask:    [history_len] bool, True where valid
        goal:            [goal_dim]
        action_targets:  [action_horizon, action_dim]
        start_action:    [action_dim]

    A sample is anchored at time index t, where the model sees the padded history ending
    at t and predicts the next action_horizon joint targets.
    """

    def __init__(
        self,
        path: str | Path,
        segment_indices: Iterable[int],
        split_name: str,
        history_len: int,
        action_horizon: int,
        joint_raw_min: Tensor,
        joint_raw_max: Tensor,
        goal_norm: Optional[Normalize] = None,
    ) -> None:
        self.path = str(path)
        self.segment_indices = list(segment_indices)
        self.split_name = split_name
        self.history_len = int(history_len)
        self.action_horizon = int(action_horizon)
        self.joint_raw_min = joint_raw_min.detach().clone().float()
        self.joint_raw_max = joint_raw_max.detach().clone().float()
        self.goal_norm = goal_norm
        self.rejected_segments: List[Tuple[int, int]] = []

        if self.history_len <= 0:
            raise ValueError(f"history_len must be positive, got {self.history_len}")
        if self.action_horizon <= 0:
            raise ValueError(f"action_horizon must be positive, got {self.action_horizon}")
        if self.joint_raw_min.ndim != 1 or self.joint_raw_max.ndim != 1:
            raise ValueError("joint_raw_min and joint_raw_max must be rank-1 tensors")
        if self.joint_raw_min.shape != self.joint_raw_max.shape:
            raise ValueError("joint_raw_min and joint_raw_max must have the same shape")
        if not torch.all(self.joint_raw_max > self.joint_raw_min):
            raise ValueError("Every entry in joint_raw_max must be strictly greater than joint_raw_min.")

        history_rows: List[Tensor] = []
        history_mask_rows: List[Tensor] = []
        target_rows: List[Tensor] = []
        goal_rows: List[Tensor] = []
        start_action_rows: List[Tensor] = []
        segment_id_rows: List[Tensor] = []
        timestep_rows: List[Tensor] = []

        with h5py.File(self.path, "r") as f:
            goals = f["goals"]
            trajectories = f["trajectories"]
            lengths = f["lengths"]

            if trajectories.ndim != 3:
                raise ValueError(
                    f"Expected trajectories with shape (S, N, state_dim), got {tuple(trajectories.shape)}"
                )

            self.state_dim = int(trajectories.shape[2])
            self.action_dim = self.state_dim
            if self.joint_raw_min.shape != (self.state_dim,) or self.joint_raw_max.shape != (self.state_dim,):
                raise ValueError(
                    "Normalization constants must match trajectory feature dimension: "
                    f"expected {(self.state_dim,)}, got {tuple(self.joint_raw_min.shape)}"
                )

            sample_goal = torch.as_tensor(goals[0])
            self.goal_dim = int(sample_goal.numel())
            if self.goal_dim != 9:
                raise ValueError(
                    f"Expected goals to flatten to size 9, got stored shape {tuple(goals.shape[1:])}"
                )

            for seg_idx in self.segment_indices:
                seg_len = int(lengths[seg_idx])
                sample_count = max(0, seg_len - self.action_horizon)
                if sample_count <= 0:
                    self.rejected_segments.append((seg_idx, seg_len))
                    warnings.warn(
                        f"[{self.split_name}] rejecting segment {seg_idx} with length {seg_len}: "
                        f"need at least {self.action_horizon + 1} steps for a full future target chunk.",
                        stacklevel=2,
                    )
                    continue

                traj = torch.from_numpy(trajectories[seg_idx, :seg_len]).float()
                traj = self._normalize_joint_tensor(traj)
                goal = torch.from_numpy(goals[seg_idx]).float()
                goal_row = self._normalize_goal(goal)

                for t in range(sample_count):
                    history, history_mask = self._build_history_window(traj, t)
                    action_targets = traj[t + 1 : t + 1 + self.action_horizon]
                    start_action = traj[t]

                    history_rows.append(history.unsqueeze(0))
                    history_mask_rows.append(history_mask.unsqueeze(0))
                    target_rows.append(action_targets.unsqueeze(0))
                    goal_rows.append(goal_row.unsqueeze(0))
                    start_action_rows.append(start_action.unsqueeze(0))
                    segment_id_rows.append(torch.tensor([seg_idx], dtype=torch.long))
                    timestep_rows.append(torch.tensor([t], dtype=torch.long))

        if history_rows:
            self.state_histories = torch.cat(history_rows, dim=0).contiguous()
            self.history_masks = torch.cat(history_mask_rows, dim=0).contiguous()
            self.action_targets = torch.cat(target_rows, dim=0).contiguous()
            self.goals = torch.cat(goal_rows, dim=0).contiguous()
            self.start_actions = torch.cat(start_action_rows, dim=0).contiguous()
            self.segment_ids = torch.cat(segment_id_rows, dim=0).contiguous()
            self.timesteps = torch.cat(timestep_rows, dim=0).contiguous()
        else:
            self.state_histories = torch.empty((0, self.history_len, self.state_dim), dtype=torch.float32)
            self.history_masks = torch.empty((0, self.history_len), dtype=torch.bool)
            self.action_targets = torch.empty((0, self.action_horizon, self.action_dim), dtype=torch.float32)
            self.goals = torch.empty((0, self.goal_dim), dtype=torch.float32)
            self.start_actions = torch.empty((0, self.action_dim), dtype=torch.float32)
            self.segment_ids = torch.empty((0,), dtype=torch.long)
            self.timesteps = torch.empty((0,), dtype=torch.long)

        total_bytes = (
            self.state_histories.numel() * self.state_histories.element_size()
            + self.history_masks.numel() * self.history_masks.element_size()
            + self.action_targets.numel() * self.action_targets.element_size()
            + self.goals.numel() * self.goals.element_size()
            + self.start_actions.numel() * self.start_actions.element_size()
            + self.segment_ids.numel() * self.segment_ids.element_size()
            + self.timesteps.numel() * self.timesteps.element_size()
        )
        print(
            f"[{self.split_name}] loaded {len(self.state_histories)} samples into RAM "
            f"({total_bytes / 1024**3:.3f} GiB)"
        )

    def __len__(self) -> int:
        return int(self.state_histories.shape[0])

    def _normalize_joint_tensor(self, x: Tensor) -> Tensor:
        x = x.float()
        return 2.0 * (x - self.joint_raw_min) / (self.joint_raw_max - self.joint_raw_min) - 1.0

    def _normalize_goal(self, goal: Tensor) -> Tensor:
        goal = goal.reshape(3, 3)
        if self.goal_norm is None:
            return goal.reshape(-1)

        g = goal.t().unsqueeze(-1)
        g = self.goal_norm(g)
        return g.squeeze(-1).t().reshape(-1)

    def _build_history_window(self, traj: Tensor, t: int) -> Tuple[Tensor, Tensor]:
        start_idx = max(0, t - self.history_len + 1)
        history_valid = traj[start_idx : t + 1]
        valid_len = int(history_valid.shape[0])

        history = torch.zeros((self.history_len, self.state_dim), dtype=torch.float32)
        history_mask = torch.zeros((self.history_len,), dtype=torch.bool)
        history[-valid_len:] = history_valid
        history_mask[-valid_len:] = True
        return history, history_mask

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        return {
            "state_history": self.state_histories[idx],
            "history_mask": self.history_masks[idx],
            "goal": self.goals[idx],
            "action_targets": self.action_targets[idx],
            "start_action": self.start_actions[idx],
            "segment_idx": self.segment_ids[idx],
            "t": self.timesteps[idx],
        }


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def teacher_forcing_ratio_for_epoch(epoch: int, total_epochs: int, start: float, end: float) -> float:
    """
    Linear teacher forcing schedule for the LSTM decoder.

    Change this function if you want a different schedule later.
    """
    if total_epochs <= 1:
        return float(end)
    progress = float(epoch - 1) / float(total_epochs - 1)
    ratio = start + progress * (end - start)
    return float(max(0.0, min(1.0, ratio)))


def move_batch_to_device(
    batch: Dict[str, Tensor],
    device: torch.device,
    teacher_forcing_ratio: Optional[float] = None,
) -> Dict[str, Tensor]:
    out: Dict[str, Tensor | float] = {
        "state_history": batch["state_history"].to(device, non_blocking=True),
        "history_mask": batch["history_mask"].to(device, non_blocking=True),
        "goal": batch["goal"].to(device, non_blocking=True),
        "action_targets": batch["action_targets"].to(device, non_blocking=True),
        "start_action": batch["start_action"].to(device, non_blocking=True),
    }
    if teacher_forcing_ratio is not None:
        out["teacher_forcing_ratio"] = float(teacher_forcing_ratio)
    return out  # type: ignore[return-value]


def build_model(args: argparse.Namespace, dataset: SequenceSegmentDataset) -> nn.Module:
    common = dict(
        state_dim=dataset.state_dim,
        goal_dim=dataset.goal_dim,
        action_dim=dataset.action_dim,
        history_len=args.history_len,
        action_horizon=args.action_horizon,
    )

    if args.model == "mlp":
        return GoalConditionedMLPPolicy(
            **common,
            hidden_dims=args.mlp_hidden_dims,
            dropout=args.mlp_dropout,
            loss_type=args.mlp_loss_type,
        )

    if args.model == "lstm":
        return LSTMSeq2SeqPolicy(
            **common,
            encoder_hidden_dim=args.lstm_encoder_hidden_dim,
            decoder_hidden_dim=args.lstm_decoder_hidden_dim,
            num_layers=args.lstm_num_layers,
            dropout=args.lstm_dropout,
            default_teacher_forcing_ratio=args.teacher_forcing_start,
            loss_type=args.lstm_loss_type,
        )

    if args.model == "act":
        return ACTStylePolicy(
            **common,
            d_model=args.act_d_model,
            latent_dim=args.act_latent_dim,
            nhead=args.act_nhead,
            num_encoder_layers=args.act_num_encoder_layers,
            num_decoder_layers=args.act_num_decoder_layers,
            ff_dim=args.act_ff_dim,
            dropout=args.act_dropout,
            kl_weight=args.act_kl_weight,
            recon_loss_type=args.act_recon_loss_type,
            sample_prior_at_inference=args.act_sample_prior_at_inference,
        )

    raise ValueError(f"Unsupported model type: {args.model!r}")


def scalar_metrics_from_output(out: Dict[str, Tensor], model_name: str) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    for key in ("loss", "recon_loss", "kl_loss"):
        value = out.get(key)
        if isinstance(value, torch.Tensor):
            metrics[key] = float(value.detach().item())

    if model_name != "act":
        metrics.pop("recon_loss", None)
        metrics.pop("kl_loss", None)
    return metrics


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    model_name: str,
) -> Dict[str, float]:
    model.eval()
    totals: Dict[str, float] = {}
    total_count = 0

    for batch in loader:
        batch = move_batch_to_device(batch, device=device, teacher_forcing_ratio=None)
        out = model(batch)
        metrics = scalar_metrics_from_output(out, model_name=model_name)
        batch_size = int(batch["action_targets"].shape[0])

        for key, value in metrics.items():
            totals[key] = totals.get(key, 0.0) + value * batch_size
        total_count += batch_size

    if total_count == 0:
        result = {"loss": float("nan")}
        if model_name == "act":
            result["recon_loss"] = float("nan")
            result["kl_loss"] = float("nan")
        return result

    return {key: value / total_count for key, value in totals.items()}


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    model_name: str,
    teacher_forcing_ratio: Optional[float],
    clip_grad_norm: Optional[float],
) -> Dict[str, float]:
    model.train()
    totals: Dict[str, float] = {}
    total_count = 0

    for batch in loader:
        batch = move_batch_to_device(batch, device=device, teacher_forcing_ratio=teacher_forcing_ratio)
        optimizer.zero_grad(set_to_none=True)
        out = model(batch)
        loss = out.get("loss")
        if not isinstance(loss, torch.Tensor):
            raise RuntimeError("Model forward did not return a loss tensor during training.")
        loss.backward()

        if clip_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)

        optimizer.step()

        metrics = scalar_metrics_from_output(out, model_name=model_name)
        batch_size = int(batch["action_targets"].shape[0])
        for key, value in metrics.items():
            totals[key] = totals.get(key, 0.0) + value * batch_size
        total_count += batch_size

    if total_count == 0:
        raise RuntimeError("Training loader produced zero samples.")

    return {key: value / total_count for key, value in totals.items()}


def save_checkpoint(
    path: Path,
    model: nn.Module,
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

    train_dataset = SequenceSegmentDataset(
        path=args.data,
        segment_indices=train_segments,
        split_name="train",
        history_len=args.history_len,
        action_horizon=args.action_horizon,
        joint_raw_min=JOINT_STATE_RAW_MIN,
        joint_raw_max=JOINT_STATE_RAW_MAX,
        goal_norm=goal_norm,
    )
    val_dataset = SequenceSegmentDataset(
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
        f"Constructed datasets: train_samples={len(train_dataset)}, val_samples={len(val_dataset)}, "
        f"train_rejected={len(train_dataset.rejected_segments)}, val_rejected={len(val_dataset.rejected_segments)}"
    )

    if args.num_workers != 0:
        print(
            f"Ignoring --num-workers={args.num_workers} because the full dataset is preloaded into RAM."
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
    )

    model = build_model(args, train_dataset).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    config: Dict[str, object] = {
        "data": str(args.data),
        "output_dir": str(output_dir),
        "model_choice": args.model,
        "train_ratio": args.train_ratio,
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "clip_grad_norm": None if args.clip_grad_norm <= 0 else args.clip_grad_norm,
        "device": str(device),
        "history_len": args.history_len,
        "action_horizon": args.action_horizon,
        "model": getattr(model, "config", {}),
        "teacher_forcing": {
            "enabled_for_model": args.model == "lstm",
            "schedule": "linear",
            "start": args.teacher_forcing_start,
            "end": args.teacher_forcing_end,
        },
        "normalization": {
            "joint_mode": "raw_joint_limits_to_-1_1",
            "joint_raw_min": JOINT_STATE_RAW_MIN.tolist(),
            "joint_raw_max": JOINT_STATE_RAW_MAX.tolist(),
            "goal_enabled": True,
            "goal_library": "manual_grouped_normalization",
            "goal_mean_grouped": goal_stats["goal_mean"].tolist(),
            "goal_std_grouped": goal_stats["goal_std"].tolist(),
            "goal_mode": "grouped_by_coordinate_across_3_entries",
            "target_uses_joint_scaling": True,
        },
        "sample_formation": {
            "anchor_definition": "history ends at t, targets are t+1..t+action_horizon",
            "history_padding": "left_zero_pad",
            "history_mask": True,
            "future_padding": False,
            "future_requirement": "full_action_horizon_required",
            "start_action": "trajectory[t]",
        },
        "split_summary": asdict(split_summary),
        "cli_args": vars(args),
    }

    history: List[Dict[str, float]] = []
    best_val_loss = float("inf")
    clip_grad_norm = None if args.clip_grad_norm <= 0 else float(args.clip_grad_norm)

    print(f"Using device: {device}")
    print(f"CUDA device: {torch.cuda.get_device_name(device)}")
    print(f"Model config: {json.dumps(getattr(model, 'config', {}), indent=2)}")

    for epoch in range(1, args.epochs + 1):
        current_teacher_forcing = None
        if args.model == "lstm":
            current_teacher_forcing = teacher_forcing_ratio_for_epoch(
                epoch=epoch,
                total_epochs=args.epochs,
                start=args.teacher_forcing_start,
                end=args.teacher_forcing_end,
            )

        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            model_name=args.model,
            teacher_forcing_ratio=current_teacher_forcing,
            clip_grad_norm=clip_grad_norm,
        )
        val_metrics = (
            evaluate(model=model, loader=val_loader, device=device, model_name=args.model)
            if len(val_dataset) > 0
            else {"loss": float("nan")}
        )

        epoch_record: Dict[str, float] = {
            "epoch": float(epoch),
            "train_loss": train_metrics["loss"],
            "val_loss": val_metrics["loss"],
        }
        if args.model == "lstm" and current_teacher_forcing is not None:
            epoch_record["teacher_forcing_ratio"] = current_teacher_forcing
        if args.model == "act":
            epoch_record["train_recon_loss"] = train_metrics.get("recon_loss", float("nan"))
            epoch_record["val_recon_loss"] = val_metrics.get("recon_loss", float("nan"))
            epoch_record["train_kl_loss"] = train_metrics.get("kl_loss", float("nan"))
            epoch_record["val_kl_loss"] = val_metrics.get("kl_loss", float("nan"))
        history.append(epoch_record)

        message = (
            f"epoch {epoch:03d} | train_loss={train_metrics['loss']:.6f} | "
            f"val_loss={val_metrics['loss']:.6f}"
        )
        if args.model == "lstm" and current_teacher_forcing is not None:
            message += f" | teacher_forcing={current_teacher_forcing:.4f}"
        if args.model == "act":
            message += (
                f" | train_recon={train_metrics.get('recon_loss', float('nan')):.6f}"
                f" | val_recon={val_metrics.get('recon_loss', float('nan')):.6f}"
                f" | train_kl={train_metrics.get('kl_loss', float('nan')):.6f}"
                f" | val_kl={val_metrics.get('kl_loss', float('nan')):.6f}"
            )
        print(message)

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
