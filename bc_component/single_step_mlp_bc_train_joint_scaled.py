"""
Single-step goal-conditioned behavioral cloning trainer.

Assumptions:
- Input data lives in an HDF5 segment store with datasets:
    goals:        (S, 9) or (S, 3, 3)
    trajectories: (S, N_max, 4)
    lengths:      (S,)
- One supervised sample is built as:
    state_t  = trajectory[t]
    target_t = trajectory[t + 1]
  so the model learns: (goal, current step) -> next step.
- Trajectories serve as both the state sequence and the action-target sequence.
- Train/val split is performed by segment, before sample extraction.
- Joint states/targets are normalized to [-1, 1] using explicit raw min/max constants.
- Goals are normalized using torchvision.transforms.Normalize with grouped statistics
  computed on the train split only.

This file defines both the model and the trainer so it can run standalone.
"""

import argparse
import json
import math
import random
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import h5py
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Normalize
import numpy as np


# Prenormalization values. The rotator is somewhat approximate and could probably be replaced with a scaled approach instead
JOINT_STATE_RAW_MIN = torch.tensor([-1.5 * np.pi, -0.45, -0.9, -1.222], dtype=torch.float32)
JOINT_STATE_RAW_MAX = torch.tensor([1.5 * np.pi, 1.0, 0.3, 0.873], dtype=torch.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a single-step goal-conditioned MLP BC policy.")
    parser.add_argument("--data", type=str, required=True, help="Path to the HDF5 segment store.")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory for checkpoints and logs.")
    parser.add_argument("--train-ratio", type=float, default=0.9, help="Segment-level train split ratio.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")

    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size. Reduce if you hit memory limits.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="AdamW weight decay.")
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout in hidden layers.")
    parser.add_argument(
        "--hidden-dims",
        type=int,
        nargs="+",
        default=[256, 256],
        help="Hidden layer sizes for the MLP.",
    )
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers.")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use: 'auto', 'cuda', or 'cpu'. 'auto' prefers CUDA when available.",
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


class SingleStepGoalConditionedMLP(nn.Module):
    """
    Simple pointwise MLP baseline.

    Maps:
        (goal, current_state) -> next_state

    Batch convention:
        batch = {
            "state": Tensor[B, state_dim],
            "goal": Tensor[B, goal_dim],
            "action_targets": Tensor[B, action_dim],
        }
    """

    def __init__(
        self,
        state_dim: int,
        goal_dim: int,
        action_dim: int,
        hidden_dims: Sequence[int] = (256, 256),
        dropout: float = 0.0,
        loss_func=F.mse_loss,
    ) -> None:
        super().__init__()

        self.state_dim = state_dim
        self.goal_dim = goal_dim
        self.action_dim = action_dim
        self.hidden_dims = list(hidden_dims)
        self.dropout = dropout
        self.loss_func = loss_func

        dims = [state_dim + goal_dim, *hidden_dims, action_dim]
        layers: List[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
                if dropout > 0.0:
                    layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def train_step(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        states = batch["state"]
        goals = batch["goal"]

        predicted_actions = self.forward(states, goals)

        out = {"predicted_actions": predicted_actions}
        action_targets = batch.get("action_targets")
        if action_targets is not None:
            loss = self.loss_func(predicted_actions, action_targets)
            out["loss"] = loss
        return out

    def forward(self, state: Tensor, goal: Tensor) -> Tensor:
        x = torch.cat([state, goal], dim=-1)
        predicted_actions = self.net(x)
        return predicted_actions


class SingleStepSegmentDataset(Dataset):
    """
    Segment-backed dataset that yields one-step transitions:
        state_t -> target_{t+1}

    Split is performed at the segment level before constructing the sample index.
    Segments with fewer than 2 valid steps are rejected, with explicit warnings.

    Normalization is applied in the dataset:
      - trajectories are scaled from raw joint ranges into [-1, 1]
      - goals are normalized in grouped form over shape (3, 3), where the 3 coordinate
        dimensions share statistics across the 3 goal entries
    """

    def __init__(
        self,
        path: str | Path,
        segment_indices: Iterable[int],
        split_name: str,
        joint_raw_min: Tensor,
        joint_raw_max: Tensor,
        goal_norm: Normalize | None = None,
    ) -> None:
        self.path = str(path)
        self.segment_indices = list(segment_indices)
        self.split_name = split_name
        self.joint_raw_min = joint_raw_min.detach().clone().float()
        self.joint_raw_max = joint_raw_max.detach().clone().float()
        self.goal_norm = goal_norm
        self._file = None
        self.sample_index: List[Tuple[int, int]] = []
        self.rejected_segments: List[Tuple[int, int]] = []

        if self.joint_raw_min.shape != (4,) or self.joint_raw_max.shape != (4,):
            raise ValueError(
                f"joint_raw_min and joint_raw_max must each have shape (4,), got "
                f"{tuple(self.joint_raw_min.shape)} and {tuple(self.joint_raw_max.shape)}"
            )
        if not torch.all(self.joint_raw_max > self.joint_raw_min):
            raise ValueError("Every entry in joint_raw_max must be strictly greater than joint_raw_min.")

        with h5py.File(self.path, "r") as f:
            goals = f["goals"]
            trajectories = f["trajectories"]
            lengths = f["lengths"]

            if trajectories.ndim != 3 or trajectories.shape[2] != 4:
                raise ValueError(
                    f"Expected trajectories with shape (S, N, 4), got {tuple(trajectories.shape)}"
                )

            for seg_idx in self.segment_indices:
                seg_len = int(lengths[seg_idx])
                if seg_len < 2:
                    self.rejected_segments.append((seg_idx, seg_len))
                    warnings.warn(
                        f"[{self.split_name}] rejecting segment {seg_idx} with length {seg_len}: "
                        "need at least 2 steps for single-step input/target pairs.",
                        stacklevel=2,
                    )
                    continue

                for t in range(seg_len - 1):
                    self.sample_index.append((seg_idx, t))

            sample_goal = goals[0]
            flat_dim = int(torch.as_tensor(sample_goal).numel())
            if flat_dim != 9:
                raise ValueError(
                    f"Expected goals to flatten to size 9, got stored shape {tuple(goals.shape[1:])}"
                )
            self.goal_dim = flat_dim
            self.state_dim = 4
            self.action_dim = 4

    def __len__(self) -> int:
        return len(self.sample_index)

    def _ensure_open(self):
        if self._file is None:
            self._file = h5py.File(self.path, "r")
        return self._file

    def _normalize_joint_vec(self, x: Tensor) -> Tensor:
        # Map raw joint values from [min, max] to [-1, 1] per dimension.
        x = x.float()
        return 2.0 * (x - self.joint_raw_min) / (self.joint_raw_max - self.joint_raw_min) - 1.0

    def _normalize_goal(self, goal: Tensor) -> Tensor:
        goal = goal.reshape(3, 3)
        if self.goal_norm is None:
            return goal.reshape(-1)

        # Treat coordinate index as channel dimension:
        # (entries=3, coords=3) -> (coords=3, entries=3, width=1)
        g = goal.t().unsqueeze(-1)
        g = self.goal_norm(g)
        return g.squeeze(-1).t().reshape(-1)

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        f = self._ensure_open()
        seg_idx, t = self.sample_index[idx]

        goal = torch.from_numpy(f["goals"][seg_idx]).float()
        state = torch.from_numpy(f["trajectories"][seg_idx, t]).float()
        target = torch.from_numpy(f["trajectories"][seg_idx, t + 1]).float()

        goal = self._normalize_goal(goal)
        state = self._normalize_joint_vec(state)
        target = self._normalize_joint_vec(target)

        return {
            "goal": goal,
            "state": state,
            "action_targets": target,
            "segment_idx": torch.tensor(seg_idx, dtype=torch.long),
            "t": torch.tensor(t, dtype=torch.long),
        }


def choose_device(requested: str) -> torch.device:
    requested = requested.lower()
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        return torch.device("cuda")
    if requested == "cpu":
        return torch.device("cpu")
    raise ValueError(f"Unsupported device option: {requested!r}")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_segment_split(path: str | Path, train_ratio: float, seed: int) -> Tuple[List[int], List[int], SplitSummary]:
    if not 0.0 < train_ratio < 1.0:
        raise ValueError(f"train_ratio must be in (0, 1), got {train_ratio}")

    with h5py.File(path, "r") as f:
        num_segments = int(f["goals"].shape[0])
        lengths = f["lengths"][:]

    rejected_segments = [idx for idx, seg_len in enumerate(lengths) if int(seg_len) < 2]
    accepted_segments = [idx for idx, seg_len in enumerate(lengths) if int(seg_len) >= 2]

    rng = random.Random(seed)
    rng.shuffle(accepted_segments)

    if len(accepted_segments) == 0:
        raise RuntimeError("No usable segments found: every segment has length < 2.")

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

    train_samples = int(sum(max(0, int(lengths[idx]) - 1) for idx in train_segments))
    val_samples = int(sum(max(0, int(lengths[idx]) - 1) for idx in val_segments))

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
    eps: float = 1e-6,
) -> Dict[str, Tensor]:
    """
    Compute train-split-only grouped goal normalization stats.

    Goals are treated as logical shape (3, 3), and statistics are pooled by
    coordinate index across the 3 goal entries.
    """
    goal_rows: List[Tensor] = []

    with h5py.File(path, "r") as f:
        goals = f["goals"]
        lengths = f["lengths"]

        for seg_idx in train_segments:
            seg_len = int(lengths[seg_idx])
            if seg_len < 2:
                continue

            goal = torch.from_numpy(goals[seg_idx]).float().reshape(3, 3)
            goal_rows.append(goal)

    if not goal_rows:
        raise RuntimeError("No usable train segments for goal normalization stats.")

    goal_all = torch.stack(goal_rows, dim=0)    # (S_train, 3, 3)
    goal_mean = goal_all.mean(dim=(0, 1))
    goal_std = goal_all.std(dim=(0, 1), unbiased=False).clamp_min(eps)

    return {
        "goal_mean": goal_mean,
        "goal_std": goal_std,
    }


def move_batch_to_device(batch: Dict[str, Tensor], device: torch.device) -> Dict[str, Tensor]:
    moved: Dict[str, Tensor] = {}
    for key, value in batch.items():
        moved[key] = value.to(device, non_blocking=True)
    return moved


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_mae = 0.0
    total_count = 0

    for batch in loader:
        batch = move_batch_to_device(batch, device)
        out = model.train_step(batch)
        loss = out["loss"]
        preds = out["predicted_actions"]
        targets = batch["action_targets"]
        mae = F.l1_loss(preds, targets)

        batch_size = targets.shape[0]
        total_loss += float(loss.item()) * batch_size
        total_mae += float(mae.item()) * batch_size
        total_count += batch_size

    if total_count == 0:
        return {"loss": float("nan"), "mae": float("nan")}

    return {
        "loss": total_loss / total_count,
        "mae": total_mae / total_count,
    }


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    total_mae = 0.0
    total_count = 0

    for batch in loader:
        batch = move_batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        out = model.train_step(batch)
        loss = out["loss"]
        preds = out["predicted_actions"]
        targets = batch["action_targets"]
        mae = F.l1_loss(preds, targets)

        loss.backward()
        optimizer.step()

        batch_size = targets.shape[0]
        total_loss += float(loss.item()) * batch_size
        total_mae += float(mae.item()) * batch_size
        total_count += batch_size

    if total_count == 0:
        raise RuntimeError("Training loader produced zero samples.")

    return {
        "loss": total_loss / total_count,
        "mae": total_mae / total_count,
    }


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
    device = choose_device(args.device)

    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    train_segments, val_segments, split_summary = build_segment_split(
        path=args.data,
        train_ratio=args.train_ratio,
        seed=args.seed,
    )

    goal_stats = compute_goal_normalization_stats(args.data, train_segments)
    goal_norm = Normalize(
        mean=goal_stats["goal_mean"].tolist(),
        std=goal_stats["goal_std"].tolist(),
    )

    train_dataset = SingleStepSegmentDataset(
        args.data,
        train_segments,
        split_name="train",
        joint_raw_min=JOINT_STATE_RAW_MIN,
        joint_raw_max=JOINT_STATE_RAW_MAX,
        goal_norm=goal_norm,
    )
    val_dataset = SingleStepSegmentDataset(
        args.data,
        val_segments,
        split_name="val",
        joint_raw_min=JOINT_STATE_RAW_MIN,
        joint_raw_max=JOINT_STATE_RAW_MAX,
        goal_norm=goal_norm,
    )

    print("Segment split summary:")
    print(json.dumps(asdict(split_summary), indent=2))
    print("Goal normalization stats (train split only):")
    print(json.dumps({
        "goal_mean_grouped": goal_stats["goal_mean"].tolist(),
        "goal_std_grouped": goal_stats["goal_std"].tolist(),
    }, indent=2))
    print("Joint scaling constants (raw -> [-1, 1]):")
    print(json.dumps({
        "joint_raw_min": JOINT_STATE_RAW_MIN.tolist(),
        "joint_raw_max": JOINT_STATE_RAW_MAX.tolist(),
    }, indent=2))
    print(
        f"Constructed datasets: train_samples={len(train_dataset)}, val_samples={len(val_dataset)}, "
        f"train_rejected={len(train_dataset.rejected_segments)}, val_rejected={len(val_dataset.rejected_segments)}"
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    model = SingleStepGoalConditionedMLP(
        state_dim=train_dataset.state_dim,
        goal_dim=train_dataset.goal_dim,
        action_dim=train_dataset.action_dim,
        hidden_dims=args.hidden_dims,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    config: Dict[str, object] = {
        "data": str(args.data),
        "output_dir": str(output_dir),
        "train_ratio": args.train_ratio,
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "dropout": args.dropout,
        "hidden_dims": list(args.hidden_dims),
        "device": str(device),
        "model": {
            "state_dim": train_dataset.state_dim,
            "goal_dim": train_dataset.goal_dim,
            "action_dim": train_dataset.action_dim,
            "hidden_dims": list(args.hidden_dims),
            "dropout": args.dropout,
            "loss_type": "mse",
        },
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
        },
        "notes": {
            "mapping": "(goal, trajectory[t]) -> trajectory[t+1]",
            "joint_scaling_todo": "Replace JOINT_STATE_RAW_MIN and JOINT_STATE_RAW_MAX with the true raw joint limits before serious training.",
            "future_batch_size": "for real GPU training, batch sizes like 512-4096 may be reasonable depending on dataset size and GPU memory",
            "future_epochs": "for a stronger baseline, 20-100 epochs is more typical than 10",
        },
        "split_summary": asdict(split_summary),
    }

    history: List[Dict[str, float]] = []
    best_val_loss = float("inf")

    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"CUDA device: {torch.cuda.get_device_name(device)}")

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, device)
        val_metrics = evaluate(model, val_loader, device) if len(val_dataset) > 0 else {"loss": float("nan"), "mae": float("nan")}

        epoch_record = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_mae": train_metrics["mae"],
            "val_loss": val_metrics["loss"],
            "val_mae": val_metrics["mae"],
        }
        history.append(epoch_record)

        print(
            f"epoch {epoch:03d} | "
            f"train_loss (mse)={train_metrics['loss']:.6f} train_mae={train_metrics['mae']:.6f} | "
            f"val_loss (mse)={val_metrics['loss']:.6f} val_mae={val_metrics['mae']:.6f}"
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
