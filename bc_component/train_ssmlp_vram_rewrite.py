"""
Single-step goal-conditioned behavioral cloning trainer.

Rewritten for a VRAM-first training flow:
- split train/val at the segment level
- materialize every supervised sample for each split up front
- normalize once during materialization
- move the full train/val tensors onto the GPU before training begins
- batch by indexing already-resident GPU tensors
- show tqdm bar-style progress for train/val each epoch

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
- Goals are normalized using grouped statistics computed on the train split only.
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
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from tqdm.auto import tqdm
from torchvision.transforms import Normalize

from models import SingleStepMLP


# Prenormalization values. The rotator is somewhat approximate and could probably
# be replaced with a scaled approach instead.
JOINT_STATE_RAW_MIN = torch.tensor([-1.5 * np.pi, -0.45, -0.9, -1.222], dtype=torch.float32)
JOINT_STATE_RAW_MAX = torch.tensor([1.5 * np.pi, 1.0, 0.3, 0.873], dtype=torch.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a single-step goal-conditioned MLP BC policy.")
    parser.add_argument("--data", type=str, required=True, help="Path to the HDF5 segment store.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./bc_component/outputs/ssmlp_js",
        help="Directory for checkpoints and logs.",
    )
    parser.add_argument("--train-ratio", type=float, default=0.9, help="Segment-level train split ratio.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")

    parser.add_argument("--epochs", type=int, default=200, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=2048, help="Batch size for GPU-side batching.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-5, help="AdamW weight decay.")
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout in hidden layers.")
    parser.add_argument(
        "--hidden-dims",
        type=int,
        nargs="+",
        default=[128, 128, 128],
        help="Hidden layer sizes for the MLP.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Ignored in this rewrite because the dataset is preloaded and batched directly on device.",
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
class MaterializedSplit:
    split_name: str
    states: Tensor
    goals: Tensor
    action_targets: Tensor
    segment_ids: Tensor
    timesteps: Tensor
    rejected_segments: List[Tuple[int, int]]
    state_dim: int
    goal_dim: int
    action_dim: int

    def __len__(self) -> int:
        return int(self.states.shape[0])

    @property
    def bytes(self) -> int:
        return (
            self.states.numel() * self.states.element_size()
            + self.goals.numel() * self.goals.element_size()
            + self.action_targets.numel() * self.action_targets.element_size()
            + self.segment_ids.numel() * self.segment_ids.element_size()
            + self.timesteps.numel() * self.timesteps.element_size()
        )


class DeviceSplit:
    """A full split that already lives on the target device."""

    def __init__(self, split: MaterializedSplit, device: torch.device) -> None:
        self.split_name = split.split_name
        self.states = split.states.to(device=device, non_blocking=False)
        self.goals = split.goals.to(device=device, non_blocking=False)
        self.action_targets = split.action_targets.to(device=device, non_blocking=False)
        self.segment_ids = split.segment_ids.to(device=device, non_blocking=False)
        self.timesteps = split.timesteps.to(device=device, non_blocking=False)
        self.rejected_segments = split.rejected_segments
        self.state_dim = split.state_dim
        self.goal_dim = split.goal_dim
        self.action_dim = split.action_dim
        self.device = device

    def __len__(self) -> int:
        return int(self.states.shape[0])

    @property
    def bytes(self) -> int:
        return (
            self.states.numel() * self.states.element_size()
            + self.goals.numel() * self.goals.element_size()
            + self.action_targets.numel() * self.action_targets.element_size()
            + self.segment_ids.numel() * self.segment_ids.element_size()
            + self.timesteps.numel() * self.timesteps.element_size()
        )

    def iter_batches(self, batch_size: int, shuffle: bool) -> Iterable[Dict[str, Tensor]]:
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
                "state": self.states.index_select(0, batch_idx),
                "goal": self.goals.index_select(0, batch_idx),
                "action_targets": self.action_targets.index_select(0, batch_idx),
                "segment_idx": self.segment_ids.index_select(0, batch_idx),
                "t": self.timesteps.index_select(0, batch_idx),
            }


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_joint_tensor(x: Tensor, joint_raw_min: Tensor, joint_raw_max: Tensor) -> Tensor:
    x = x.float()
    return 2.0 * (x - joint_raw_min) / (joint_raw_max - joint_raw_min) - 1.0


def normalize_goal_grouped(goal: Tensor, goal_mean_grouped: Tensor, goal_std_grouped: Tensor) -> Tensor:
    """
    Normalize goal coordinates with shared stats per coordinate index.

    Goal is interpreted as logical shape (3, 3):
      - all x-like entries share mean/std index 0
      - all y-like entries share mean/std index 1
      - all z-like entries share mean/std index 2
    """
    goal = goal.float().reshape(3, 3)
    mean = goal_mean_grouped.float().reshape(1, 3)
    std = goal_std_grouped.float().reshape(1, 3)
    return ((goal - mean) / std).reshape(-1)


def unnormalize_goal_grouped(goal: Tensor, goal_mean_grouped: Tensor, goal_std_grouped: Tensor) -> Tensor:
    """Inverse of normalize_goal_grouped for logical shape (3, 3) goals."""
    goal = goal.float().reshape(3, 3)
    mean = goal_mean_grouped.float().reshape(1, 3)
    std = goal_std_grouped.float().reshape(1, 3)
    return (goal * std + mean).reshape(-1)


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

    goal_all = torch.stack(goal_rows, dim=0)  # (S_train, 3, 3)
    goal_mean = goal_all.mean(dim=(0, 1))
    goal_std = goal_all.std(dim=(0, 1), unbiased=False).clamp_min(eps)

    return {
        "goal_mean": goal_mean,
        "goal_std": goal_std,
    }


def materialize_split(
    path: str | Path,
    segment_indices: Iterable[int],
    split_name: str,
    joint_raw_min: Tensor,
    joint_raw_max: Tensor,
    goal_norm: Normalize | None = None,
) -> MaterializedSplit:
    segment_indices = list(segment_indices)
    joint_raw_min = joint_raw_min.detach().clone().float()
    joint_raw_max = joint_raw_max.detach().clone().float()
    rejected_segments: List[Tuple[int, int]] = []

    if joint_raw_min.shape != (4,) or joint_raw_max.shape != (4,):
        raise ValueError(
            f"joint_raw_min and joint_raw_max must each have shape (4,), got "
            f"{tuple(joint_raw_min.shape)} and {tuple(joint_raw_max.shape)}"
        )
    if not torch.all(joint_raw_max > joint_raw_min):
        raise ValueError("Every entry in joint_raw_max must be strictly greater than joint_raw_min.")

    state_rows: List[Tensor] = []
    target_rows: List[Tensor] = []
    goal_rows: List[Tensor] = []
    segment_id_rows: List[Tensor] = []
    timestep_rows: List[Tensor] = []

    with h5py.File(path, "r") as f:
        goals = f["goals"]
        trajectories = f["trajectories"]
        lengths = f["lengths"]

        if trajectories.ndim != 3 or trajectories.shape[2] != 4:
            raise ValueError(f"Expected trajectories with shape (S, N, 4), got {tuple(trajectories.shape)}")

        sample_goal = goals[0]
        flat_dim = int(torch.as_tensor(sample_goal).numel())
        if flat_dim != 9:
            raise ValueError(f"Expected goals to flatten to size 9, got stored shape {tuple(goals.shape[1:])}")

        state_dim = 4
        goal_dim = flat_dim
        action_dim = 4

        for seg_idx in segment_indices:
            seg_len = int(lengths[seg_idx])
            if seg_len < 2:
                rejected_segments.append((seg_idx, seg_len))
                warnings.warn(
                    f"[{split_name}] rejecting segment {seg_idx} with length {seg_len}: "
                    "need at least 2 steps for single-step input/target pairs.",
                    stacklevel=2,
                )
                continue

            traj = torch.from_numpy(trajectories[seg_idx, :seg_len]).float()
            goal = torch.from_numpy(goals[seg_idx]).float()

            states = normalize_joint_tensor(traj[:-1], joint_raw_min, joint_raw_max)
            targets = normalize_joint_tensor(traj[1:], joint_raw_min, joint_raw_max)

            goal_reshaped = goal.reshape(3, 3)
            if goal_norm is None:
                goal_row = goal_reshaped.reshape(-1)
            else:
                # Treat coordinate index as channel dimension:
                # (entries=3, coords=3) -> (coords=3, entries=3, width=1)
                g = goal_reshaped.t().unsqueeze(-1)
                g = goal_norm(g)
                goal_row = g.squeeze(-1).t().reshape(-1)
            goal_row = goal_row.unsqueeze(0).repeat(seg_len - 1, 1)

            state_rows.append(states)
            target_rows.append(targets)
            goal_rows.append(goal_row)
            segment_id_rows.append(torch.full((seg_len - 1,), seg_idx, dtype=torch.long))
            timestep_rows.append(torch.arange(seg_len - 1, dtype=torch.long))

    if state_rows:
        states = torch.cat(state_rows, dim=0).contiguous()
        action_targets = torch.cat(target_rows, dim=0).contiguous()
        goals = torch.cat(goal_rows, dim=0).contiguous()
        segment_ids = torch.cat(segment_id_rows, dim=0).contiguous()
        timesteps = torch.cat(timestep_rows, dim=0).contiguous()
    else:
        state_dim = 4
        goal_dim = 9
        action_dim = 4
        states = torch.empty((0, 4), dtype=torch.float32)
        action_targets = torch.empty((0, 4), dtype=torch.float32)
        goals = torch.empty((0, 9), dtype=torch.float32)
        segment_ids = torch.empty((0,), dtype=torch.long)
        timesteps = torch.empty((0,), dtype=torch.long)

    split = MaterializedSplit(
        split_name=split_name,
        states=states,
        goals=goals,
        action_targets=action_targets,
        segment_ids=segment_ids,
        timesteps=timesteps,
        rejected_segments=rejected_segments,
        state_dim=state_dim,
        goal_dim=goal_dim,
        action_dim=action_dim,
    )
    print(
        f"[{split_name}] materialized {len(split)} samples in host memory "
        f"({split.bytes / 1024**3:.3f} GiB)"
    )
    return split


def print_cuda_memory(prefix: str, device: torch.device) -> None:
    if device.type != "cuda":
        return
    free_bytes, total_bytes = torch.cuda.mem_get_info(device=device)
    used_bytes = total_bytes - free_bytes
    print(
        f"{prefix} | allocated={torch.cuda.memory_allocated(device) / 1024**3:.3f} GiB | "
        f"reserved={torch.cuda.memory_reserved(device) / 1024**3:.3f} GiB | "
        f"used={used_bytes / 1024**3:.3f} / {total_bytes / 1024**3:.3f} GiB"
    )


def move_split_to_device(split: MaterializedSplit, device: torch.device) -> DeviceSplit:
    print(f"[{split.split_name}] moving full split to {device} ({split.bytes / 1024**3:.3f} GiB)")
    device_split = DeviceSplit(split, device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        print_cuda_memory(f"[{split.split_name}] after VRAM load", device)
    print(
        f"[{split.split_name}] resident on {device}: {len(device_split)} samples "
        f"({device_split.bytes / 1024**3:.3f} GiB)"
    )
    return device_split


def run_epoch(
    model: nn.Module,
    split: DeviceSplit,
    batch_size: int,
    optimizer: torch.optim.Optimizer | None,
    epoch: int,
    total_epochs: int,
) -> Dict[str, float]:
    is_train = optimizer is not None
    phase = "train" if is_train else "val"

    if len(split) == 0:
        if is_train:
            raise RuntimeError("Training split contains zero samples.")
        return {"loss": float("nan")}

    if is_train:
        model.train()
        grad_context = torch.enable_grad()
    else:
        model.eval()
        grad_context = torch.inference_mode()

    total_loss = 0.0
    total_count = 0
    batch_count = math.ceil(len(split) / batch_size)

    with grad_context:
        progress = tqdm(
            split.iter_batches(batch_size=batch_size, shuffle=is_train),
            total=batch_count,
            desc=f"Epoch {epoch:03d}/{total_epochs:03d} [{phase}]",
            leave=False,
            dynamic_ncols=True,
        )
        for batch in progress:
            if is_train:
                optimizer.zero_grad(set_to_none=True)

            loss: Tensor = model.train_step(batch)

            if is_train:
                loss.backward()
                optimizer.step()

            current_batch_size = int(batch["action_targets"].shape[0])
            total_loss += float(loss.item()) * current_batch_size
            total_count += current_batch_size
            avg_loss = total_loss / total_count
            progress.set_postfix(loss=f"{avg_loss:.6f}")

    return {"loss": total_loss / total_count}


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

    if not torch.cuda.is_available():
        raise RuntimeError(
            "This rewrite is VRAM-first and requires CUDA. No CUDA device is available."
        )

    device = torch.device("cuda:0")
    print(f"Using device: {device}")
    print(f"CUDA device: {torch.cuda.get_device_name(device)}")
    print_cuda_memory("Startup CUDA memory", device)

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

    train_split_cpu = materialize_split(
        args.data,
        train_segments,
        split_name="train",
        joint_raw_min=JOINT_STATE_RAW_MIN,
        joint_raw_max=JOINT_STATE_RAW_MAX,
        goal_norm=goal_norm,
    )
    val_split_cpu = materialize_split(
        args.data,
        val_segments,
        split_name="val",
        joint_raw_min=JOINT_STATE_RAW_MIN,
        joint_raw_max=JOINT_STATE_RAW_MAX,
        goal_norm=goal_norm,
    )

    total_cpu_bytes = train_split_cpu.bytes + val_split_cpu.bytes
    print(f"Total materialized dataset footprint before VRAM load: {total_cpu_bytes / 1024**3:.3f} GiB")

    # User-requested flow: load the full dataset into VRAM before training starts,
    # then batch out of the resident GPU tensors.
    train_split = move_split_to_device(train_split_cpu, device)
    val_split = move_split_to_device(val_split_cpu, device)

    # Free the host-side copies now that the canonical training source is on device.
    del train_split_cpu
    del val_split_cpu

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
        f"Constructed device-resident splits: train_samples={len(train_split)}, "
        f"val_samples={len(val_split)}, train_rejected={len(train_split.rejected_segments)}, "
        f"val_rejected={len(val_split.rejected_segments)}"
    )

    if args.num_workers != 0:
        print(
            f"Ignoring --num-workers={args.num_workers} because the dataset is preloaded and batched directly on device."
        )

    model = SingleStepMLP(
        state_dim=train_split.state_dim,
        goal_dim=train_split.goal_dim,
        action_dim=train_split.action_dim,
        hidden_dims=args.hidden_dims,
        dropout=args.dropout,
        loss_func=F.l1_loss,
    ).to(device)
    print_cuda_memory("After model creation", device)

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
        "dataset_residency": "full_train_and_val_preloaded_to_vram_before_training",
        "model": {
            "state_dim": train_split.state_dim,
            "goal_dim": train_split.goal_dim,
            "action_dim": train_split.action_dim,
            "hidden_dims": list(args.hidden_dims),
            "dropout": args.dropout,
            "loss_type": "l1_loss",
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
        "split_summary": asdict(split_summary),
        "memory": {
            "train_split_bytes": train_split.bytes,
            "val_split_bytes": val_split.bytes,
            "total_split_bytes": train_split.bytes + val_split.bytes,
        },
    }

    history: List[Dict[str, float]] = []
    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model=model,
            split=train_split,
            batch_size=args.batch_size,
            optimizer=optimizer,
            epoch=epoch,
            total_epochs=args.epochs,
        )
        val_metrics = run_epoch(
            model=model,
            split=val_split,
            batch_size=args.batch_size,
            optimizer=None,
            epoch=epoch,
            total_epochs=args.epochs,
        ) if len(val_split) > 0 else {"loss": float("nan")}

        epoch_record = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "val_loss": val_metrics["loss"],
        }
        history.append(epoch_record)

        tqdm.write(
            f"epoch {epoch:03d} | train_loss={train_metrics['loss']:.6f} | val_loss={val_metrics['loss']:.6f}"
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


if __name__ == "__main__":
    main()
