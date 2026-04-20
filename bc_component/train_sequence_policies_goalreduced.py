"""
Sequence goal-conditioned behavioral cloning trainer.

Rewritten for a VRAM-first training flow:
- split train/val at the segment level
- materialize every supervised sample for each split up front
- normalize once during materialization
- move the full train/val tensors onto the GPU before training begins
- batch by indexing already-resident GPU tensors
- show tqdm bar-style progress for train/val each epoch

This trainer supports the three sequence policies defined in models_phased.py:
    - GoalConditionedMLPPolicy
    - LSTMSeq2SeqPolicy
    - ACTStylePolicy

Assumptions carried over from the original sequence trainer:
- Input data lives in an HDF5 segment store with datasets:
    goals:        (S, goal_dim) reduced per-stream goals; in the current setup goal_dim=2
    trajectories: (S, N_max, 4)
    lengths:      (S,)
- Trajectories serve as both the state-history source and the future target source.
- Train/val split is performed by segment, before sample extraction.
- Joint states/targets are normalized to [-1, 1] using explicit raw min/max constants.
- Goals are normalized independently per reduced goal component using train-split statistics only.
- CUDA is assumed to be available.

Supervised sample formation:
- One sample is anchored at time index t.
- Input history is the fixed-length window ending at t:
      state_history = trajectory[max(0, t-history_len+1) : t+1]
  This is left-padded with zeros when the segment prefix is shorter than history_len.
- history_mask marks which entries in the fixed history window are valid.
- Target chunk is the next action_horizon steps:
      action_targets = trajectory[t+1 : t+1+action_horizon]
- Only anchors with a full future chunk are kept.
- start_action is set to trajectory[t], which is the decoder warm-start.
"""

import argparse
import json
import math
import random
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

import h5py
import numpy as np
import torch
from torch import Tensor, nn


from tqdm.auto import tqdm

from models_phased import (
    ACTStylePolicy,
    BEHAVIOR_STREAM_CHOICES,
    GoalConditionedMLPPolicy,
    LSTMSeq2SeqPolicy,
)


# Prenormalization values carried over from the single-step trainer.
JOINT_STATE_RAW_MIN = torch.tensor([-1.5 * np.pi, -0.45, -0.9, -1.222], dtype=torch.float32)
JOINT_STATE_RAW_MAX = torch.tensor([1.5 * np.pi, 1.0, 0.3, 0.873], dtype=torch.float32)

MODEL_CHOICES = ("mlp", "lstm", "act")
LOSS_TYPE_CHOICES = ("mse", "smooth_l1", "huber")
BEHAVIOR_STREAM_SELECTION_CHOICES = (*BEHAVIOR_STREAM_CHOICES, "all")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train goal-conditioned sequence BC policies.")
    parser.add_argument("--data", type=str, required=True, help="Path to the HDF5 segment store.")
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
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

    parser.add_argument("--epochs", type=int, default=80, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=512, help="Batch size for GPU-side batching.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-5, help="AdamW weight decay.")
    parser.add_argument(
        "--clip-grad-norm",
        type=float,
        default=1.0,
        help="Gradient clipping max norm. Set <= 0 to disable.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Ignored in this rewrite because the dataset is preloaded and batched directly on device.",
    )

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
    parser.add_argument(
        "--behavior-stream",
        type=str,
        default="all",
        choices=BEHAVIOR_STREAM_SELECTION_CHOICES,
        help="Which behavior stream to train: one stream or all labeled streams sequentially.",
    )
    return parser.parse_args()


@dataclass
class SplitSummary:
    behavior_stream: str
    total_segments_in_file: int
    matching_behavior_segments: int
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
            + self.segment_ids.numel() * self.segment_ids.element_size()
            + self.timesteps.numel() * self.timesteps.element_size()
        )


class DeviceSequenceSplit:
    """A full split that already lives on the target device."""

    def __init__(self, split: MaterializedSequenceSplit, device: torch.device) -> None:
        self.split_name = split.split_name
        self.state_histories = split.state_histories.to(device=device, non_blocking=False)
        self.history_masks = split.history_masks.to(device=device, non_blocking=False)
        self.goals = split.goals.to(device=device, non_blocking=False)
        self.action_targets = split.action_targets.to(device=device, non_blocking=False)
        self.start_actions = split.start_actions.to(device=device, non_blocking=False)
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
            + self.segment_ids.numel() * self.segment_ids.element_size()
            + self.timesteps.numel() * self.timesteps.element_size()
        )

    def iter_batches(
        self,
        batch_size: int,
        shuffle: bool,
        teacher_forcing_ratio: Optional[float] = None,
    ) -> Iterator[Dict[str, Tensor | float]]:
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
            batch: Dict[str, Tensor | float] = {
                "state_history": self.state_histories.index_select(0, batch_idx),
                "history_mask": self.history_masks.index_select(0, batch_idx),
                "goal": self.goals.index_select(0, batch_idx),
                "action_targets": self.action_targets.index_select(0, batch_idx),
                "start_action": self.start_actions.index_select(0, batch_idx),
                "segment_idx": self.segment_ids.index_select(0, batch_idx),
                "t": self.timesteps.index_select(0, batch_idx),
            }
            if teacher_forcing_ratio is not None:
                batch["teacher_forcing_ratio"] = float(teacher_forcing_ratio)
            yield batch


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


def normalize_goal_components(goal: Tensor, goal_mean: Tensor, goal_std: Tensor) -> Tensor:
    goal = goal.float().reshape(-1)
    goal_mean = goal_mean.to(device=goal.device, dtype=goal.dtype).reshape(-1)
    goal_std = goal_std.to(device=goal.device, dtype=goal.dtype).reshape(-1)
    if goal.shape != goal_mean.shape or goal.shape != goal_std.shape:
        raise ValueError(
            f"Goal normalization shapes must match exactly: goal={tuple(goal.shape)}, "
            f"goal_mean={tuple(goal_mean.shape)}, goal_std={tuple(goal_std.shape)}"
        )
    return (goal - goal_mean) / goal_std


def _decode_behavior_stream_label(raw: object) -> str:
    if isinstance(raw, bytes):
        return raw.decode("utf-8").strip()
    if hasattr(raw, "decode"):
        return raw.decode("utf-8").strip()
    return str(raw).strip()


def load_behavior_stream_labels(path: str | Path) -> list[str]:
    with h5py.File(path, "r") as f:
        num_segments = int(f["goals"].shape[0])
        labels_ds = f.get("behavior_stream_labels")
        if labels_ds is None:
            return ["full_cycle"] * num_segments
        return [_decode_behavior_stream_label(x) for x in labels_ds[:]]


def resolve_behavior_streams(selection: str) -> list[str]:
    if selection == "all":
        return list(BEHAVIOR_STREAM_CHOICES)
    return [selection]


def materialize_sequence_split(
    path: str | Path,
    segment_indices: Iterable[int],
    split_name: str,
    history_len: int,
    action_horizon: int,
    joint_raw_min: Tensor,
    joint_raw_max: Tensor,
    goal_mean: Optional[Tensor] = None,
    goal_std: Optional[Tensor] = None,
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
    target_rows: List[Tensor] = []
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

        sample_goal = torch.as_tensor(goals[0]).float().reshape(-1)
        goal_dim = int(sample_goal.numel())
        if goal_dim <= 0:
            raise ValueError(f"Expected goal_dim > 0, got stored shape {tuple(goals.shape[1:])}")

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

            goal_row = torch.from_numpy(goals[seg_idx]).float().reshape(-1)
            if goal_mean is not None or goal_std is not None:
                if goal_mean is None or goal_std is None:
                    raise ValueError("goal_mean and goal_std must either both be provided or both be omitted.")
                goal_row = normalize_goal_components(goal_row, goal_mean=goal_mean, goal_std=goal_std)

            for t in range(sample_count):
                start_idx = max(0, t - history_len + 1)
                history_valid = traj[start_idx : t + 1]
                valid_len = int(history_valid.shape[0])

                history = torch.zeros((history_len, state_dim), dtype=torch.float32)
                history_mask = torch.zeros((history_len,), dtype=torch.bool)
                history[-valid_len:] = history_valid
                history_mask[-valid_len:] = True

                action_targets = traj[t + 1 : t + 1 + action_horizon]
                start_action = traj[t]

                history_rows.append(history.unsqueeze(0))
                history_mask_rows.append(history_mask.unsqueeze(0))
                target_rows.append(action_targets.unsqueeze(0))
                goal_rows.append(goal_row.unsqueeze(0))
                start_action_rows.append(start_action.unsqueeze(0))
                segment_id_rows.append(torch.tensor([seg_idx], dtype=torch.long))
                timestep_rows.append(torch.tensor([t], dtype=torch.long))

    if history_rows:
        state_histories = torch.cat(history_rows, dim=0).contiguous()
        history_masks = torch.cat(history_mask_rows, dim=0).contiguous()
        action_targets = torch.cat(target_rows, dim=0).contiguous()
        goals_tensor = torch.cat(goal_rows, dim=0).contiguous()
        start_actions = torch.cat(start_action_rows, dim=0).contiguous()
        segment_ids = torch.cat(segment_id_rows, dim=0).contiguous()
        timesteps = torch.cat(timestep_rows, dim=0).contiguous()
    else:
        state_histories = torch.empty((0, history_len, state_dim), dtype=torch.float32)
        history_masks = torch.empty((0, history_len), dtype=torch.bool)
        action_targets = torch.empty((0, action_horizon, action_dim), dtype=torch.float32)
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
        segment_ids=segment_ids,
        timesteps=timesteps,
        rejected_segments=rejected_segments,
        state_dim=state_dim,
        goal_dim=goal_dim,
        action_dim=action_dim,
        history_len=history_len,
        action_horizon=action_horizon,
    )
    return split


def build_segment_split(
    path: str | Path,
    train_ratio: float,
    seed: int,
    action_horizon: int,
    behavior_stream: str,
) -> Tuple[List[int], List[int], SplitSummary]:
    if not 0.0 < train_ratio < 1.0:
        raise ValueError(f"train_ratio must be in (0, 1), got {train_ratio}")

    with h5py.File(path, "r") as f:
        num_segments = int(f["goals"].shape[0])
        lengths = f["lengths"][:]
        labels_ds = f.get("behavior_stream_labels")
        if labels_ds is None:
            raise RuntimeError(
                "The dataset does not contain behavior_stream_labels. Rebuild the segment store "
                "with the phased segment builder before training per-stream models."
            )
        labels = [_decode_behavior_stream_label(x) for x in labels_ds[:]]

    matching_segments = [idx for idx, label in enumerate(labels) if label == behavior_stream]
    if not matching_segments:
        raise RuntimeError(f"No segments found for behavior_stream={behavior_stream!r}.")

    rejected_segments = [idx for idx in matching_segments if int(lengths[idx]) <= action_horizon]
    accepted_segments = [idx for idx in matching_segments if int(lengths[idx]) > action_horizon]

    rng = random.Random(seed)
    rng.shuffle(accepted_segments)

    if len(accepted_segments) == 0:
        raise RuntimeError(
            f"No usable segments found for behavior_stream={behavior_stream!r}: "
            f"every matching segment has length <= action_horizon ({action_horizon})."
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
            f"Validation split is empty for behavior_stream={behavior_stream!r}. "
            "Consider using more data or a different train_ratio.",
            stacklevel=2,
        )

    train_samples = int(sum(max(0, int(lengths[idx]) - action_horizon) for idx in train_segments))
    val_samples = int(sum(max(0, int(lengths[idx]) - action_horizon) for idx in val_segments))

    summary = SplitSummary(
        behavior_stream=behavior_stream,
        total_segments_in_file=num_segments,
        matching_behavior_segments=len(matching_segments),
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

            goal = torch.from_numpy(goals[seg_idx]).float().reshape(-1)
            goal_rows.append(goal)

    if not goal_rows:
        raise RuntimeError("No usable train segments for goal normalization stats.")

    goal_all = torch.stack(goal_rows, dim=0)
    goal_mean = goal_all.mean(dim=0)
    goal_std = goal_all.std(dim=0, unbiased=False).clamp_min(eps)

    return {
        "goal_mean": goal_mean,
        "goal_std": goal_std,
    }


def teacher_forcing_ratio_for_epoch(epoch: int, total_epochs: int, start: float, end: float) -> float:
    """Linear teacher forcing schedule for the LSTM decoder."""
    if total_epochs <= 1:
        return float(end)
    progress = float(epoch - 1) / float(total_epochs - 1)
    ratio = start + progress * (end - start)
    return float(max(0.0, min(1.0, ratio)))


def build_model(args: argparse.Namespace, split: MaterializedSequenceSplit, behavior_stream: str) -> nn.Module:
    common = dict(
        state_dim=split.state_dim,
        goal_dim=split.goal_dim,
        action_dim=split.action_dim,
        history_len=args.history_len,
        action_horizon=args.action_horizon,
    )

    if args.model == "mlp":
        return GoalConditionedMLPPolicy(
            **common,
            hidden_dims=args.mlp_hidden_dims,
            dropout=args.mlp_dropout,
            loss_type=args.mlp_loss_type,
            behavior_stream=behavior_stream,
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
            behavior_stream=behavior_stream,
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
            behavior_stream=behavior_stream,
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
    split: DeviceSequenceSplit,
    batch_size: int,
    model_name: str,
    epoch: int,
    total_epochs: int,
) -> Dict[str, float]:
    model.eval()
    totals: Dict[str, float] = {}
    total_count = 0
    num_batches = math.ceil(len(split) / batch_size) if len(split) > 0 else 0

    progress = tqdm(
        split.iter_batches(batch_size=batch_size, shuffle=False, teacher_forcing_ratio=None),
        total=num_batches,
        desc=f"Epoch {epoch:03d}/{total_epochs:03d} [val]",
        leave=False,
        dynamic_ncols=True,
    )
    for batch in progress:
        out = model(batch)
        metrics = scalar_metrics_from_output(out, model_name=model_name)
        batch_size_actual = int(batch["action_targets"].shape[0])

        for key, value in metrics.items():
            totals[key] = totals.get(key, 0.0) + value * batch_size_actual
        total_count += batch_size_actual

        if "loss" in metrics:
            progress.set_postfix(loss=f"{metrics['loss']:.4f}")

    if total_count == 0:
        result = {"loss": float("nan")}
        if model_name == "act":
            result["recon_loss"] = float("nan")
            result["kl_loss"] = float("nan")
        return result

    return {key: value / total_count for key, value in totals.items()}


def train_one_epoch(
    model: nn.Module,
    split: DeviceSequenceSplit,
    optimizer: torch.optim.Optimizer,
    model_name: str,
    teacher_forcing_ratio: Optional[float],
    clip_grad_norm: Optional[float],
    batch_size: int,
    epoch: int,
    total_epochs: int,
) -> Dict[str, float]:
    model.train()
    totals: Dict[str, float] = {}
    total_count = 0
    num_batches = math.ceil(len(split) / batch_size) if len(split) > 0 else 0

    progress = tqdm(
        split.iter_batches(
            batch_size=batch_size,
            shuffle=True,
            teacher_forcing_ratio=teacher_forcing_ratio,
        ),
        total=num_batches,
        desc=f"Epoch {epoch:03d}/{total_epochs:03d} [train]",
        leave=False,
        dynamic_ncols=True,
    )
    for batch in progress:
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
        batch_size_actual = int(batch["action_targets"].shape[0])
        for key, value in metrics.items():
            totals[key] = totals.get(key, 0.0) + value * batch_size_actual
        total_count += batch_size_actual

        postfix = {}
        if "loss" in metrics:
            postfix["loss"] = f"{metrics['loss']:.4f}"
        if teacher_forcing_ratio is not None:
            postfix["tf"] = f"{teacher_forcing_ratio:.3f}"
        if postfix:
            progress.set_postfix(**postfix)

    if total_count == 0:
        raise RuntimeError("Training split produced zero samples.")

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


def train_for_behavior_stream(
    args: argparse.Namespace,
    root_output_dir: Path,
    device: torch.device,
    behavior_stream: str,
) -> None:
    stream_output_dir = root_output_dir / behavior_stream
    stream_output_dir.mkdir(parents=True, exist_ok=True)

    train_segments, val_segments, split_summary = build_segment_split(
        path=args.data,
        train_ratio=args.train_ratio,
        seed=args.seed,
        action_horizon=args.action_horizon,
        behavior_stream=behavior_stream,
    )

    goal_stats = compute_goal_normalization_stats(
        path=args.data,
        train_segments=train_segments,
        action_horizon=args.action_horizon,
    )
    train_split_cpu = materialize_sequence_split(
        path=args.data,
        segment_indices=train_segments,
        split_name=f"{behavior_stream}/train",
        history_len=args.history_len,
        action_horizon=args.action_horizon,
        joint_raw_min=JOINT_STATE_RAW_MIN,
        joint_raw_max=JOINT_STATE_RAW_MAX,
        goal_mean=goal_stats["goal_mean"],
        goal_std=goal_stats["goal_std"],
    )
    val_split_cpu = materialize_sequence_split(
        path=args.data,
        segment_indices=val_segments,
        split_name=f"{behavior_stream}/val",
        history_len=args.history_len,
        action_horizon=args.action_horizon,
        joint_raw_min=JOINT_STATE_RAW_MIN,
        joint_raw_max=JOINT_STATE_RAW_MAX,
        goal_mean=goal_stats["goal_mean"],
        goal_std=goal_stats["goal_std"],
    )

    print(f"\n=== behavior_stream={behavior_stream} ===")
    print("Segment split summary:")
    print(json.dumps(asdict(split_summary), indent=2))
    print("Goal normalization stats (train split only):")
    print(
        json.dumps(
            {
                "goal_mean": goal_stats["goal_mean"].tolist(),
                "goal_std": goal_stats["goal_std"].tolist(),
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
        f"Constructed CPU splits: train_samples={len(train_split_cpu)}, val_samples={len(val_split_cpu)}, "
        f"train_rejected={len(train_split_cpu.rejected_segments)}, val_rejected={len(val_split_cpu.rejected_segments)}"
    )

    if args.num_workers != 0:
        print(
            f"Ignoring --num-workers={args.num_workers} because the full dataset is preloaded and batched directly on device."
        )

    print(f"Using device: {device}")
    print(f"CUDA device: {torch.cuda.get_device_name(device)}")
    if hasattr(torch.cuda, "mem_get_info"):
        free_bytes, total_bytes = torch.cuda.mem_get_info(device=device)
        print(
            f"CUDA memory before dataset upload: free={format_bytes(free_bytes)} / total={format_bytes(total_bytes)}"
        )

    train_split = DeviceSequenceSplit(train_split_cpu, device=device)
    val_split = DeviceSequenceSplit(val_split_cpu, device=device)

    print(f"[train] moved full split to VRAM ({format_bytes(train_split.bytes)})")
    print(f"[val] moved full split to VRAM ({format_bytes(val_split.bytes)})")
    print(f"[total] dataset tensors resident in VRAM: {format_bytes(train_split.bytes + val_split.bytes)}")

    del train_split_cpu
    del val_split_cpu

    model = build_model(args, train_split, behavior_stream=behavior_stream).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    config: Dict[str, object] = {
        "data": str(args.data),
        "output_dir": str(stream_output_dir),
        "root_output_dir": str(root_output_dir),
        "behavior_stream": behavior_stream,
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
        "dataset_residency": "full_train_and_val_splits_in_vram",
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
            "goal_library": None,
            "goal_mean": goal_stats["goal_mean"].tolist(),
            "goal_std": goal_stats["goal_std"].tolist(),
            "goal_mode": "independent_component_standardization",
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
        "dataset_bytes": {
            "train_vram_bytes": train_split.bytes,
            "val_vram_bytes": val_split.bytes,
            "total_vram_bytes": train_split.bytes + val_split.bytes,
        },
        "cli_args": vars(args),
    }

    history: List[Dict[str, float]] = []
    best_val_loss = float("inf")
    clip_grad_norm = None if args.clip_grad_norm <= 0 else float(args.clip_grad_norm)

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
            split=train_split,
            optimizer=optimizer,
            model_name=args.model,
            teacher_forcing_ratio=current_teacher_forcing,
            clip_grad_norm=clip_grad_norm,
            batch_size=args.batch_size,
            epoch=epoch,
            total_epochs=args.epochs,
        )
        val_metrics = (
            evaluate(
                model=model,
                split=val_split,
                batch_size=args.batch_size,
                model_name=args.model,
                epoch=epoch,
                total_epochs=args.epochs,
            )
            if len(val_split) > 0
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
            f"[{behavior_stream}] epoch {epoch:03d} | train_loss={train_metrics['loss']:.6f} | "
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

        save_checkpoint(stream_output_dir / "latest.pt", model, optimizer, epoch, history, config)

        if not math.isnan(val_metrics["loss"]) and val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            save_checkpoint(stream_output_dir / "best.pt", model, optimizer, epoch, history, config)

    with open(stream_output_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    with open(stream_output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    print(f"Finished behavior_stream={behavior_stream}. Wrote outputs to: {stream_output_dir}")
    print("Saved: latest.pt, best.pt, history.json, summary.json")


def main() -> None:
    args = parse_args()
    root_output_dir = Path(args.output_dir)
    root_output_dir.mkdir(parents=True, exist_ok=True)

    seed_everything(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("This VRAM-first rewrite requires CUDA, but no CUDA device is available.")
    device = torch.device("cuda")

    requested_streams = resolve_behavior_streams(args.behavior_stream)
    print(f"Training behavior streams: {requested_streams}")

    for behavior_stream in requested_streams:
        train_for_behavior_stream(
            args=args,
            root_output_dir=root_output_dir,
            device=device,
            behavior_stream=behavior_stream,
        )

    print(f"Finished all requested behavior streams. Root output dir: {root_output_dir}")


if __name__ == "__main__":
    main()
