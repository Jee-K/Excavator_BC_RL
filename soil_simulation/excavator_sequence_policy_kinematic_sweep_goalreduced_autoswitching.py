#!/usr/bin/env python3
"""
Physics-less excavator sequence-policy rollout.

This is the sequence-model analogue of the single-step kinematic sweep runner:
- load the excavator URDF,
- load a trained sequence policy checkpoint (MLP / LSTM / ACT-style),
- drive the excavator kinematically from policy outputs without stepping physics,
- query the policy at COMMAND_HZ,
- linearly interpolate between command targets for smoother viewing at VIEWER_FPS.

Inference behavior matches `run_sequence_policy.py`:
- maintain a rolling command-time history of observed 4-D joint states,
- left-pad the history and emit a mask until enough states are seen,
- predict a future chunk,
- apply only the FIRST predicted action at each command tick.

Fill in STARTING_POLICY_Q_RAW with a good 4-D raw joint pose:
    [swing, arm, stick, bucket]

The goal is a reduced 2-D vector: for dig_swing use the second 1/3 with the middle coordinate removed; for dump_return use the third 1/3 with the middle coordinate removed.
"""

from collections import deque
from pathlib import Path
import time
from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np
import torch
from torch import Tensor, nn


class GroupedNormalize:
    """Minimal channel-wise normalization helper compatible with the original tensor usage."""

    def __init__(self, mean, std) -> None:
        self.mean = torch.as_tensor(mean, dtype=torch.float32).view(-1, 1, 1)
        self.std = torch.as_tensor(std, dtype=torch.float32).view(-1, 1, 1)

    def __call__(self, x: Tensor) -> Tensor:
        mean = self.mean.to(device=x.device, dtype=x.dtype)
        std = self.std.to(device=x.device, dtype=x.dtype)
        return (x - mean) / std
import warp as wp
import newton
import newton.examples

from models_phased import ACTStylePolicy, GoalConditionedMLPPolicy, LSTMSeq2SeqPolicy


URDF_RELATIVE_PATH = "./excavatorURDF/excavator_lowpoly_locked_splitbucket.urdf"
CHECKPOINT_ROOT_DIR = "./bc_component/outputs/emlp_large_batch"
INITIAL_BEHAVIOR_STREAM = "dig_swing"
CHECKPOINT_NAME = "best.pt"

VIEWER_FPS = 60.0
COMMAND_HZ = 10.0

AUTO_SWITCH_BETWEEN_STREAMS = True
MODE_SWITCH_WINDOW_FRAMES = 10
MODE_SWITCH_MIN_TRUE_FRAMES = 8
THETA_SWITCH_TOLERANCE = 0.20
BUCKET_FLAT_ANGLE = 1.1
BUCKET_ANGLE_TOLERANCE = 0.05

# Fill these in with good raw 4-joint poses: [swing, arm, stick, bucket].
STARTING_POLICY_Q_RAW_BY_STREAM = {
    "dig_swing": np.array([
        0.0,
        0.0,
        -0.9,
        0.8,
    ], dtype=np.float32),
    "dump_return": np.array([
        0.0,
        0.0,
        -0.9,
        0.8,
    ], dtype=np.float32),
}

# Replace with your desired reduced goal if needed. Shape must remain (2,).
HARDCODED_GOAL_BY_STREAM = {
    "dig_swing": np.array([4.0, 0.5 * np.pi], dtype=np.float32),
    "dump_return": np.array([3.0, 0.0], dtype=np.float32),
}

POLICY_TYPE_TO_NAME = {
    "GoalConditionedMLPPolicy": "mlp",
    "LSTMSeq2SeqPolicy": "lstm",
    "ACTStylePolicy": "act",
}


def choose_device(requested: str | torch.device = "auto") -> torch.device:
    if isinstance(requested, torch.device):
        return requested

    requested_l = str(requested).lower()
    if requested_l == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested_l == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Requested device='cuda' but CUDA is not available.")
        return torch.device("cuda")
    if requested_l == "cpu":
        return torch.device("cpu")
    raise ValueError(f"Unsupported device request: {requested!r}. Use 'auto', 'cuda', or 'cpu'.")


def validate_behavior_stream(stream: str) -> str:
    stream_l = str(stream).strip().lower()
    if stream_l not in {"dig_swing", "dump_return"}:
        raise ValueError(
            f"Unsupported behavior stream={stream!r}. Expected 'dig_swing' or 'dump_return'."
        )
    return stream_l


def next_behavior_stream(stream: str) -> str:
    stream_l = validate_behavior_stream(stream)
    return "dump_return" if stream_l == "dig_swing" else "dig_swing"


def checkpoint_dir_for_behavior_stream(root_dir: str | Path, behavior_stream: str) -> Path:
    return Path(root_dir) / validate_behavior_stream(behavior_stream)


def wrap_angle_pi(angle: float) -> float:
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


def angular_distance(a: float, b: float) -> float:
    return abs(wrap_angle_pi(float(a) - float(b)))


def bucket_composite_angle(policy_q_raw: np.ndarray | Sequence[float]) -> float:
    q = np.asarray(policy_q_raw, dtype=np.float32).reshape(-1)
    if q.shape[0] != 4:
        raise ValueError(f"Expected 4 policy joints for bucket_composite_angle, got shape {q.shape}")
    _, arm, stick, bucket = q
    return float(arm - stick + bucket)


def boundary_condition_for_mode(policy_q_raw: np.ndarray | Sequence[float], mode: str) -> bool:
    comp = bucket_composite_angle(policy_q_raw)
    mode_l = validate_behavior_stream(mode)
    if mode_l == "dig_swing":
        return comp <= (BUCKET_FLAT_ANGLE + BUCKET_ANGLE_TOLERANCE)
    return comp >= (BUCKET_FLAT_ANGLE - BUCKET_ANGLE_TOLERANCE)


def checkpoint_theta_for_goal(goal: np.ndarray | Sequence[float]) -> float:
    g = np.asarray(goal, dtype=np.float32).reshape(-1)
    if g.shape[0] < 2:
        raise ValueError(f"Expected reduced goal with at least 2 values, got shape {g.shape}")
    return float(g[1])


def _as_float_tensor(x: Any, device: torch.device | None = None) -> Tensor:
    t = torch.as_tensor(x, dtype=torch.float32)
    if device is not None:
        t = t.to(device)
    return t


def scale_joint_to_unit(raw: Tensor, raw_min: Tensor, raw_max: Tensor) -> Tensor:
    return 2.0 * (raw - raw_min) / (raw_max - raw_min) - 1.0


def unscale_joint_from_unit(unit: Tensor, raw_min: Tensor, raw_max: Tensor) -> Tensor:
    return 0.5 * (unit + 1.0) * (raw_max - raw_min) + raw_min


def _extract_norm_config(config: Mapping[str, Any]) -> Mapping[str, Any]:
    norm_cfg = config.get("normalization")
    if norm_cfg is None:
        raise KeyError("Checkpoint config does not contain a 'normalization' section.")

    required_keys = [
        "joint_raw_min",
        "joint_raw_max",
        "goal_mean",
        "goal_std",
    ]
    missing = [k for k in required_keys if k not in norm_cfg]
    if missing:
        raise KeyError(f"Checkpoint normalization config is missing keys: {missing}")
    return norm_cfg


def _extract_model_choice(config: Mapping[str, Any], model_cfg: Mapping[str, Any]) -> str:
    policy_type = model_cfg.get("policy_type")
    if policy_type in POLICY_TYPE_TO_NAME:
        return POLICY_TYPE_TO_NAME[str(policy_type)]

    model_choice = config.get("model_choice")
    if model_choice in {"mlp", "lstm", "act"}:
        return str(model_choice)

    raise KeyError(
        "Could not infer model type from checkpoint config. Expected model['policy_type'] "
        "or config['model_choice'] to identify one of: mlp, lstm, act."
    )


def build_model_from_config(config: Mapping[str, Any]) -> nn.Module:
    model_cfg = config.get("model")
    if model_cfg is None:
        raise KeyError("Checkpoint config does not contain a 'model' section.")

    model_name = _extract_model_choice(config, model_cfg)

    common = dict(
        state_dim=int(model_cfg["state_dim"]),
        goal_dim=int(model_cfg["goal_dim"]),
        action_dim=int(model_cfg["action_dim"]),
        history_len=int(model_cfg["history_len"]),
        action_horizon=int(model_cfg["action_horizon"]),
        behavior_stream=model_cfg.get("behavior_stream"),
    )

    if model_name == "mlp":
        return GoalConditionedMLPPolicy(
            **common,
            hidden_dims=model_cfg["hidden_dims"],
            dropout=float(model_cfg.get("dropout", 0.0)),
            loss_type=str(model_cfg.get("loss_type", "smooth_l1")),
        )

    if model_name == "lstm":
        return LSTMSeq2SeqPolicy(
            **common,
            encoder_hidden_dim=int(model_cfg.get("encoder_hidden_dim", 256)),
            decoder_hidden_dim=int(model_cfg.get("decoder_hidden_dim", 256)),
            num_layers=int(model_cfg.get("num_layers", 2)),
            dropout=float(model_cfg.get("dropout", 0.1)),
            default_teacher_forcing_ratio=float(model_cfg.get("default_teacher_forcing_ratio", 0.0)),
            loss_type=str(model_cfg.get("loss_type", "smooth_l1")),
        )

    if model_name == "act":
        return ACTStylePolicy(
            **common,
            d_model=int(model_cfg.get("d_model", 256)),
            latent_dim=int(model_cfg.get("latent_dim", 32)),
            nhead=int(model_cfg.get("nhead", 8)),
            num_encoder_layers=int(model_cfg.get("num_encoder_layers", 4)),
            num_decoder_layers=int(model_cfg.get("num_decoder_layers", 3)),
            ff_dim=int(model_cfg.get("ff_dim", 512)),
            dropout=float(model_cfg.get("dropout", 0.1)),
            kl_weight=float(model_cfg.get("kl_weight", 1e-4)),
            recon_loss_type=str(model_cfg.get("recon_loss_type", "smooth_l1")),
            sample_prior_at_inference=bool(model_cfg.get("sample_prior_at_inference", False)),
        )

    raise ValueError(f"Unsupported model type: {model_name!r}")


class SequenceChunkPolicy:
    """
    Thin inference wrapper around a trained sequence goal-conditioned policy.

    The policy expects:
        raw current joint positions -> selected down to the 4 policy joints if needed,
        then scaled to [-1, 1]
        raw reduced goal in shape (2,) or flattenable to goal_dim values -> feature-wise normalized

    It returns:
        one raw next-step target joint position vector in shape (4,), chosen as the first
        step from the predicted action chunk.
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        history_len: int,
        action_horizon: int,
        joint_raw_min: Tensor,
        joint_raw_max: Tensor,
        goal_mean: Tensor,
        goal_std: Tensor,
        joint_indices: Sequence[int] | None = None,
        clamp_output: bool = True,
    ) -> None:
        self.model = model
        self.device = device
        self.history_len = int(history_len)
        self.action_horizon = int(action_horizon)
        self.joint_raw_min = _as_float_tensor(joint_raw_min, device)
        self.joint_raw_max = _as_float_tensor(joint_raw_max, device)
        self.goal_norm = GroupedNormalize(
            mean=_as_float_tensor(goal_mean).tolist(),
            std=_as_float_tensor(goal_std).tolist(),
        )
        self.joint_indices = list(joint_indices) if joint_indices is not None else None
        self.clamp_output = clamp_output
        self.history: deque[Tensor] = deque(maxlen=self.history_len)

        if self.joint_raw_min.shape != (4,) or self.joint_raw_max.shape != (4,):
            raise ValueError(
                f"Expected joint_raw_min and joint_raw_max to each have shape (4,), got "
                f"{tuple(self.joint_raw_min.shape)} and {tuple(self.joint_raw_max.shape)}"
            )
        if not torch.all(self.joint_raw_max > self.joint_raw_min):
            raise ValueError("Every entry in joint_raw_max must be strictly greater than joint_raw_min.")
        if self.history_len <= 0:
            raise ValueError(f"history_len must be positive, got {self.history_len}")
        if self.action_horizon <= 0:
            raise ValueError(f"action_horizon must be positive, got {self.action_horizon}")

    def reset_history(self) -> None:
        self.history.clear()

    def preprocess_goal(self, goal: np.ndarray | Sequence[float] | Tensor) -> Tensor:
        goal_t = _as_float_tensor(goal, self.device).reshape(-1)
        mean_dim = int(self.goal_norm.mean.shape[0])
        if goal_t.numel() != mean_dim:
            raise ValueError(f"Expected goal to contain {mean_dim} values, got shape {tuple(goal_t.shape)}")

        g = self.goal_norm(goal_t.view(-1, 1, 1)).view(1, -1)
        return g

    def preprocess_state(self, current_joint_positions: np.ndarray | Sequence[float] | Tensor) -> Tensor:
        if isinstance(current_joint_positions, Tensor):
            q = current_joint_positions.detach().cpu().numpy().astype(np.float32)
        else:
            q = np.asarray(current_joint_positions, dtype=np.float32)

        if self.joint_indices is not None:
            q = q[self.joint_indices]
        if q.shape != (4,):
            raise ValueError(
                f"Expected selected current_joint_positions to have shape (4,), got {q.shape}. "
                "Pass joint_indices if the state exposes more than the 4 policy joints."
            )

        q_t = _as_float_tensor(q, self.device)
        q_t = scale_joint_to_unit(q_t, self.joint_raw_min, self.joint_raw_max)
        return q_t

    def _append_state(self, state_unit: Tensor) -> None:
        self.history.append(state_unit.detach().clone())

    def _build_history_tensors(self) -> tuple[Tensor, Tensor]:
        state_dim = int(self.joint_raw_min.numel())
        history = torch.zeros((self.history_len, state_dim), dtype=torch.float32, device=self.device)
        history_mask = torch.zeros((self.history_len,), dtype=torch.bool, device=self.device)

        valid_len = len(self.history)
        if valid_len > 0:
            stacked = torch.stack(list(self.history), dim=0)
            history[-valid_len:] = stacked
            history_mask[-valid_len:] = True

        return history.unsqueeze(0), history_mask.unsqueeze(0)

    @torch.no_grad()
    def predict_chunk(
        self,
        current_joint_positions: np.ndarray | Sequence[float] | Tensor,
        goal: np.ndarray | Sequence[float] | Tensor,
    ) -> np.ndarray:
        state_t = self.preprocess_state(current_joint_positions)
        self._append_state(state_t)
        state_history, history_mask = self._build_history_tensors()
        goal_t = self.preprocess_goal(goal)

        batch = {
            "state_history": state_history,
            "history_mask": history_mask,
            "goal": goal_t,
            "start_action": state_t.unsqueeze(0),
        }
        out = self.model(batch)
        pred_unit = out["predicted_actions"].squeeze(0)
        pred_raw = unscale_joint_from_unit(pred_unit, self.joint_raw_min, self.joint_raw_max)
        if self.clamp_output:
            pred_raw = torch.clamp(pred_raw, min=self.joint_raw_min, max=self.joint_raw_max)
        return pred_raw.detach().cpu().numpy()

    @torch.no_grad()
    def act(
        self,
        current_joint_positions: np.ndarray | Sequence[float] | Tensor,
        goal: np.ndarray | Sequence[float] | Tensor,
    ) -> np.ndarray:
        pred_chunk = self.predict_chunk(current_joint_positions, goal)
        return pred_chunk[0]


def load_sequence_policy(
    checkpoint_dir: str | Path,
    which: str = "best.pt",
    device: str | torch.device = "auto",
    joint_indices: Sequence[int] | None = None,
    clamp_output: bool = True,
    expected_behavior_stream: str | None = None,
) -> SequenceChunkPolicy:
    device_obj = choose_device(device)
    checkpoint_path = Path(checkpoint_dir) / which
    ckpt = torch.load(checkpoint_path, map_location=device_obj)

    if "config" not in ckpt or "model_state_dict" not in ckpt:
        raise KeyError(
            f"Checkpoint at {checkpoint_path} must contain 'config' and 'model_state_dict'."
        )

    config = ckpt["config"]
    checkpoint_behavior_stream = config.get("behavior_stream")
    if expected_behavior_stream is not None:
        expected_behavior_stream = validate_behavior_stream(expected_behavior_stream)
        if checkpoint_behavior_stream is not None and str(checkpoint_behavior_stream) != expected_behavior_stream:
            raise ValueError(
                f"Checkpoint behavior_stream={checkpoint_behavior_stream!r} does not match "
                f"expected_behavior_stream={expected_behavior_stream!r}."
            )
    norm_cfg = _extract_norm_config(config)
    model = build_model_from_config(config).to(device_obj)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    model_cfg = config["model"]
    return SequenceChunkPolicy(
        model=model,
        device=device_obj,
        history_len=int(model_cfg["history_len"]),
        action_horizon=int(model_cfg["action_horizon"]),
        joint_raw_min=_as_float_tensor(norm_cfg["joint_raw_min"]),
        joint_raw_max=_as_float_tensor(norm_cfg["joint_raw_max"]),
        goal_mean=_as_float_tensor(norm_cfg["goal_mean"]),
        goal_std=_as_float_tensor(norm_cfg["goal_std"]),
        joint_indices=joint_indices,
        clamp_output=clamp_output,
    )


class ExcavatorSequencePolicyKinematicSweep:
    def __init__(self, viewer) -> None:
        self.viewer = viewer
        self.device = wp.get_device()

        self.behavior_stream = validate_behavior_stream(INITIAL_BEHAVIOR_STREAM)
        self.urdf_path = Path(URDF_RELATIVE_PATH)
        self.checkpoint_dirs = {
            stream: checkpoint_dir_for_behavior_stream(CHECKPOINT_ROOT_DIR, stream)
            for stream in ("dig_swing", "dump_return")
        }
        self.goals = {
            stream: np.asarray(HARDCODED_GOAL_BY_STREAM[stream], dtype=np.float32)
            for stream in ("dig_swing", "dump_return")
        }

        self.frame_dt = 1.0 / VIEWER_FPS
        self.command_dt = 1.0 / COMMAND_HZ
        self.sim_time = 0.0
        self.command_phase_time = 0.0

        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        builder.add_urdf(
            str(self.urdf_path),
            xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
            floating=False,
            enable_self_collisions=False,
            collapse_fixed_joints=True,
            ignore_inertial_definitions=False,
        )
        self.model = builder.finalize()
        self.state = self.model.state()

        self.joint_q_host = self.state.joint_q.numpy()
        self.joint_qd_host = self.state.joint_qd.numpy()
        self.q_init = np.asarray(self.joint_q_host, dtype=np.float32).copy()
        self.joint_qd_host[...] = 0.0

        self.control_size = int(self.joint_q_host.shape[0])
        self.control_joint_names = self._extract_control_joint_names()
        self.control_lower, self.control_upper = self._extract_control_limits()
        self.joint_map = self._identify_joint_map()
        self.policy_joint_indices = self._policy_joint_indices_from_map()

        self.policies = {
            stream: load_sequence_policy(
                checkpoint_dir=self.checkpoint_dirs[stream],
                which=CHECKPOINT_NAME,
                device="auto",
                joint_indices=None,
                clamp_output=True,
                expected_behavior_stream=stream,
            )
            for stream in ("dig_swing", "dump_return")
        }
        for policy in self.policies.values():
            policy.reset_history()

        self.current_policy_q = np.asarray(STARTING_POLICY_Q_RAW_BY_STREAM[self.behavior_stream], dtype=np.float32).copy()
        if self.current_policy_q.shape != (4,):
            raise ValueError(
                f"STARTING_POLICY_Q_RAW must have shape (4,), got {self.current_policy_q.shape}"
            )

        self.command_start_q = self.current_policy_q.copy()
        self.command_target_q = self.current_policy_q.copy()
        self.last_predicted_chunk: Optional[np.ndarray] = None
        self.switch_condition_history: deque[bool] = deque(maxlen=MODE_SWITCH_WINDOW_FRAMES)
        self.mode_hold_frames = 0

        self._apply_policy_joints(self.current_policy_q)

        self.viewer.set_model(self.model)
        self.viewer.show_particles = False
        self.viewer.show_visual = True
        self.viewer.show_collision = False
        self.viewer.show_cloth = False

        print(f"Loaded URDF: {self.urdf_path}")
        print(f"Initial behavior stream: {self.behavior_stream}")
        print(f"Checkpoint dirs: {self.checkpoint_dirs}")
        print(f"Joint map: {self.joint_map}")
        print(f"Policy joint indices: {self.policy_joint_indices}")
        print(f"Starting policy q: {self.current_policy_q}")
        print(f"Goals: {self.goals}")
        print(
            f"Auto switch: {AUTO_SWITCH_BETWEEN_STREAMS}, theta_tol={THETA_SWITCH_TOLERANCE:.3f}, "
            f"bucket_flat={BUCKET_FLAT_ANGLE:.3f}, hold={MODE_SWITCH_WINDOW_FRAMES}, need={MODE_SWITCH_MIN_TRUE_FRAMES}"
        )

    def _extract_control_joint_names(self) -> list[str]:
        if hasattr(self.model, "joint_name"):
            names = list(self.model.joint_name)
            if len(names) == self.control_size:
                return [str(n) for n in names]
        return [f"q_{i}" for i in range(self.control_size)]

    def _extract_control_limits(self) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        lower = getattr(self.model, "joint_limit_lower", None)
        upper = getattr(self.model, "joint_limit_upper", None)
        if lower is None or upper is None:
            return None, None

        lower_np = lower.numpy()
        upper_np = upper.numpy()
        if lower_np.shape[0] != self.control_size or upper_np.shape[0] != self.control_size:
            return None, None
        return lower_np, upper_np

    def _identify_joint_map(self) -> dict[str, Optional[int]]:
        names_l = [name.lower() for name in self.control_joint_names]
        canonical = {
            "swing": ("tankspin_wheel", "tankspin", "swing", "slew"),
            "arm": ("lower_arm", "lowerarm", "back_arm", "arm"),
            "stick": ("uppertolow", "upper_to_low", "middle", "stick", "dipper"),
            "bucket": ("scoop1", "scoop", "bucket"),
        }

        def find_best(tokens: Iterable[str], used: set[int]) -> Optional[int]:
            for idx, name in enumerate(names_l):
                compact = name.replace("_", "")
                for token in tokens:
                    token_c = token.replace("_", "")
                    if compact == token_c and idx not in used:
                        return idx
            for idx, name in enumerate(names_l):
                compact = name.replace("_", "")
                for token in tokens:
                    token_c = token.replace("_", "")
                    if token_c in compact and idx not in used:
                        return idx
            return None

        resolved: dict[str, Optional[int]] = {}
        used: set[int] = set()
        for key in ("swing", "arm", "stick", "bucket"):
            idx = find_best(canonical[key], used)
            if idx is not None:
                used.add(idx)
            resolved[key] = idx

        if sum(idx is not None for idx in resolved.values()) < 4 and self.control_size >= 4:
            resolved = {
                "swing": max(0, self.control_size - 4),
                "arm": max(0, self.control_size - 3),
                "stick": max(0, self.control_size - 2),
                "bucket": max(0, self.control_size - 1),
            }
        return {k: (v if v is not None and v < self.control_size else None) for k, v in resolved.items()}

    def _policy_joint_indices_from_map(self) -> list[int]:
        ordered = [self.joint_map[key] for key in ("swing", "arm", "stick", "bucket")]
        if any(idx is None for idx in ordered):
            raise RuntimeError(f"Failed to resolve all policy joints from joint map: {self.joint_map}")
        return [int(idx) for idx in ordered]

    def _clip_target(self, index: Optional[int], value: float) -> float:
        if index is None:
            return value
        if self.control_lower is None or self.control_upper is None:
            return value
        lower = float(self.control_lower[index])
        upper = float(self.control_upper[index])
        if lower < upper:
            return float(np.clip(value, lower, upper))
        return value

    def _apply_policy_joints(self, policy_q_raw: np.ndarray) -> None:
        q = self.q_init.copy()
        for policy_i, joint_idx in enumerate(self.policy_joint_indices):
            q[joint_idx] = self._clip_target(joint_idx, float(policy_q_raw[policy_i]))

        self.joint_q_host[...] = q
        self.joint_qd_host[...] = 0.0
        self.state.joint_q.assign(self.joint_q_host)
        self.state.joint_qd.assign(self.joint_qd_host)
        newton.eval_fk(self.model, self.state.joint_q, self.state.joint_qd, self.state)

    def _active_goal(self) -> np.ndarray:
        return self.goals[self.behavior_stream]

    def _active_policy(self) -> SequenceChunkPolicy:
        return self.policies[self.behavior_stream]

    def _maybe_switch_behavior_stream(self) -> None:
        if not AUTO_SWITCH_BETWEEN_STREAMS:
            return

        goal_theta = checkpoint_theta_for_goal(self._active_goal())
        current_theta = float(self.current_policy_q[0])
        theta_error = angular_distance(current_theta, goal_theta)
        theta_ok = theta_error <= THETA_SWITCH_TOLERANCE
        posture_ok = boundary_condition_for_mode(self.current_policy_q, self.behavior_stream)
        condition_ok = theta_ok and posture_ok

        self.switch_condition_history.append(condition_ok)
        self.mode_hold_frames += 1

        true_count = int(sum(self.switch_condition_history))
        should_switch = (
            len(self.switch_condition_history) >= MODE_SWITCH_WINDOW_FRAMES
            and true_count >= MODE_SWITCH_MIN_TRUE_FRAMES
        )

        if not should_switch:
            return

        previous_stream = self.behavior_stream
        self.behavior_stream = next_behavior_stream(self.behavior_stream)
        self._active_policy().reset_history()
        self.switch_condition_history.clear()
        self.mode_hold_frames = 0
        self.command_start_q = self.current_policy_q.copy()
        self.command_target_q = self.current_policy_q.copy()
        self.command_phase_time = 0.0
        self.last_predicted_chunk = None

        print()
        print(
            f"SWITCH {previous_stream} -> {self.behavior_stream} | "
            f"theta={current_theta:.3f}, goal_theta={goal_theta:.3f}, theta_error={theta_error:.3f}, "
            f"bucket_angle={bucket_composite_angle(self.current_policy_q):.3f}, "
            f"window_true={true_count}/{MODE_SWITCH_WINDOW_FRAMES}"
        )

    def _advance_policy(self) -> None:
        current_joint_positions = self.current_policy_q.copy()
        pred_chunk = np.asarray(
            self._active_policy().predict_chunk(current_joint_positions, self._active_goal()),
            dtype=np.float32,
        )
        if pred_chunk.ndim != 2 or pred_chunk.shape[1] != 4:
            raise ValueError(f"Policy returned chunk with shape {pred_chunk.shape}, expected (K, 4)")

        next_q = pred_chunk[0]
        self.last_predicted_chunk = pred_chunk

        print()
        print()
        print(f"mode: {self.behavior_stream}")
        print("current:", current_joint_positions)
        print("next:", next_q)
        print("chunk:")
        print(pred_chunk)

        self.command_start_q = self.current_policy_q.copy()
        self.command_target_q = next_q
        self.command_phase_time = 0.0

    def step(self) -> None:
        if self.command_phase_time <= 0.0:
            self._advance_policy()

        alpha = min(self.command_phase_time / self.command_dt, 1.0)
        display_q = (1.0 - alpha) * self.command_start_q + alpha * self.command_target_q
        self.current_policy_q = display_q.astype(np.float32, copy=False)
        self._apply_policy_joints(self.current_policy_q)
        self._maybe_switch_behavior_stream()

        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state)
        self.viewer.end_frame()

        self.sim_time += self.frame_dt
        self.command_phase_time += self.frame_dt

        if self.command_phase_time >= self.command_dt:
            self.current_policy_q = self.command_target_q.copy()
            self.command_phase_time = 0.0


def main() -> None:
    viewer, _args = newton.examples.init()
    example = ExcavatorSequencePolicyKinematicSweep(viewer)

    next_frame = time.perf_counter()
    while viewer.is_running():
        example.step()
        next_frame += example.frame_dt
        delay = next_frame - time.perf_counter()
        if delay > 0.0:
            time.sleep(delay)
        else:
            next_frame = time.perf_counter()


if __name__ == "__main__":
    main()
