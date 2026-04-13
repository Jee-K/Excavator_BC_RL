#!/usr/bin/env python3
"""
Physics-less excavator policy rollout.

This is a standalone simulator-side script: it loads the excavator URDF,
loads a trained single-step MLP policy checkpoint, and drives the excavator
kinematically from policy outputs without stepping physics.

The viewer runs at VIEWER_FPS, while the policy is queried at COMMAND_HZ.
Between policy commands, joint targets are linearly interpolated for smooth
viewing.

Fill in STARTING_POLICY_Q_RAW with a good 4-D start pose in raw joint space:
    [swing, arm, stick, bucket]

The goal should remain a (3, 3) array, matching the training/inference code.
"""

from __future__ import annotations

from pathlib import Path
import time
from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np
import torch
from torch import Tensor, nn
from torchvision.transforms import Normalize
import warp as wp
import newton
import newton.examples


URDF_RELATIVE_PATH = "./excavatorURDF/excavator_lowpoly_locked_splitbucket.urdf"
CHECKPOINT_DIR = "./bc_component/outputs/ssmlp_js_dag"
CHECKPOINT_NAME = "best.pt"

VIEWER_FPS = 60.0
COMMAND_HZ = 10.0

# Fill this in with a good raw 4-joint pose: [swing, arm, stick, bucket]
STARTING_POLICY_Q_RAW = np.array([
    0.0,
    0.0,
    0.0,
    0.0,
], dtype=np.float32)

# Replace with your desired goal if needed. Shape must be (3, 3).
HARDCODED_GOAL = np.array(
    [
        [0.8, 7.4940055485687e-17, 0.0],
        [0.8, 7.4940055485687e-17, 2.0],
        [0.8, 7.4940055485687e-17, 0.0],
    ],
    dtype=np.float32,
)


class SingleStepMLP(nn.Module):
    def __init__(
        self,
        state_dim: int,
        goal_dim: int,
        action_dim: int,
        hidden_dims: Sequence[int],
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        dims = [state_dim + goal_dim, *hidden_dims, action_dim]
        layers: list[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
                if dropout > 0.0:
                    layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, state: Tensor, goal: Tensor) -> Tensor:
        x = torch.cat([state, goal], dim=-1)
        return self.net(x)


def choose_device(requested: str = "auto") -> torch.device:
    requested_l = str(requested).lower()
    if requested_l == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Requested device 'cuda' but CUDA is not available.")
        return torch.device("cuda")
    if requested_l == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _as_float_tensor(x: Any, device: torch.device | None = None) -> Tensor:
    t = torch.as_tensor(x, dtype=torch.float32)
    if device is not None:
        t = t.to(device)
    return t


def scale_joint_to_unit(raw: Tensor, raw_min: Tensor, raw_max: Tensor) -> Tensor:
    return 2.0 * (raw - raw_min) / (raw_max - raw_min) - 1.0


def unscale_joint_from_unit(unit: Tensor, raw_min: Tensor, raw_max: Tensor) -> Tensor:
    return 0.5 * (unit + 1.0) * (raw_max - raw_min) + raw_min


class JointScaledSSMLPPolicy:
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        joint_raw_min: Tensor,
        joint_raw_max: Tensor,
        goal_mean_grouped: Tensor,
        goal_std_grouped: Tensor,
        joint_indices: Sequence[int] | None = None,
        clamp_output: bool = True,
    ) -> None:
        self.model = model
        self.device = device
        self.joint_raw_min = _as_float_tensor(joint_raw_min, device)
        self.joint_raw_max = _as_float_tensor(joint_raw_max, device)
        self.goal_norm = Normalize(
            mean=_as_float_tensor(goal_mean_grouped).tolist(),
            std=_as_float_tensor(goal_std_grouped).tolist(),
        )
        self.joint_indices = list(joint_indices) if joint_indices is not None else None
        self.clamp_output = clamp_output

        if self.joint_raw_min.shape != (4,) or self.joint_raw_max.shape != (4,):
            raise ValueError(
                "Expected joint_raw_min and joint_raw_max to each have shape (4,), "
                f"got {tuple(self.joint_raw_min.shape)} and {tuple(self.joint_raw_max.shape)}"
            )
        if not torch.all(self.joint_raw_max > self.joint_raw_min):
            raise ValueError("Every entry in joint_raw_max must be strictly greater than joint_raw_min.")

    def preprocess_goal(self, goal: np.ndarray | Sequence[float] | Tensor) -> Tensor:
        goal_t = _as_float_tensor(goal, self.device)
        if goal_t.numel() != 9:
            raise ValueError(f"Expected goal to contain 9 values, got shape {tuple(goal_t.shape)}")

        goal_t = goal_t.reshape(3, 3)
        g = goal_t.t().unsqueeze(-1)
        g = self.goal_norm(g)
        g = g.squeeze(-1).t().reshape(1, 9)
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
        return q_t.unsqueeze(0)

    @torch.no_grad()
    def act(
        self,
        current_joint_positions: np.ndarray | Sequence[float] | Tensor,
        goal: np.ndarray | Sequence[float] | Tensor,
    ) -> np.ndarray:
        state_t = self.preprocess_state(current_joint_positions)
        goal_t = self.preprocess_goal(goal)
        pred_unit = self.model(state_t, goal_t).squeeze(0)
        pred_raw = unscale_joint_from_unit(pred_unit, self.joint_raw_min, self.joint_raw_max)
        if self.clamp_output:
            pred_raw = torch.clamp(pred_raw, min=self.joint_raw_min, max=self.joint_raw_max)
        return pred_raw.detach().cpu().numpy()


def _extract_norm_config(config: Mapping[str, Any]) -> Mapping[str, Any]:
    norm_cfg = config.get("normalization")
    if norm_cfg is None:
        raise KeyError("Checkpoint config does not contain a 'normalization' section.")

    required_keys = [
        "joint_raw_min",
        "joint_raw_max",
        "goal_mean_grouped",
        "goal_std_grouped",
    ]
    missing = [k for k in required_keys if k not in norm_cfg]
    if missing:
        raise KeyError(f"Checkpoint normalization config is missing keys: {missing}")
    return norm_cfg


def load_ssmlp_policy(
    checkpoint_dir: str | Path,
    which: str = "best.pt",
    device: str = "auto",
    joint_indices: Sequence[int] | None = None,
    clamp_output: bool = True,
) -> JointScaledSSMLPPolicy:
    device_obj = choose_device(device)
    checkpoint_path = Path(checkpoint_dir) / which
    ckpt = torch.load(checkpoint_path, map_location=device_obj)

    if "config" not in ckpt or "model_state_dict" not in ckpt:
        raise KeyError(
            f"Checkpoint at {checkpoint_path} must contain 'config' and 'model_state_dict'."
        )

    config = ckpt["config"]
    model_cfg = config["model"]
    norm_cfg = _extract_norm_config(config)

    model = SingleStepMLP(
        state_dim=int(model_cfg["state_dim"]),
        goal_dim=int(model_cfg["goal_dim"]),
        action_dim=int(model_cfg["action_dim"]),
        hidden_dims=model_cfg["hidden_dims"],
        dropout=float(model_cfg.get("dropout", 0.0)),
    ).to(device_obj)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    return JointScaledSSMLPPolicy(
        model=model,
        device=device_obj,
        joint_raw_min=_as_float_tensor(norm_cfg["joint_raw_min"]),
        joint_raw_max=_as_float_tensor(norm_cfg["joint_raw_max"]),
        goal_mean_grouped=_as_float_tensor(norm_cfg["goal_mean_grouped"]),
        goal_std_grouped=_as_float_tensor(norm_cfg["goal_std_grouped"]),
        joint_indices=joint_indices,
        clamp_output=clamp_output,
    )


class ExcavatorPolicyKinematicSweep:
    def __init__(self, viewer) -> None:
        self.viewer = viewer
        self.device = wp.get_device()

        self.urdf_path = (URDF_RELATIVE_PATH)
        self.checkpoint_dir = (CHECKPOINT_DIR)

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

        self.policy = load_ssmlp_policy(
            checkpoint_dir=self.checkpoint_dir,
            which=CHECKPOINT_NAME,
            device="auto",
            joint_indices=None,
            clamp_output=True,
        )

        self.current_policy_q = np.asarray(STARTING_POLICY_Q_RAW, dtype=np.float32).copy()
        if self.current_policy_q.shape != (4,):
            raise ValueError(
                f"STARTING_POLICY_Q_RAW must have shape (4,), got {self.current_policy_q.shape}"
            )

        self.command_start_q = self.current_policy_q.copy()
        self.command_target_q = self.current_policy_q.copy()

        self._apply_policy_joints(self.current_policy_q)

        self.viewer.set_model(self.model)
        self.viewer.show_particles = False
        self.viewer.show_visual = True
        self.viewer.show_collision = False
        self.viewer.show_cloth = False

        print(f"Loaded URDF: {self.urdf_path}")
        print(f"Joint map: {self.joint_map}")
        print(f"Policy joint indices: {self.policy_joint_indices}")
        print(f"Starting policy q: {self.current_policy_q}")
        print(f"Goal: {HARDCODED_GOAL}")

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
        names = ("swing", "arm", "stick", "bucket")
        indices: list[int] = []
        for name in names:
            idx = self.joint_map.get(name)
            if idx is None:
                raise ValueError(f"Could not resolve a URDF joint index for '{name}'.")
            indices.append(int(idx))
        return indices

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

    def _advance_policy(self) -> None:
        current_joint_positions = self.current_policy_q.copy()
        next_q = np.asarray(
            self.policy.act(current_joint_positions, HARDCODED_GOAL),
            dtype=np.float32,
        )
        if next_q.shape != (4,):
            raise ValueError(f"Policy returned shape {next_q.shape}, expected (4,)")

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
    example = ExcavatorPolicyKinematicSweep(viewer)

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
