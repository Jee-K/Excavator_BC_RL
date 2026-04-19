#!/usr/bin/env python3
"""
Physics-less excavator rollout for the delta-target sequence MLP policy.

This is the kinematic sweep analogue of run_sequence_policy_mlp_delta.py:
- load the excavator URDF,
- load a trained delta-target sequence MLP checkpoint,
- drive the excavator kinematically without stepping physics,
- query the policy at COMMAND_HZ,
- interpolate between commanded joint targets at VIEWER_FPS,
- execute the first `EXECUTE_STEPS_PER_INFERENCE` steps from each predicted chunk
  before replanning.
"""
from __future__ import annotations

from collections import deque
from pathlib import Path
import time
from typing import Iterable, Optional

import numpy as np
import warp as wp
import newton
import newton.examples

from run_sequence_policy_mlp_delta import load_delta_sequence_policy


URDF_RELATIVE_PATH = "./excavatorURDF/excavator_lowpoly_locked_splitbucket.urdf"
CHECKPOINT_DIR = "./bc_component/outputs/emlp_deltas_small_mse"
CHECKPOINT_NAME = "best.pt"

VIEWER_FPS = 60.0
COMMAND_HZ = 10.0
EXECUTE_STEPS_PER_INFERENCE = 6

STARTING_POLICY_Q_RAW = np.array([
    0.0,
    0.0,
    -0.9,
    0.8,
], dtype=np.float32)

HARDCODED_GOAL = np.array([
    [3.0, -0.8, 0.0],
    [4.0, 0.8, 0.5 * np.pi],
    [3.0, -0.8, 0.0],
], dtype=np.float32)


class ExcavatorDeltaSequencePolicyKinematicSweep:
    def __init__(self, viewer) -> None:
        if EXECUTE_STEPS_PER_INFERENCE <= 0:
            raise ValueError(
                f"EXECUTE_STEPS_PER_INFERENCE must be positive, got {EXECUTE_STEPS_PER_INFERENCE}"
            )

        self.viewer = viewer
        self.device = wp.get_device()

        self.urdf_path = Path(URDF_RELATIVE_PATH)
        self.checkpoint_dir = Path(CHECKPOINT_DIR)

        self.frame_dt = 1.0 / VIEWER_FPS
        self.command_dt = 1.0 / COMMAND_HZ
        self.sim_time = 0.0
        self.command_phase_time = 0.0
        self.pending_policy_targets: deque[np.ndarray] = deque()

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

        self.policy = load_delta_sequence_policy(
            checkpoint_dir=self.checkpoint_dir,
            which=CHECKPOINT_NAME,
            device="auto",
            joint_indices=None,
            clamp_output=True,
        )
        self.policy.reset_history()

        self.current_policy_q = np.asarray(STARTING_POLICY_Q_RAW, dtype=np.float32).copy()
        if self.current_policy_q.shape != (4,):
            raise ValueError(
                f"STARTING_POLICY_Q_RAW must have shape (4,), got {self.current_policy_q.shape}"
            )

        self.command_start_q = self.current_policy_q.copy()
        self.command_target_q = self.current_policy_q.copy()
        self.last_predicted_chunk: Optional[np.ndarray] = None

        self._apply_policy_joints(self.current_policy_q)

        self.viewer.set_model(self.model)
        self.viewer.show_particles = False
        self.viewer.show_visual = True
        self.viewer.show_collision = False
        self.viewer.show_cloth = False

        print(f"Loaded URDF: {self.urdf_path}")
        print(f"Checkpoint dir: {self.checkpoint_dir}")
        print(f"Joint map: {self.joint_map}")
        print(f"Policy joint indices: {self.policy_joint_indices}")
        print(f"Starting policy q: {self.current_policy_q}")
        print(f"Goal: {HARDCODED_GOAL}")
        print(f"EXECUTE_STEPS_PER_INFERENCE: {EXECUTE_STEPS_PER_INFERENCE}")

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

    def _advance_policy(self) -> None:
        self.policy.observe(self.current_policy_q.copy())

        if not self.pending_policy_targets:
            pred_chunk = np.asarray(self.policy.predict_chunk(HARDCODED_GOAL), dtype=np.float32)
            if pred_chunk.ndim != 2 or pred_chunk.shape[1] != 4:
                raise ValueError(f"Policy returned chunk with shape {pred_chunk.shape}, expected (K, 4)")

            self.last_predicted_chunk = pred_chunk
            run_len = min(int(EXECUTE_STEPS_PER_INFERENCE), int(pred_chunk.shape[0]))
            for step_target in pred_chunk[:run_len]:
                self.pending_policy_targets.append(np.asarray(step_target, dtype=np.float32))

            print()
            print("current:", self.current_policy_q)
            print("predicted chunk:")
            print(pred_chunk)
            print(f"queued {run_len} step(s) before replanning")

        next_q = self.pending_policy_targets.popleft()
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
    example = ExcavatorDeltaSequencePolicyKinematicSweep(viewer)

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
