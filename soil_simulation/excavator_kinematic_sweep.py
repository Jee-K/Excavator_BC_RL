#!/usr/bin/env python3
"""
Grounded, no-physics excavator URDF replay.

This script loads the excavator URDF, reads `target_angles.npz` from the same
folder, and replays the joint targets kinematically at 10 Hz with linear
interpolation for smooth viewing. It does not add soil, a floor, or run any
physics stepping.
"""

from __future__ import annotations

from pathlib import Path
import time
from typing import Iterable, Optional

import numpy as np
import warp as wp
import newton
import newton.examples


NPZ_NAME = "./bc_component/BC_dataset/swing/swing_3_bc_dataset.npz"
URDF_RELATIVE_PATH = "./excavatorURDF/excavator_lowpoly_locked_splitbucket.urdf"
TARGET_HZ = 10.0
VIEWER_FPS = 60.0
LOOP_TRAJECTORY = True


class ExcavatorKinematicSweep:
    def __init__(self, viewer) -> None:
        self.viewer = viewer
        self.device = wp.get_device()
        self.script_dir = Path(__file__).resolve().parent
        self.urdf_path = URDF_RELATIVE_PATH
        self.target_npz_path = "./" + NPZ_NAME

        self.frame_dt = 1.0 / VIEWER_FPS
        self.sim_time = 0.0

        self.target_series = self._load_target_series(self.target_npz_path)
        self.source_hz = TARGET_HZ
        self.duration = max((len(self.target_series) - 1) / self.source_hz, 0.0)

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
        self.data_columns = self._infer_data_columns(self.target_series.shape[1])

        newton.eval_fk(self.model, self.state.joint_q, self.state.joint_qd, self.state)

        self.viewer.set_model(self.model)
        self.viewer.show_particles = False
        self.viewer.show_visual = True
        self.viewer.show_collision = False
        self.viewer.show_cloth = False

        print(f"Loaded URDF: {self.urdf_path}")
        print(f"Loaded targets: {self.target_npz_path}")
        print(f"Target series shape: {self.target_series.shape}")
        print(f"Playback duration: {self.duration:.2f}s at {self.source_hz:.1f} Hz")
        print(f"Joint map: {self.joint_map}")
        print(f"Data columns: {self.data_columns}")

    @staticmethod
    def _load_target_series(npz_path: Path) -> np.ndarray:
        preferred_keys = ("target_angles", "target_states", "states", "targets", "target", "arr_0")
        with np.load(npz_path, allow_pickle=False) as data:
            keys = list(data.keys())
            for key in preferred_keys:
                if key in data:
                    arr = np.asarray(data[key])
                    if arr.ndim == 2 and np.issubdtype(arr.dtype, np.number):
                        return arr.astype(np.float32, copy=False)
            for key in keys:
                arr = np.asarray(data[key])
                if arr.ndim == 2 and np.issubdtype(arr.dtype, np.number):
                    return arr.astype(np.float32, copy=False)
        raise ValueError(f"No numeric 2D array found in {npz_path}")

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

    @staticmethod
    def _infer_data_columns(data_dim: int) -> dict[str, int]:
        if data_dim == 4:
            return {"swing": 0, "arm": 1, "stick": 2, "bucket": 3}
        if data_dim >= 10:
            return {"swing": 6, "arm": 7, "stick": 8, "bucket": 9}
        if data_dim >= 4:
            return {"swing": 0, "arm": 1, "stick": 2, "bucket": 3}
        raise ValueError(f"Target series must have at least 4 columns, got {data_dim}")

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

    def _sample_targets(self, t: float) -> np.ndarray:
        seq = self.target_series
        if len(seq) == 0:
            raise ValueError("Target series is empty")
        if len(seq) == 1:
            return seq[0]

        if LOOP_TRAJECTORY and self.duration > 0.0:
            t = t % self.duration
        else:
            t = float(np.clip(t, 0.0, self.duration))

        s = t * self.source_hz
        i0 = int(np.floor(s))
        i1 = min(i0 + 1, len(seq) - 1)
        u = float(s - i0)
        return (1.0 - u) * seq[i0] + u * seq[i1]

    def _apply_targets(self, sample: np.ndarray) -> None:
        q = self.q_init.copy()
        for name in ("swing", "arm", "stick", "bucket"):
            joint_idx = self.joint_map.get(name)
            data_idx = self.data_columns[name]
            if joint_idx is None or data_idx >= sample.shape[0]:
                continue
            q[joint_idx] = self._clip_target(joint_idx, float(sample[data_idx]))

        self.joint_q_host[...] = q
        self.joint_qd_host[...] = 0.0
        self.state.joint_q.assign(self.joint_q_host)
        self.state.joint_qd.assign(self.joint_qd_host)
        newton.eval_fk(self.model, self.state.joint_q, self.state.joint_qd, self.state)

    def step(self) -> None:
        sample = self._sample_targets(self.sim_time)
        self._apply_targets(sample)

        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state)
        self.viewer.end_frame()

        self.sim_time += self.frame_dt


def main() -> None:
    viewer, _args = newton.examples.init()
    example = ExcavatorKinematicSweep(viewer)

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
