from pathlib import Path
from typing import Iterable, Iterator

import h5py
import numpy as np
import os

KEYPOINT_GOAL_INDEX = 6

def build_segments(trajectories: np.ndarray, keypoints_cylindrical, lead_frames_before_dig : int = 20, close_frames_before_reset : int = 0, bucket_flat_angle : float = 1.1) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    With trajectories in Nx[base, boom, arm, scoop], lead frames and close frames being nonnegative ints, and bucket_flat_angle being a float-like representing radians
    """
  
    candidates = []
  
    for idx, step in enumerate(trajectories[1:-1]):
  
        # step consists of [turret, main boom, lower to upper, scoop]
        # [..., down, up, in] are the positive directions
        last_step_angle = trajectories[idx - 1][1] - trajectories[idx - 1][2] + trajectories[idx - 1][3]
        step_angle = step[1] - step[2] + step[3]
        next_step_angle = trajectories[idx + 1][1] - trajectories[idx + 1][2] + trajectories[idx + 1][3]


        # dig heuristic
        if step_angle >= bucket_flat_angle and (last_step_angle < step_angle >= next_step_angle):
            candidates.append((idx, "Dig"))

        # dump heuristic
        elif step_angle <= bucket_flat_angle and (last_step_angle > step_angle <= next_step_angle):
            candidates.append((idx, "Dump"))
  
    # add the last position as a false-dig which will serve as a return for the last segment, if no good dig exists
    # strictly speaking, this is a shortcut to avoid a loss of training data which may otherwise be useful even when
    # a well formed end goal does not exist
    candidates.append((len(trajectories), "Dig"))
  
    # form segments
    segments : list[tuple[int, int, int]] = []
  
    idx = 0
    while idx < len(candidates) and candidates[idx][1] != 'Dig':
        idx += 1
  
    last_dig_idx = idx
    idx += 1
  
    while idx < len(candidates):
        while idx < len(candidates) and candidates[idx][1] != 'Dump':
            idx += 1

        if idx >= len(candidates):
            break

        dump_idx = idx

        while idx < len(candidates) and candidates[idx][1] != 'Dig':
            idx += 1

        if idx >= len(candidates):
            break

        segments.append((last_dig_idx, dump_idx, idx))
        last_dig_idx = idx

    labeled_segments = []
  
    for dig_idx, dump_idx, reset_idx in segments:
  
        start_idx = max(0, dig_idx - lead_frames_before_dig)
        seq_end_idx = int(np.clip(reset_idx + 1 - close_frames_before_reset, 0, len(trajectories)))

        # !!! this pulls it in r, h, theta
        segment_goal = np.asarray([keypoints_cylindrical[dig_idx][KEYPOINT_GOAL_INDEX], keypoints_cylindrical[dump_idx][KEYPOINT_GOAL_INDEX], keypoints_cylindrical[reset_idx][KEYPOINT_GOAL_INDEX]], dtype=np.float32)
        segment_traj = np.asarray(trajectories[start_idx:seq_end_idx], dtype=np.float32)

        labeled_segments.append((segment_goal, segment_traj))

    return labeled_segments


def iter_joint_angle_files(source_dirs: Iterable[str | Path]) -> Iterator[Path]:
    """
    Yield every joint_angles.npz under:
        <source_dir>/sequences/**/joint_angles.npz
    """
    for root in map(Path, source_dirs):
        seq_root = root / "sequences"
        if not seq_root.is_dir():
            continue
        yield from sorted(seq_root.rglob("joint_angles.npz"))

def build_segment_store(
    seq_data_path: Iterable[str],
    output_path: str,
    manifest_path: str,
    n_max: int,
) -> Path:
    """
    Build one HDF5 file containing:
      - goals:        (S, 3, 3)
      - trajectories: (S, n_max, 4)
      - lengths:      (S,)

    The builder pads each trajectory to n_max rows. If a produced trajectory is longer
    than n_max, it raises ValueError instead of silently truncating.
    """
    assert n_max > 0

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    goals_list: list[np.ndarray] = []
    trajs_list: list[np.ndarray] = []
    num_source_files = 0

    included_paths = []

    for npz_path in iter_joint_angle_files([Path(seq_data_path).joinpath(x) for x in os.listdir(seq_data_path) if os.path.isdir(Path(seq_data_path).joinpath(x))]):
        num_source_files += 1
        included_paths.append(str(npz_path))

        with np.load(npz_path) as data:
            _cabin_yaw : np.ndarray = data["cabin_yaw_rad"]
            initial_joint_angles : np.ndarray = data["joint_angles"]
            joint_angles = np.concatenate([_cabin_yaw.reshape(initial_joint_angles.shape[0], 1), initial_joint_angles], axis=1) # seems to work right, but a little dubious
            keypoints_cylindrical = data["keypoints_cylindrical"]

        segments = build_segments(joint_angles, keypoints_cylindrical)

        for segment in segments:
            goal, traj = segment
            if traj.shape[0] > n_max:
                raise ValueError(
                    f"Trajectory length {traj.shape[0]} exceeds n_max={n_max}. "
                    f"Offending file: {npz_path}"
                )
            goals_list.append(goal)
            trajs_list.append(traj)

    if not goals_list:
        raise RuntimeError(
            "No segments were produced."
        )

    num_segments = len(goals_list)

    goals = np.stack(goals_list, axis=0).astype(np.float32)  # (S, 3, 3)
    trajectories = np.zeros((num_segments, n_max, 4), dtype=np.float32)
    lengths = np.zeros((num_segments,), dtype=np.int64)

    for i, traj in enumerate(trajs_list):
        n = traj.shape[0]
        trajectories[i, :n] = traj
        lengths[i] = n

    with h5py.File(output_path, "w") as f:
        f.create_dataset("goals", data=goals, compression="gzip")
        f.create_dataset("trajectories", data=trajectories, compression="gzip")
        f.create_dataset("lengths", data=lengths)

        f.attrs["num_source_files"] = num_source_files
        f.attrs["num_segments"] = num_segments
        f.attrs["n_max"] = n_max

    with open(manifest_path, "w") as f:
        f.write("\n".join(included_paths))

    return output_path


def main() -> None:
    output_file = build_segment_store(
        seq_data_path="./seq_data/",
        output_path="./training_data/latest.h5",
        manifest_path="./training_data/latest_manifest.txt",
        n_max=400,
    )
    print(f"Wrote segment store to: {output_file}")


if __name__ == "__main__":
    main()
