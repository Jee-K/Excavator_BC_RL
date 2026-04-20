from pathlib import Path
from typing import Iterable, Iterator
import csv
import os

import h5py
import numpy as np

KEYPOINT_GOAL_INDEX = 0 # 0 - scoop tip, 1 - scoop middle, 2 - scoop base
ROTATION_LIMIT = 1.5 * np.pi
ROTATION_STEP = 0.1
EXCEL_ROW_LIMIT = 1_048_576
EXCEL_SAFETY_MARGIN = 5_000

BEHAVIOR_STREAM_DIG_SWING = "dig_swing"
BEHAVIOR_STREAM_DUMP_RETURN = "dump_return"
BEHAVIOR_STREAMS = (BEHAVIOR_STREAM_DIG_SWING, BEHAVIOR_STREAM_DUMP_RETURN)
GOAL_SOURCE_INDEX_BY_BEHAVIOR = {
    BEHAVIOR_STREAM_DIG_SWING: 1,  # keep only the second 1/3 (dump waypoint)
    BEHAVIOR_STREAM_DUMP_RETURN: 2,  # keep only the third 1/3 (reset waypoint)
}
GOAL_KEEP_COORD_INDICES = (0, 2)  # drop the middle / 2nd column entirely


def reduce_goal_for_behavior(full_cycle_goal: np.ndarray, behavior_stream: str) -> np.ndarray:
    full_cycle_goal = np.asarray(full_cycle_goal, dtype=np.float32)
    goal_source_index = GOAL_SOURCE_INDEX_BY_BEHAVIOR[behavior_stream]
    selected = full_cycle_goal[goal_source_index]
    return np.asarray(selected[list(GOAL_KEEP_COORD_INDICES)], dtype=np.float32)


def find_candidates(
    trajectories: np.ndarray,
    bucket_flat_angle: float = 1.1,
) -> list[tuple[int, str]]:
    """
    Mirror the original candidate detection logic.
    Returns tuples of (trajectory_index, label), where label is "Dig" or "Dump".
    """
    candidates: list[tuple[int, str]] = []

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

    # add the last position as a false-dig which will serve as a return for the last segment,
    # if no good dig exists
    candidates.append((len(trajectories), "Dig"))

    # ??? Motivation here isn't perfect, but helps create some additional segments. Remove if the creation is particularly unstable or data large
    # if the first candidate is not a dig, add a false-dig at 0
    if len(candidates) > 0 and candidates[0][1] != "Dig":
        candidates.insert(0, (0, "Dig"))

    return candidates



def find_segment_candidate_triples(candidates: list[tuple[int, str]]) -> list[tuple[int, int, int]]:
    """
    Mirror the original segment pairing logic exactly.

    IMPORTANT:
    These triples are candidate list positions, not trajectory frame indices.
    That matches the current behavior of the original script.
    """
    segments: list[tuple[int, int, int]] = []

    idx = 0
    while idx < len(candidates) and candidates[idx][1] != "Dig":
        idx += 1

    last_dig_idx = idx
    idx += 1

    while idx < len(candidates):
        while idx < len(candidates) and candidates[idx][1] != "Dump":
            idx += 1

        if idx >= len(candidates):
            break

        dump_idx = idx

        while idx < len(candidates) and candidates[idx][1] != "Dig":
            idx += 1

        if idx >= len(candidates):
            break

        segments.append((last_dig_idx, dump_idx, idx))
        last_dig_idx = idx

    return segments



def build_segments_detailed(
    trajectories: np.ndarray,
    keypoints_cylindrical: np.ndarray,
    lead_frames_before_dig: int = 20,
    close_frames_before_reset: int = 0,
    bucket_flat_angle: float = 1.1,
    *,
    mirror_original_indexing: bool = True,
) -> list[dict]:
    """
    Build two detailed segment records for every dig->dump->return cycle:
      - dig_swing   : cycle start through the dump timestep
      - dump_return : dump timestep through the return / next dig timestep

    Each produced segment stores a reduced 2-D goal vector. For dig_swing this is the
    second 1/3 of the cycle goal, and for dump_return it is the third 1/3. In both
    cases the middle coordinate is removed, so the stored goal is [col0, col2].

    When mirror_original_indexing=True (default), the goal extraction and trajectory slicing
    use the same indexes as the original script, so this builder stays aligned with the
    current HDF5 semantics.

    keypoints_cylindrical[...][KEYPOINT_GOAL_INDEX] is interpreted as [r, h, theta].
    After reduction, the stored goal is [r, theta], so rotational augmentation modifies
    the second entry of the reduced goal.
    """
    candidates = find_candidates(trajectories, bucket_flat_angle=bucket_flat_angle)
    segments = find_segment_candidate_triples(candidates)

    records: list[dict] = []
    next_record_index = 0

    for parent_cycle_index_in_file, (dig_candidate_pos, dump_candidate_pos, reset_candidate_pos) in enumerate(segments):
        dig_event_frame = int(candidates[dig_candidate_pos][0])
        dump_event_frame = int(candidates[dump_candidate_pos][0])
        reset_event_frame = int(candidates[reset_candidate_pos][0])

        if mirror_original_indexing:
            dig_goal_idx = int(dig_candidate_pos)
            dump_goal_idx = int(dump_candidate_pos)
            reset_goal_idx = int(reset_candidate_pos)

            dig_swing_start_idx = max(0, int(dig_candidate_pos) - lead_frames_before_dig)
            dig_swing_end_idx = int(np.clip(int(dump_candidate_pos) + 1, 0, len(trajectories)))
            dump_return_start_idx = int(np.clip(int(dump_candidate_pos), 0, len(trajectories)))
            dump_return_end_idx = int(np.clip(int(reset_candidate_pos) + 1 - close_frames_before_reset, 0, len(trajectories)))
        else:
            dig_goal_idx = dig_event_frame
            dump_goal_idx = dump_event_frame
            reset_goal_idx = reset_event_frame

            dig_swing_start_idx = max(0, dig_event_frame - lead_frames_before_dig)
            dig_swing_end_idx = int(np.clip(dump_event_frame + 1, 0, len(trajectories)))
            dump_return_start_idx = int(np.clip(dump_event_frame, 0, len(trajectories)))
            dump_return_end_idx = int(np.clip(reset_event_frame + 1 - close_frames_before_reset, 0, len(trajectories)))

        full_cycle_goal = np.asarray(
            [
                keypoints_cylindrical[dig_goal_idx][KEYPOINT_GOAL_INDEX],
                keypoints_cylindrical[dump_goal_idx][KEYPOINT_GOAL_INDEX],
                keypoints_cylindrical[reset_goal_idx][KEYPOINT_GOAL_INDEX],
            ],
            dtype=np.float32,
        )

        stream_specs = [
            (BEHAVIOR_STREAM_DIG_SWING, 0, dig_swing_start_idx, dig_swing_end_idx),
            (BEHAVIOR_STREAM_DUMP_RETURN, 1, dump_return_start_idx, dump_return_end_idx),
        ]

        for behavior_stream, behavior_stream_index, start_idx, seq_end_idx in stream_specs:
            if seq_end_idx <= start_idx:
                continue

            segment_traj = np.asarray(trajectories[start_idx:seq_end_idx], dtype=np.float32)

            records.append(
                {
                    "segment_index_in_file": int(next_record_index),
                    "parent_cycle_index_in_file": int(parent_cycle_index_in_file),
                    "behavior_stream": behavior_stream,
                    "behavior_stream_index": int(behavior_stream_index),
                    "dig_candidate_pos": int(dig_candidate_pos),
                    "dump_candidate_pos": int(dump_candidate_pos),
                    "reset_candidate_pos": int(reset_candidate_pos),
                    "dig_event_frame": int(dig_event_frame),
                    "dump_event_frame": int(dump_event_frame),
                    "reset_event_frame": int(reset_event_frame),
                    "goal_index_used_for_dig": int(dig_goal_idx),
                    "goal_index_used_for_dump": int(dump_goal_idx),
                    "goal_index_used_for_reset": int(reset_goal_idx),
                    "start_idx_used": int(start_idx),
                    "seq_end_idx_used": int(seq_end_idx),
                    "length": int(segment_traj.shape[0]),
                    "goal": reduce_goal_for_behavior(full_cycle_goal, behavior_stream),
                    "full_cycle_goal": full_cycle_goal,
                    "goal_source_index": int(GOAL_SOURCE_INDEX_BY_BEHAVIOR[behavior_stream]),
                    "trajectory": segment_traj,
                }
            )
            next_record_index += 1

    return records



def iter_joint_angle_files(source_dirs: Iterable[str | Path]) -> Iterator[Path]:
    """
    Yield every joint_angles.npz under:
        <source_dir>/sequences/**/joint_angles.npz
    """
    for root in map(Path, source_dirs):
        seq_root = root / "sequences"
        if not seq_root.is_dir():
            continue
        yield from sorted(seq_root.rglob("joint_angles_v2.npz"))



def _source_dirs_under(seq_data_path: str | Path) -> list[Path]:
    seq_data_path = Path(seq_data_path)
    return [seq_data_path.joinpath(x) for x in os.listdir(seq_data_path) if seq_data_path.joinpath(x).is_dir()]



def iter_base_segment_records(
    seq_data_path: str | Path,
    lead_frames_before_dig: int = 20,
    close_frames_before_reset: int = 0,
    bucket_flat_angle: float = 1.1,
    *,
    mirror_original_indexing: bool = True,
) -> Iterator[dict]:
    global_base_segment_index = 0

    for npz_path in iter_joint_angle_files(_source_dirs_under(seq_data_path)):
        with np.load(npz_path) as data:
            cabin_yaw = data["cabin_yaw_rad"]
            initial_joint_angles = data["joint_angles"]
            joint_angles = np.concatenate(
                [cabin_yaw.reshape(initial_joint_angles.shape[0], 1), initial_joint_angles],
                axis=1,
            )
            keypoints_cylindrical = data["keypoints_cylindrical"]

        records = build_segments_detailed(
            joint_angles,
            keypoints_cylindrical,
            lead_frames_before_dig=lead_frames_before_dig,
            close_frames_before_reset=close_frames_before_reset,
            bucket_flat_angle=bucket_flat_angle,
            mirror_original_indexing=mirror_original_indexing,
        )

        for record in records:
            record = dict(record)
            record["global_base_segment_index"] = global_base_segment_index
            record["source_npz_path"] = str(npz_path)
            global_base_segment_index += 1
            yield record



def compute_valid_rotation_offsets(
    traj: np.ndarray,
    goal: np.ndarray,
    rotation_limit: float = ROTATION_LIMIT,
    rotation_step: float = ROTATION_STEP,
) -> np.ndarray:
    """
    Compute all additive offsets on a global rotation_step grid that keep the rotatable
    dimensions within [-rotation_limit, rotation_limit].

    Rotated fields:
      - traj[:, 0]     : cabin/turret rotation in joint-control space
      - goal[-1]       : theta in the reduced goal representation [col0, col2]
    """
    if rotation_step <= 0:
        raise ValueError("rotation_step must be positive")

    goal = np.asarray(goal, dtype=np.float32).reshape(-1)
    if goal.shape != (2,):
        raise ValueError(f"Expected reduced goal shape (2,), got {goal.shape}")
    rot_values = np.concatenate([traj[:, 0], goal[-1:]]).astype(np.float64)
    lower_bound = float(np.max(-rotation_limit - rot_values))
    upper_bound = float(np.min(rotation_limit - rot_values))

    eps = 1e-9
    if lower_bound > upper_bound + eps:
        raise ValueError(
            "No valid rotation offset exists for this segment under the requested bounds. "
            f"Computed interval: [{lower_bound}, {upper_bound}]"
        )

    start_k = int(np.ceil((lower_bound - eps) / rotation_step))
    end_k = int(np.floor((upper_bound + eps) / rotation_step))

    if end_k < start_k:
        midpoint = 0.5 * (lower_bound + upper_bound)
        snapped = rotation_step * round(midpoint / rotation_step)
        clipped = min(max(snapped, lower_bound), upper_bound)
        return np.asarray([np.float32(clipped)], dtype=np.float32)

    offsets = (np.arange(start_k, end_k + 1, dtype=np.float64) * rotation_step).astype(np.float64)

    if lower_bound <= 0.0 <= upper_bound and not np.any(np.isclose(offsets, 0.0, atol=1e-9)):
        offsets = np.concatenate([offsets, np.asarray([0.0])])

    offsets = np.unique(np.round(offsets, decimals=10))
    offsets.sort()
    return offsets.astype(np.float32)



def iter_augmented_segment_records(
    seq_data_path: str | Path,
    lead_frames_before_dig: int = 20,
    close_frames_before_reset: int = 0,
    bucket_flat_angle: float = 1.1,
    *,
    mirror_original_indexing: bool = True,
    rotation_limit: float = ROTATION_LIMIT,
    rotation_step: float = ROTATION_STEP,
) -> Iterator[dict]:
    global_augmented_segment_index = 0

    for base_record in iter_base_segment_records(
        seq_data_path=seq_data_path,
        lead_frames_before_dig=lead_frames_before_dig,
        close_frames_before_reset=close_frames_before_reset,
        bucket_flat_angle=bucket_flat_angle,
        mirror_original_indexing=mirror_original_indexing,
    ):
        goal = np.asarray(base_record["goal"], dtype=np.float32)
        traj = np.asarray(base_record["trajectory"], dtype=np.float32)
        offsets = compute_valid_rotation_offsets(
            traj,
            goal,
            rotation_limit=rotation_limit,
            rotation_step=rotation_step,
        )

        for offset in offsets:
            augmented = dict(base_record)
            augmented["rotation_offset"] = float(offset)
            augmented["global_augmented_segment_index"] = global_augmented_segment_index
            augmented["num_rotations_for_base_segment"] = int(len(offsets))

            rotated_goal = goal.copy()
            rotated_goal[-1] += np.float32(offset)

            rotated_traj = traj.copy()
            rotated_traj[:, 0] += np.float32(offset)

            augmented["goal"] = rotated_goal
            augmented["trajectory"] = rotated_traj

            global_augmented_segment_index += 1
            yield augmented



def build_segment_store(
    seq_data_path: str | Path,
    output_path: str | Path,
    manifest_path: str | Path,
    n_max: int,
    *,
    lead_frames_before_dig: int = 20,
    close_frames_before_reset: int = 0,
    bucket_flat_angle: float = 1.1,
    mirror_original_indexing: bool = True,
    rotation_limit: float = ROTATION_LIMIT,
    rotation_step: float = ROTATION_STEP,
) -> Path:
    """
    Build one HDF5 file containing:
      - goals:               (S, 2) reduced per-stream goals [col0, col2]
      - trajectories:        (S, n_max, 4)
      - lengths:             (S,)
      - rotation_offsets:     (S,)
      - base_segment_indices: (S,)
      - behavior_stream_indices: (S,)
      - behavior_stream_labels:  (S,) fixed-width ASCII labels

    The builder pads each trajectory to n_max rows. If a produced trajectory is longer
    than n_max, it raises ValueError instead of silently truncating.
    """
    assert n_max > 0

    output_path = Path(output_path)
    manifest_path = Path(manifest_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    goals_list: list[np.ndarray] = []
    trajs_list: list[np.ndarray] = []
    rotation_offsets_list: list[float] = []
    base_segment_indices_list: list[int] = []
    behavior_stream_indices_list: list[int] = []
    behavior_stream_labels_list: list[bytes] = []
    included_paths: list[str] = []
    num_base_segments = 0

    for record in iter_augmented_segment_records(
        seq_data_path=seq_data_path,
        lead_frames_before_dig=lead_frames_before_dig,
        close_frames_before_reset=close_frames_before_reset,
        bucket_flat_angle=bucket_flat_angle,
        mirror_original_indexing=mirror_original_indexing,
        rotation_limit=rotation_limit,
        rotation_step=rotation_step,
    ):
        source_npz_path = record["source_npz_path"]
        if not included_paths or included_paths[-1] != source_npz_path:
            included_paths.append(source_npz_path)

        goal = np.asarray(record["goal"], dtype=np.float32)
        traj = np.asarray(record["trajectory"], dtype=np.float32)

        if traj.shape[0] > n_max:
            raise ValueError(
                f"Trajectory length {traj.shape[0]} exceeds n_max={n_max}. "
                f"Offending file: {source_npz_path}"
            )

        goals_list.append(goal)
        trajs_list.append(traj)
        rotation_offsets_list.append(float(record["rotation_offset"]))
        base_segment_indices_list.append(int(record["global_base_segment_index"]))
        behavior_stream_indices_list.append(int(record["behavior_stream_index"]))
        behavior_stream_labels_list.append(str(record["behavior_stream"]).encode("utf-8"))
        num_base_segments = max(num_base_segments, int(record["global_base_segment_index"]) + 1)

    if not goals_list:
        raise RuntimeError("No segments were produced.")

    num_segments = len(goals_list)
    num_source_files = len(included_paths)

    goals = np.stack(goals_list, axis=0).astype(np.float32)  # (S, 2)
    trajectories = np.zeros((num_segments, n_max, 4), dtype=np.float32)
    lengths = np.zeros((num_segments,), dtype=np.int64)
    rotation_offsets = np.asarray(rotation_offsets_list, dtype=np.float32)
    base_segment_indices = np.asarray(base_segment_indices_list, dtype=np.int64)
    behavior_stream_indices = np.asarray(behavior_stream_indices_list, dtype=np.int64)
    behavior_stream_labels = np.asarray(behavior_stream_labels_list, dtype="S16")

    for i, traj in enumerate(trajs_list):
        n = traj.shape[0]
        trajectories[i, :n] = traj
        lengths[i] = n

    with h5py.File(output_path, "w") as f:
        f.create_dataset("goals", data=goals, compression="gzip")
        f.create_dataset("trajectories", data=trajectories, compression="gzip")
        f.create_dataset("lengths", data=lengths)
        f.create_dataset("rotation_offsets", data=rotation_offsets)
        f.create_dataset("base_segment_indices", data=base_segment_indices)
        f.create_dataset("behavior_stream_indices", data=behavior_stream_indices)
        f.create_dataset("behavior_stream_labels", data=behavior_stream_labels)

        f.attrs["num_source_files"] = num_source_files
        f.attrs["num_base_segments"] = num_base_segments
        f.attrs["num_segments"] = num_segments
        f.attrs["n_max"] = n_max
        f.attrs["rotation_limit"] = float(rotation_limit)
        f.attrs["rotation_step"] = float(rotation_step)
        f.attrs["mirror_original_indexing"] = bool(mirror_original_indexing)
        f.attrs["behavior_stream_choices"] = np.asarray(BEHAVIOR_STREAMS, dtype="S16")

    manifest_path.write_text("\n".join(included_paths))
    return output_path



def write_segment_csv_exports(
    seq_data_path: str | Path,
    output_dir: str | Path,
    *,
    output_prefix: str = "latest",
    lead_frames_before_dig: int = 20,
    close_frames_before_reset: int = 0,
    bucket_flat_angle: float = 1.1,
    mirror_original_indexing: bool = True,
    rotation_limit: float = ROTATION_LIMIT,
    rotation_step: float = ROTATION_STEP,
) -> dict[str, object]:
    """
    Write spreadsheet-friendly CSVs from the same augmented records used by the HDF5 builder.

    Outputs:
      - <output_prefix>_segments_summary.csv
          One row per augmented segment.
      - <output_prefix>_segments_flat_partXXXX.csv
          One row per frame, with segment metadata and goals repeated on each row.
      - <output_prefix>_manifest.txt
          One included joint_angles.npz path per line.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_csv_path = output_dir / f"{output_prefix}_segments_summary.csv"
    manifest_path = output_dir / f"{output_prefix}_manifest.txt"

    flat_headers = [
        "global_augmented_segment_index",
        "global_base_segment_index",
        "segment_index_in_file",
        "parent_cycle_index_in_file",
        "behavior_stream",
        "behavior_stream_index",
        "source_npz_path",
        "rotation_offset",
        "num_rotations_for_base_segment",
        "mirror_original_indexing",
        "dig_candidate_pos",
        "dump_candidate_pos",
        "reset_candidate_pos",
        "dig_event_frame",
        "dump_event_frame",
        "reset_event_frame",
        "goal_index_used_for_dig",
        "goal_index_used_for_dump",
        "goal_index_used_for_reset",
        "start_idx_used",
        "seq_end_idx_used",
        "segment_length",
        "frame_offset_in_segment",
        "absolute_frame_index_if_mirrored",
        "goal_source_index",
        "goal_col0",
        "goal_col2",
        "joint_control_0",
        "joint_control_1",
        "joint_control_2",
        "joint_control_3",
    ]

    summary_headers = [
        "global_augmented_segment_index",
        "global_base_segment_index",
        "segment_index_in_file",
        "parent_cycle_index_in_file",
        "behavior_stream",
        "behavior_stream_index",
        "source_npz_path",
        "rotation_offset",
        "num_rotations_for_base_segment",
        "mirror_original_indexing",
        "dig_candidate_pos",
        "dump_candidate_pos",
        "reset_candidate_pos",
        "dig_event_frame",
        "dump_event_frame",
        "reset_event_frame",
        "goal_index_used_for_dig",
        "goal_index_used_for_dump",
        "goal_index_used_for_reset",
        "start_idx_used",
        "seq_end_idx_used",
        "segment_length",
        "goal_source_index",
        "goal_col0",
        "goal_col2",
    ]

    flat_csv_paths: list[Path] = []
    included_paths: list[str] = []
    num_flat_rows = 0
    part_idx = 0
    rows_in_current_part = 0
    flat_f = None
    flat_writer = None

    def open_next_flat_part() -> tuple[object, csv.writer, Path]:
        nonlocal part_idx, rows_in_current_part
        part_idx += 1
        rows_in_current_part = 0
        flat_csv_path = output_dir / f"{output_prefix}_segments_flat_part{part_idx:04d}.csv"
        f = flat_csv_path.open("w", newline="")
        writer = csv.writer(f)
        writer.writerow(flat_headers)
        rows_in_current_part = 1
        return f, writer, flat_csv_path

    with summary_csv_path.open("w", newline="") as summary_f:
        summary_writer = csv.writer(summary_f)
        summary_writer.writerow(summary_headers)

        try:
            flat_f, flat_writer, first_flat_path = open_next_flat_part()
            flat_csv_paths.append(first_flat_path)

            for record in iter_augmented_segment_records(
                seq_data_path=seq_data_path,
                lead_frames_before_dig=lead_frames_before_dig,
                close_frames_before_reset=close_frames_before_reset,
                bucket_flat_angle=bucket_flat_angle,
                mirror_original_indexing=mirror_original_indexing,
                rotation_limit=rotation_limit,
                rotation_step=rotation_step,
            ):
                source_npz_path = record["source_npz_path"]
                if not included_paths or included_paths[-1] != source_npz_path:
                    included_paths.append(source_npz_path)

                goal = np.asarray(record["goal"], dtype=np.float32)
                traj = np.asarray(record["trajectory"], dtype=np.float32)

                summary_writer.writerow(
                    [
                        record["global_augmented_segment_index"],
                        record["global_base_segment_index"],
                        record["segment_index_in_file"],
                        record["parent_cycle_index_in_file"],
                        record["behavior_stream"],
                        record["behavior_stream_index"],
                        source_npz_path,
                        record["rotation_offset"],
                        record["num_rotations_for_base_segment"],
                        mirror_original_indexing,
                        record["dig_candidate_pos"],
                        record["dump_candidate_pos"],
                        record["reset_candidate_pos"],
                        record["dig_event_frame"],
                        record["dump_event_frame"],
                        record["reset_event_frame"],
                        record["goal_index_used_for_dig"],
                        record["goal_index_used_for_dump"],
                        record["goal_index_used_for_reset"],
                        record["start_idx_used"],
                        record["seq_end_idx_used"],
                        record["length"],
                        record["goal_source_index"],
                        float(goal[0]),
                        float(goal[1]),
                    ]
                )

                for frame_offset, frame in enumerate(traj):
                    if rows_in_current_part >= EXCEL_ROW_LIMIT - EXCEL_SAFETY_MARGIN:
                        assert flat_f is not None
                        flat_f.close()
                        flat_f, flat_writer, next_flat_path = open_next_flat_part()
                        flat_csv_paths.append(next_flat_path)

                    absolute_frame_index_if_mirrored = int(record["start_idx_used"]) + frame_offset
                    assert flat_writer is not None
                    flat_writer.writerow(
                        [
                            record["global_augmented_segment_index"],
                            record["global_base_segment_index"],
                            record["segment_index_in_file"],
                            record["parent_cycle_index_in_file"],
                            record["behavior_stream"],
                            record["behavior_stream_index"],
                            source_npz_path,
                            record["rotation_offset"],
                            record["num_rotations_for_base_segment"],
                            mirror_original_indexing,
                            record["dig_candidate_pos"],
                            record["dump_candidate_pos"],
                            record["reset_candidate_pos"],
                            record["dig_event_frame"],
                            record["dump_event_frame"],
                            record["reset_event_frame"],
                            record["goal_index_used_for_dig"],
                            record["goal_index_used_for_dump"],
                            record["goal_index_used_for_reset"],
                            record["start_idx_used"],
                            record["seq_end_idx_used"],
                            record["length"],
                            frame_offset,
                            absolute_frame_index_if_mirrored,
                            record["goal_source_index"],
                            float(goal[0]),
                            float(goal[1]),
                            float(frame[0]),
                            float(frame[1]),
                            float(frame[2]),
                            float(frame[3]),
                        ]
                    )
                    rows_in_current_part += 1
                    num_flat_rows += 1
        finally:
            if flat_f is not None:
                flat_f.close()

    manifest_path.write_text("\n".join(included_paths))

    return {
        "summary_csv": summary_csv_path,
        "flat_csv_parts": flat_csv_paths,
        "manifest": manifest_path,
        "num_flat_rows": num_flat_rows,
    }



def main() -> None:
    training_dir = Path("./training_data")

    output_file = build_segment_store(
        seq_data_path="./seq_data/",
        output_path=training_dir / "goal_reduce.h5",
        manifest_path=training_dir / "goal_reduce_manifest.txt",
        n_max=400,
        lead_frames_before_dig=20,
        close_frames_before_reset=0,
        bucket_flat_angle=1.1,
        mirror_original_indexing=True,
        rotation_limit=ROTATION_LIMIT,
        rotation_step=ROTATION_STEP,
    )

    # exports = write_segment_csv_exports(
    #     seq_data_path="./seq_data/",
    #     output_dir=training_dir,
    #     output_prefix="new_data_form",
    #     lead_frames_before_dig=20,
    #     close_frames_before_reset=0,
    #     bucket_flat_angle=1.1,
    #     mirror_original_indexing=True,
    #     rotation_limit=ROTATION_LIMIT,
    #     rotation_step=ROTATION_STEP,
    # )


if __name__ == "__main__":
    main()
