"""
Version 04/08

This is the py execution file equivalent of the ik_from_annotations_w_rotation.ipynb notebook. 
I used the notebook for development and visualization, but this script is for execution alone.

Usage:
    python run_ik.py <seq_dir>
    python run_ik.py ../../RL_project/caterpillar352.../sequences/seq_0000

Input (inside seq_dir):
    pred_tracks_anchored.npy      (1, T, 10, 2)
    pred_visibility_anchored.npy  (1, T, 10) or (1, T, 10, 1)
    frame_list.txt

Output (inside seq_dir):
    joint_angles.npz
"""

import argparse
import sys
import os
from pathlib import Path

import numpy as np
import cv2
from scipy.ndimage import uniform_filter1d, median_filter
from scipy.spatial.transform import Rotation as R

# Allow importing from this directory regardless of cwd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from urdf_skeleton_custom import build_excavator_skeleton, SideViewIK

# ── constants ─────────────────────────────────────────────────────────────────

KEYPOINT_ORDER = [
    'bucket_tip',       # 0
    'stick_tip',        # 1
    'bucket_floor',     # 2
    'boom_tip',         # 3
    'arm_base',         # 4
    'turret_center',    # 5
    'frame_front_mid',  # 6
    'frame_rear_mid',   # 7
]

YAW_SMOOTH_WINDOW  = 20   # frames — median + uniform filter for cabin yaw
JOINT_SMOOTH_WIN   = 10   # frames — uniform filter for joint angles
TEMPORAL_WEIGHT    = 0.01 # IK temporal regularization


# ── step 1: load npy data ─────────────────────────────────────────────────────

def load_sequence(seq_dir: Path):
    tracks_raw = np.load(seq_dir / 'pred_tracks_anchored.npy')[0]      # (T, 10, 2)
    vis_raw    = np.load(seq_dir / 'pred_visibility_anchored.npy')[0]  # (T, 10) or (T, 10, 1)
    if vis_raw.ndim == 3:
        vis_raw = vis_raw[:, :, 0]

    with open(seq_dir / 'frame_list.txt') as f:
        frame_files = [l.strip() for l in f if l.strip()]

    n_frames = tracks_raw.shape[0]
    assert n_frames == len(frame_files), (
        f'Track frames ({n_frames}) != frame_list entries ({len(frame_files)})'
    )

    keypoints_2d  = tracks_raw[:, :8, :]           # (T, 8, 2)
    visibility    = vis_raw[:, :8]                  # (T, 8)
    cabin_corners = tracks_raw[:, [6, 7, 8, 9], :] # (T, 4, 2) — for yaw estimation

    return keypoints_2d, visibility, cabin_corners, frame_files


# ── step 2: cabin yaw via homography ─────────────────────────────────────────

def estimate_cabin_yaw(cabin_corners: np.ndarray) -> tuple:
    """Returns (cabin_yaw_deg, cabin_yaw_rad) both shape (T,), smoothed."""
    n_frames  = cabin_corners.shape[0]
    corners_0 = cabin_corners[0].astype(np.float32)
    yaw_deg   = np.zeros(n_frames)

    for t in range(1, n_frames):
        corners_t = cabin_corners[t].astype(np.float32)
        H, _ = cv2.findHomography(corners_0, corners_t)
        if H is None:
            yaw_deg[t] = yaw_deg[t - 1]
            continue
        U, _, Vt = np.linalg.svd(H[:2, :2])
        R_2d = U @ Vt
        yaw_deg[t] = np.degrees(np.arctan2(R_2d[1, 0], R_2d[0, 0]))

    # smooth: median to kill spikes, then uniform to smooth
    yaw_deg = median_filter(yaw_deg, size=YAW_SMOOTH_WINDOW)
    yaw_deg = uniform_filter1d(yaw_deg, size=YAW_SMOOTH_WINDOW, mode='nearest')

    return yaw_deg, np.radians(yaw_deg)


# ── step 3: smooth joint angles ───────────────────────────────────────────────

def smooth_angles(joint_angles_raw: np.ndarray, window: int = JOINT_SMOOTH_WIN) -> np.ndarray:
    smoothed = np.zeros_like(joint_angles_raw)
    for j in range(joint_angles_raw.shape[1]):
        smoothed[:, j] = uniform_filter1d(
            joint_angles_raw[:, j], size=window, mode='nearest'
        )
    return smoothed


# ── step 4: save ─────────────────────────────────────────────────────────────

def save_results(seq_dir: Path, skeleton, ik: SideViewIK,
                 joint_angles_raw, joint_angles_smooth,
                 cabin_yaw_rad, cabin_yaw_deg,
                 keypoints_2d, projected_2d, errors, visibility):

    n_frames = len(joint_angles_smooth)

    # cylindrical keypoint positions (r, z, theta) — origin = turret_center
    keypoints_local = np.stack(
        [skeleton.forward_kinematics(joint_angles_smooth[t]) for t in range(n_frames)]
    )  # (T, 8, 3)
    kp_r     = np.sqrt(keypoints_local[:, :, 0]**2 + keypoints_local[:, :, 1]**2)  # (T, 8)
    kp_z     = keypoints_local[:, :, 2]                                              # (T, 8)
    kp_theta = np.tile(cabin_yaw_rad[:, np.newaxis], (1, len(KEYPOINT_ORDER)))      # (T, 8)
    keypoints_cylindrical = np.stack([kp_r, kp_z, kp_theta], axis=-1)              # (T, 8, 3)

    output_path = seq_dir / 'joint_angles.npz'
    np.savez(
        output_path,
        joint_angles          = joint_angles_smooth,
        joint_angles_raw      = joint_angles_raw,
        joint_names           = np.array(skeleton.arm_joints),
        cabin_yaw_rad         = cabin_yaw_rad,
        cabin_yaw_deg         = cabin_yaw_deg,
        keypoint_names        = np.array(KEYPOINT_ORDER),
        keypoints_cylindrical = keypoints_cylindrical,
        keypoints_2d          = keypoints_2d,
        projected_2d          = projected_2d,
        reprojection_errors   = errors,
        visibility            = visibility,
        scale_px_per_m        = np.array(ik.fixed_scale if ik.fixed_scale else 0.0),
        seq_dir               = np.array(str(seq_dir)),
    )
    print(f'Saved: {output_path}')
    print(f'  joint_angles          : {joint_angles_smooth.shape}  (radians, smoothed)')
    print(f'  cabin_yaw_rad         : {cabin_yaw_rad.shape}')
    print(f'  keypoints_cylindrical : {keypoints_cylindrical.shape}  [r, z, theta]')
    print(f'  mean reprojection err : {np.mean(errors):.2f} px²')


# ── main ──────────────────────────────────────────────────────────────────────

def run(seq_dir: Path, verbose: bool = True):
    print(f'=== run_ik: {seq_dir} ===')

    # 1. load
    keypoints_2d, visibility, cabin_corners, frame_files = load_sequence(seq_dir)
    n_frames = len(keypoints_2d)
    print(f'Loaded {n_frames} frames')

    # 2. cabin yaw
    cabin_yaw_deg, cabin_yaw_rad = estimate_cabin_yaw(cabin_corners)
    print(f'Cabin yaw: [{cabin_yaw_deg.min():.1f}, {cabin_yaw_deg.max():.1f}] deg')

    # 3. skeleton + IK
    skeleton = build_excavator_skeleton()
    ik = SideViewIK(skeleton)

    joint_angles_raw, errors, projected_2d = ik.fit_sequence(
        keypoints_2d, visibility,
        cabin_yaw_seq=cabin_yaw_rad,
        temporal_weight=TEMPORAL_WEIGHT,
        verbose=verbose,
    )

    # 4. smooth joint angles
    joint_angles_smooth = smooth_angles(joint_angles_raw, window=JOINT_SMOOTH_WIN)

    # 5. save
    save_results(
        seq_dir, skeleton, ik,
        joint_angles_raw, joint_angles_smooth,
        cabin_yaw_rad, cabin_yaw_deg,
        keypoints_2d, projected_2d, errors, visibility,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run side-view IK on a sequence directory.')
    parser.add_argument('seq_dir', type=Path, help='Path to sequence directory')
    parser.add_argument('--quiet', action='store_true', help='Suppress per-frame output')
    args = parser.parse_args()

    if not args.seq_dir.exists():
        print(f'Error: {args.seq_dir} does not exist')
        sys.exit(1)

    run(args.seq_dir, verbose=not args.quiet)
