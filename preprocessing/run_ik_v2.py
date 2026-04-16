"""
Version 2

This is the py execution file equivalent of the ik_from_annotations_testing.ipynb notebook.

Usage:
    python run_ik_v2.py <seq_dir>
    python run_ik_v2.py ../../RL_project/caterpillar352.../sequences/seq_0000

Input (inside seq_dir):
    pred_tracks_anchored.npy      (1, T, 10, 2)
    pred_visibility_anchored.npy  (1, T, 10) or (1, T, 10, 1)
    frame_list.txt

Output (inside seq_dir):
    joint_angles_v2.npz
    cabin_yaw_smoothed.png
    scoop1_smoothed.png
    ik_overlay.mp4
"""

import argparse
import sys
import os
from pathlib import Path

import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')  # non-interactive backend for saving figures
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d, median_filter
from scipy.spatial.transform import Rotation as R

# Allow importing from this directory regardless of cwd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from urdf_skeleton_custom_v2 import build_excavator_skeleton, SideViewIK

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
JOINT_SMOOTH_WIN   = 20   # frames — uniform filter for joint angles
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
    """Returns (cabin_yaw_deg, cabin_yaw_rad) both shape (T,), smoothed.

    v2 improvements over v1:
    - Uses RANSAC homography to reject outlier corner matches
    - Skips frames where fewer than 50% of points agree (inlier_ratio < 0.5)
    - Clamps per-frame delta > 30° (tracking glitch, not real rotation)
    """
    n_frames  = cabin_corners.shape[0]
    ref_pts   = cabin_corners[0].astype(np.float32)
    yaw_deg   = np.zeros(n_frames)

    for t in range(1, n_frames):
        curr_pts = cabin_corners[t].astype(np.float32)
        H, mask = cv2.findHomography(ref_pts, curr_pts, cv2.RANSAC, ransacReprojThreshold=3.0)
        if H is None:
            yaw_deg[t] = yaw_deg[t - 1]
            continue
        inlier_ratio = mask.sum() / len(mask)
        if inlier_ratio < 0.5:
            yaw_deg[t] = yaw_deg[t - 1]
            continue
        U, _, Vt = np.linalg.svd(H[:2, :2])
        R_2d = U @ Vt
        yaw_candidate = np.degrees(np.arctan2(R_2d[1, 0], R_2d[0, 0]))
        # Clamp: skip frames where delta > 30° (tracking glitch, not real rotation)
        if abs(yaw_candidate - yaw_deg[t - 1]) > 30:
            yaw_deg[t] = yaw_deg[t - 1]
        else:
            yaw_deg[t] = yaw_candidate

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


def smooth_scoop1_occlusions(joint_angles: np.ndarray, vis: np.ndarray,
                              ik: SideViewIK,
                              scoop_joint_idx: int = 2,
                              bucket_tip_idx: int = 0,
                              bucket_floor_idx: int = 2) -> np.ndarray:
    """Post-process scoop1 during windows where both bucket points are occluded.

    Replaces scoop1 with a smooth arc: angle_before → peak at joint limit → angle_after.
    Runs AFTER temporal smoothing so the arc is not washed out.
    """
    angles     = joint_angles.copy()
    T          = len(angles)
    scoop_limit = ik.joint_upper[scoop_joint_idx]
    occluded   = (vis[:, bucket_tip_idx] <= 0.5) & (vis[:, bucket_floor_idx] <= 0.5)

    windows = []
    in_window = False
    for t in range(T):
        if occluded[t] and not in_window:
            win_start = t
            in_window = True
        elif not occluded[t] and in_window:
            windows.append((win_start, t - 1))
            in_window = False
    if in_window:
        windows.append((win_start, T - 1))

    print(f'Occlusion windows (both bucket points lost): {len(windows)}')
    for ws, we in windows:
        print(f'  frames {ws}–{we} ({we - ws + 1} frames)')
        angle_before = angles[ws - 1, scoop_joint_idx] if ws > 0 else angles[we + 1, scoop_joint_idx]
        angle_after  = angles[we + 1, scoop_joint_idx] if we < T - 1 else angles[ws - 1, scoop_joint_idx]
        peak   = scoop_limit
        n      = we - ws + 1
        t_norm = np.linspace(0, np.pi, n)
        s      = np.linspace(0, 1, n)
        base   = angle_before * (1 - s) + angle_after * s
        arc    = base + (peak - base) * np.sin(t_norm)
        angles[ws:we + 1, scoop_joint_idx] = arc

        # Blend transitions over a padding region on each side
        pad         = 15
        blend_start = max(0, ws - pad)
        blend_end   = min(T, we + 1 + pad)
        angles[blend_start:blend_end, scoop_joint_idx] = uniform_filter1d(
            angles[blend_start:blend_end, scoop_joint_idx], size=pad, mode='nearest'
        )

    return angles


# ── skeleton edges for visualization ─────────────────────────────────────────

SKELETON_EDGES = [
    ('frame_front_mid', 'turret_center'),
    ('turret_center',   'frame_rear_mid'),
    ('turret_center',   'arm_base'),
    ('arm_base',        'boom_tip'),
    ('boom_tip',        'stick_tip'),
    ('stick_tip',       'bucket_floor'),
    ('bucket_floor',    'bucket_tip'),
    ('bucket_tip',      'stick_tip'),
]

# BGR colors per keypoint (matches KEYPOINT_ORDER)
KP_COLORS_BGR = [
    (0,   0,   255),  # 0 bucket_tip     red
    (0,  165,  255),  # 1 stick_tip      orange
    (0,  255,  255),  # 2 bucket_floor   yellow
    (0,  255,   0),   # 3 boom_tip       lime
    (255, 255,   0),  # 4 arm_base       cyan
    (255,   0,   0),  # 5 turret_center  blue
    (255,   0,  255), # 6 frame_front    magenta
    (255, 255,  255), # 7 frame_rear     white
]


# ── step 4: diagnostics ───────────────────────────────────────────────────────

def save_diagnostics(seq_dir: Path, skeleton,
                     cabin_yaw_deg: np.ndarray,
                     joint_angles_smooth: np.ndarray,
                     joint_angles_raw: np.ndarray,
                     frame_files: list,
                     keypoints_2d: np.ndarray,
                     projected_2d: np.ndarray,
                     visibility: np.ndarray,
                     cabin_yaw_rad: np.ndarray) -> None:
    """Save cabin yaw plot, scoop angle plot, and overlay video."""
    n_frames = len(joint_angles_smooth)
    frames   = np.arange(n_frames)

    # ── 1. Cabin yaw plot ────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.plot(frames, cabin_yaw_deg, color='steelblue', linewidth=1.2)
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.8)
    ax.set_xlabel('Frame')
    ax.set_ylabel('Yaw (deg)')
    ax.set_title('Cabin yaw — smoothed')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = seq_dir / 'cabin_yaw_smoothed.png'
    plt.savefig(out, dpi=150)
    plt.close(fig)
    print(f'Saved: {out}')

    # ── 2. Scoop angle plot ──────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.plot(frames, np.degrees(joint_angles_raw[:, 2]),
            color='gray', linewidth=0.8, alpha=0.6, label='raw')
    ax.plot(frames, np.degrees(joint_angles_smooth[:, 2]),
            color='tomato', linewidth=1.4, label='smoothed')
    # shade occlusion windows
    occluded = (visibility[:, 0] <= 0.5) & (visibility[:, 2] <= 0.5)
    in_win = False
    for t in range(n_frames):
        if occluded[t] and not in_win:
            ws = t; in_win = True
        elif not occluded[t] and in_win:
            ax.axvspan(ws, t - 1, alpha=0.15, color='red')
            in_win = False
    if in_win:
        ax.axvspan(ws, n_frames - 1, alpha=0.15, color='red')
    ax.set_xlabel('Frame')
    ax.set_ylabel('scoop1 (deg)')
    ax.set_title('scoop1 joint angle — smoothed  (red shading = both bucket points occluded)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = seq_dir / 'scoop1_smoothed.png'
    plt.savefig(out, dpi=150)
    plt.close(fig)
    print(f'Saved: {out}')

    # ── 3. Overlay video ─────────────────────────────────────────────────────
    img0 = cv2.imread(frame_files[0])
    if img0 is None:
        print(f'Warning: could not read {frame_files[0]}, skipping video.')
        return
    img_h, img_w = img0.shape[:2]

    video_path = str(seq_dir / 'ik_overlay.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps    = 15
    writer = cv2.VideoWriter(video_path, fourcc, fps, (img_w, img_h))

    for t in range(n_frames):
        img = cv2.imread(frame_files[t])
        if img is None:
            img = np.zeros((img_h, img_w, 3), dtype=np.uint8)

        proj = projected_2d[t]   # (8, 2)
        obs  = keypoints_2d[t]   # (8, 2)
        vis  = visibility[t]     # (8,)

        # Draw skeleton edges (cyan)
        for p1_name, p2_name in SKELETON_EDGES:
            i1 = KEYPOINT_ORDER.index(p1_name)
            i2 = KEYPOINT_ORDER.index(p2_name)
            pt1 = (int(round(proj[i1, 0])), int(round(proj[i1, 1])))
            pt2 = (int(round(proj[i2, 0])), int(round(proj[i2, 1])))
            cv2.line(img, pt1, pt2, (255, 255, 0), 2, cv2.LINE_AA)

        # Draw projected keypoints (filled circles, per-kp color)
        for i, color in enumerate(KP_COLORS_BGR):
            px = int(round(proj[i, 0]))
            py = int(round(proj[i, 1]))
            cv2.circle(img, (px, py), 6, color, -1, cv2.LINE_AA)
            cv2.circle(img, (px, py), 6, (0, 0, 0), 1, cv2.LINE_AA)

        # Draw observed keypoints (X marker, green if visible)
        for i in range(len(KEYPOINT_ORDER)):
            if vis[i] > 0.5:
                ox = int(round(obs[i, 0]))
                oy = int(round(obs[i, 1]))
                cv2.drawMarker(img, (ox, oy), (0, 255, 0),
                               cv2.MARKER_CROSS, 12, 2, cv2.LINE_AA)

        # HUD: frame number, yaw, scoop angle
        angles_deg = np.degrees(joint_angles_smooth[t])
        label = (f'F{t:03d}  yaw={cabin_yaw_deg[t]:.1f}deg  '
                 f'arm={angles_deg[0]:.1f}  upper={angles_deg[1]:.1f}  scoop={angles_deg[2]:.1f}')
        cv2.putText(img, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 255, 255), 1, cv2.LINE_AA)

        writer.write(img)

    writer.release()
    print(f'Saved: {video_path}  ({n_frames} frames @ {fps} fps)')


# ── step 5: save npz ─────────────────────────────────────────────────────────

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

    output_path = seq_dir / 'joint_angles_v2.npz'
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

    # 5. post-process scoop1 during bucket occlusion windows
    joint_angles_smooth = smooth_scoop1_occlusions(joint_angles_smooth, visibility, ik)

    # 6. recompute projected_2d from final smoothed angles
    projected_2d_smooth = np.zeros_like(projected_2d)
    for t in range(n_frames):
        kp_3d  = skeleton.forward_kinematics(joint_angles_smooth[t])
        proj   = ik._rotate_and_project(kp_3d, cabin_yaw_rad[t])
        scale_t, tx_t, ty_t = ik._projection_params(kp_3d, cabin_yaw_rad[t], keypoints_2d[t])
        projected_2d_smooth[t] = ik._apply_proj(proj, scale_t, tx_t, ty_t)
    projected_2d = projected_2d_smooth

    # 7. save diagnostics (plots + video)
    save_diagnostics(
        seq_dir, skeleton,
        cabin_yaw_deg, joint_angles_smooth, joint_angles_raw,
        frame_files, keypoints_2d, projected_2d, visibility, cabin_yaw_rad,
    )

    # 8. save npz
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
