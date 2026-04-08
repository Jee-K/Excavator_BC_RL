"""
Custom Keypoint Definition for URDF Skeleton IK

Allows you to define your own keypoints as specific points on the URDF skeleton.
Each keypoint is defined by:
- Which link it's attached to
- Offset from that link's origin (in link frame)

This lets you match exactly what's visible in your videos!

Author: Claude
Date: 2026-02-12
"""

import numpy as np
import xml.etree.ElementTree as ET
from pathlib import Path
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as R
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class Keypoint:
    """Definition of a keypoint on the skeleton"""
    name: str           # Human-readable name (e.g., "boom_base_joint")
    link: str           # Link name from URDF
    offset: np.ndarray  # 3D offset from link origin (in link frame)


class CustomURDFSkeleton:
    """URDF skeleton with user-defined keypoints"""

    def __init__(self, urdf_path: str):
        self.urdf_path = urdf_path
        self.tree = ET.parse(urdf_path)
        self.root = self.tree.getroot()

        # Parse URDF
        self.joint_info = self._parse_joints()
        self.link_info = self._parse_links()

        # User will define these
        self.keypoints: List[Keypoint] = []
        self.arm_joints = []  # Ordered list of joint names for control

        # --- ADDED: rest-pose world transforms for all links (used by world_to_local) ---
        self.all_link_transforms = self._compute_all_link_transforms()

        print(f"Loaded URDF: {urdf_path}")
        print(f"Available joints: {list(self.joint_info.keys())}")

    def _parse_joints(self) -> Dict:
        """Parse all joints from URDF"""
        joints = {}
        for joint in self.root.findall('joint'):
            name = joint.get('name')
            joint_type = joint.get('type')

            origin_elem = joint.find('origin')
            xyz = [0, 0, 0]
            rpy = [0, 0, 0]
            if origin_elem is not None:
                if origin_elem.get('xyz'):
                    xyz = [float(x) for x in origin_elem.get('xyz').split()]
                if origin_elem.get('rpy'):
                    rpy = [float(x) for x in origin_elem.get('rpy').split()]

            limit_elem = joint.find('limit')
            lower, upper = -np.inf, np.inf
            if limit_elem is not None:
                if limit_elem.get('lower'):
                    lower = float(limit_elem.get('lower'))
                if limit_elem.get('upper'):
                    upper = float(limit_elem.get('upper'))

            axis_elem = joint.find('axis')
            axis = [0, 0, 1]
            if axis_elem is not None:
                axis = [float(x) for x in axis_elem.get('xyz').split()]

            parent = joint.find('parent').get('link')
            child = joint.find('child').get('link')

            joints[name] = {
                'type': joint_type,
                'parent': parent,
                'child': child,
                'xyz': np.array(xyz),
                'rpy': np.array(rpy),
                'axis': np.array(axis),
                'lower': lower,
                'upper': upper
            }

        return joints

    def _parse_links(self) -> Dict:
        """Parse link information"""
        links = {}
        for link in self.root.findall('link'):
            name = link.get('name')
            links[name] = {'name': name}
        return links

    # --- ADDED: two helpers that mirror what interactive_skeleton_builder.ipynb does ---

    def _compute_all_link_transforms(self) -> Dict:
        """BFS over the full URDF at zero joint angles to get each link's world transform."""
        parent_links = {j['parent'] for j in self.joint_info.values()}
        child_links  = {j['child']  for j in self.joint_info.values()}
        root_link = (parent_links - child_links).pop()

        link_transforms = {root_link: np.eye(4)}

        parent_to_children: Dict[str, list] = {}
        for jinfo in self.joint_info.values():
            parent_to_children.setdefault(jinfo['parent'], []).append(jinfo)

        queue = [root_link]
        while queue:
            current = queue.pop(0)
            for jinfo in parent_to_children.get(current, []):
                T = np.eye(4)
                T[:3, :3] = R.from_euler('xyz', jinfo['rpy']).as_matrix()
                T[:3, 3] = jinfo['xyz']
                link_transforms[jinfo['child']] = link_transforms[current] @ T
                queue.append(jinfo['child'])

        return link_transforms

    def world_to_local(self, world_xyz, link_name: str) -> np.ndarray:
        """Convert a world-frame point (rest pose) to a local offset for *link_name*.
        Use the coordinates you read off the interactive notebook hover, then pass
        the result directly as the offset in define_skeleton()."""
        T = self.all_link_transforms[link_name]
        pt = np.array([*world_xyz, 1.0])
        return (np.linalg.inv(T) @ pt)[:3]

    # --- END ADDED ---

    def define_skeleton(
        self,
        arm_joints: List[str],
        keypoints: List[Tuple[str, str, np.ndarray]]
    ):
        """
        Define your custom skeleton.

        Args:
            arm_joints: Ordered list of joint names for the arm chain
                       Example: ["full_arm_rotation", "lower_arm", "upperToLow", "scoop1"]
            keypoints: List of (name, link, offset) tuples
                       Example: [
                           ("cabin_center", "compact_excavator_cabin_body_cmpl", [0, 0, 0.5]),
                           ("boom_base", "part01_pin_1", [0, 0, 0]),
                           ("boom_end", "part02_cmpl", [0, 2.0, 0]),
                           ...
                       ]
        """
        self.arm_joints = arm_joints

        # Verify joints exist
        for joint_name in arm_joints:
            if joint_name not in self.joint_info:
                raise ValueError(f"Joint '{joint_name}' not found in URDF. "
                               f"Available: {list(self.joint_info.keys())}")

        # Create keypoint objects
        self.keypoints = []
        for name, link, offset in keypoints:
            if link not in self.link_info:
                raise ValueError(f"Link '{link}' not found in URDF")
            self.keypoints.append(Keypoint(name, link, np.array(offset)))

        print(f"\n✓ Skeleton defined:")
        print(f"  Arm joints ({len(self.arm_joints)}): {self.arm_joints}")
        print(f"  Keypoints ({len(self.keypoints)}):")
        for i, kp in enumerate(self.keypoints):
            print(f"    {i}. {kp.name} -> link '{kp.link}' + offset {kp.offset}")

    def forward_kinematics(self, joint_angles: np.ndarray) -> np.ndarray:
        """
        Compute 3D positions of all keypoints from joint angles.

        Args:
            joint_angles: Array of shape (n_joints,) with arm joint angles

        Returns:
            keypoints_3d: Array of shape (n_keypoints, 3)
        """
        if len(joint_angles) != len(self.arm_joints):
            raise ValueError(f"Expected {len(self.arm_joints)} joint angles, got {len(joint_angles)}")

        # Build forward kinematics chain
        # We need to traverse the URDF tree and compute transforms for each link

        # First, build the kinematic tree
        link_transforms = {}  # link_name -> (position, rotation_matrix)

        # Start from the arm chain root using its URDF world position so that
        # arm keypoints are in the same coordinate frame as body keypoints.
        root_link = self.joint_info[self.arm_joints[0]]['parent']
        T_root = self.all_link_transforms[root_link]
        link_transforms[root_link] = (T_root[:3, 3].copy(), T_root[:3, :3].copy())

        # Traverse the arm joints in order
        for i, joint_name in enumerate(self.arm_joints):
            joint = self.joint_info[joint_name]
            parent_link = joint['parent']
            child_link = joint['child']

            # Get parent transform
            if parent_link not in link_transforms:
                # If parent not computed yet, skip (shouldn't happen with ordered joints)
                continue

            parent_pos, parent_rot = link_transforms[parent_link]

            # Apply joint static transform
            rot_static = R.from_euler('xyz', joint['rpy']).as_matrix()
            pos_offset = joint['xyz']

            # Transform offset by parent rotation
            current_pos = parent_pos + parent_rot @ pos_offset
            current_rot = parent_rot @ rot_static

            # Apply joint angle
            angle = joint_angles[i]
            axis = joint['axis']
            axis_rot = R.from_rotvec(angle * axis).as_matrix()
            current_rot = current_rot @ axis_rot

            # Store child link transform
            link_transforms[child_link] = (current_pos, current_rot)

        # Now compute keypoint positions
        keypoints_3d = np.zeros((len(self.keypoints), 3))

        for i, kp in enumerate(self.keypoints):
            if kp.link not in link_transforms:
                # Link not in the arm chain — use the full URDF rest-pose transform
                # (e.g. frame_body, turret_cabin_roller for body keypoints)
                T = self.all_link_transforms.get(kp.link)
                if T is not None:
                    link_pos = T[:3, 3]
                    link_rot = T[:3, :3]
                else:
                    link_pos, link_rot = np.zeros(3), np.eye(3)
            else:
                link_pos, link_rot = link_transforms[kp.link]

            # Apply keypoint offset in link frame
            keypoints_3d[i] = link_pos + link_rot @ kp.offset

        return keypoints_3d

    def get_joint_limits(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get joint limits for arm joints"""
        lower = np.zeros(len(self.arm_joints))
        upper = np.zeros(len(self.arm_joints))

        for i, joint_name in enumerate(self.arm_joints):
            joint = self.joint_info[joint_name]
            lower[i] = joint['lower']
            upper[i] = joint['upper']

        return lower, upper


class CustomKeypointIK:
    """IK solver for custom keypoint skeleton"""

    def __init__(self, skeleton: CustomURDFSkeleton, camera_intrinsics: Optional[np.ndarray] = None):
        self.skeleton = skeleton
        self.camera_intrinsics = camera_intrinsics
        self.joint_lower, self.joint_upper = skeleton.get_joint_limits()

        self.prev_angles = None
        self.prev_camera = None

    def project_3d_to_2d(self, points_3d: np.ndarray, camera_params: np.ndarray) -> np.ndarray:
        """Project 3D points to 2D (weak perspective)"""
        if self.camera_intrinsics is None:
            scale, tx, ty = camera_params[:3]
            points_2d = points_3d[:, :2] * scale + np.array([tx, ty])
        else:
            fx, fy, cx, cy, tx, ty, tz = camera_params
            points_cam = points_3d + np.array([tx, ty, tz])
            points_2d = np.zeros((len(points_3d), 2))
            points_2d[:, 0] = fx * points_cam[:, 0] / points_cam[:, 2] + cx
            points_2d[:, 1] = fy * points_cam[:, 1] / points_cam[:, 2] + cy
        return points_2d

    def fit_frame(
        self,
        keypoints_2d: np.ndarray,
        visibility: np.ndarray,
        init_angles: Optional[np.ndarray] = None,
        temporal_weight: float = 0.1
    ) -> Dict:
        """
        Fit skeleton to 2D keypoints.

        Args:
            keypoints_2d: (n_keypoints, 2) detected keypoint positions
            visibility: (n_keypoints,) visibility mask
            init_angles: Initial joint angles (if None, uses middle of range)
            temporal_weight: Temporal smoothness weight

        Returns:
            Dict with joint_angles, camera_params, reprojection_error, etc.
        """
        n_joints = len(self.skeleton.arm_joints)
        n_keypoints = len(self.skeleton.keypoints)

        if keypoints_2d.shape != (n_keypoints, 2):
            raise ValueError(f"Expected keypoints shape ({n_keypoints}, 2), got {keypoints_2d.shape}")

        # Initialize
        if init_angles is None:
            if self.prev_angles is not None:
                init_angles = self.prev_angles.copy()
            else:
                init_angles = (self.joint_lower + self.joint_upper) / 2

        # Initialize camera
        if self.prev_camera is not None:
            init_camera = self.prev_camera.copy()
        else:
            visible_kpts = keypoints_2d[visibility > 0.5]
            if len(visible_kpts) > 0:
                spread = np.std(visible_kpts)
                init_camera = np.array([spread / 2.0,
                                       np.mean(visible_kpts[:, 0]),
                                       np.mean(visible_kpts[:, 1])])
            else:
                init_camera = np.array([100.0, 320.0, 240.0])

        x0 = np.concatenate([init_angles, init_camera])

        # Loss function
        def loss_fn(x):
            joint_angles = x[:n_joints]
            camera_params = x[n_joints:]

            keypoints_3d = self.skeleton.forward_kinematics(joint_angles)
            keypoints_2d_proj = self.project_3d_to_2d(keypoints_3d, camera_params)

            diff = keypoints_2d - keypoints_2d_proj
            weighted_diff = diff * visibility[:, np.newaxis]
            reprojection_loss = np.sum(weighted_diff ** 2)

            temporal_loss = 0.0
            if self.prev_angles is not None and temporal_weight > 0:
                temporal_loss = temporal_weight * np.sum((joint_angles - self.prev_angles) ** 2)

            return reprojection_loss + temporal_loss

        # Bounds
        bounds = []
        for i in range(n_joints):
            bounds.append((self.joint_lower[i], self.joint_upper[i]))
        bounds.append((10.0, 500.0))   # scale
        bounds.append((0.0, 1000.0))   # tx
        bounds.append((0.0, 1000.0))   # ty

        # Optimize
        result = minimize(
            loss_fn,
            x0,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 500, 'ftol': 1e-6}
        )

        joint_angles = result.x[:n_joints]
        camera_params = result.x[n_joints:]

        self.prev_angles = joint_angles.copy()
        self.prev_camera = camera_params.copy()

        keypoints_3d = self.skeleton.forward_kinematics(joint_angles)
        keypoints_2d_proj = self.project_3d_to_2d(keypoints_3d, camera_params)

        visible_error = np.sum(((keypoints_2d - keypoints_2d_proj) ** 2)[visibility > 0.5])

        return {
            'joint_angles': joint_angles,
            'camera_params': camera_params,
            'reprojection_error': visible_error,
            'keypoints_3d': keypoints_3d,
            'keypoints_2d_proj': keypoints_2d_proj,
            'success': result.success
        }

    def fit_video(
        self,
        video_keypoints: np.ndarray,
        visibility: np.ndarray,
        temporal_weight: float = 0.1,
        verbose: bool = True
    ) -> Dict:
        """Fit skeleton to video sequence"""
        T = len(video_keypoints)
        n_joints = len(self.skeleton.arm_joints)

        joint_angles_seq = np.zeros((T, n_joints))
        camera_params_seq = []
        errors = np.zeros(T)

        for t in range(T):
            if verbose and t % 10 == 0:
                print(f"Processing frame {t}/{T}...")

            result = self.fit_frame(
                video_keypoints[t],
                visibility[t],
                temporal_weight=temporal_weight
            )

            joint_angles_seq[t] = result['joint_angles']
            camera_params_seq.append(result['camera_params'])
            errors[t] = result['reprojection_error']

        if verbose:
            print(f"\nDone! Mean reprojection error: {np.mean(errors):.2f} px")

        return {
            'joint_angles_sequence': joint_angles_seq,
            'camera_params_sequence': np.array(camera_params_seq),
            'errors': errors
        }

    def reset(self):
        """Reset temporal state"""
        self.prev_angles = None
        self.prev_camera = None


# --- ADDED: pre-built factory so other scripts can do:
#       from urdf_skeleton_custom import build_excavator_skeleton
#       skeleton = build_excavator_skeleton(urdf_path)
#   World coordinates below come from interactive_skeleton_builder.ipynb (cell 9).
#   To update: hover the mesh in the notebook, copy the world [x,y,z], replace here.

DEFAULT_URDF_PATH = str(
    Path(__file__).resolve().parent.parent / "excavatorURDF" / "excavator_lowpoly_locked_splitbucket.urdf"
)


class SideViewIK:
    """Two-stage FK + IK for orthographic side-view excavator fitting.

    Stage 1 — analytic:
      Rotate 3D skeleton by known cabin_yaw, orthographic-project (Y horiz, -Z vert),
      estimate TRUE physical scale once from the most side-on frame, anchor
      translation to turret_center.

    Stage 2 — IK:
      Minimize reprojection error of 5 arm keypoints over 3 arm joint angles.

    Keypoint index convention (KEYPOINT_ORDER):
      0:bucket_tip  1:stick_tip  2:bucket_floor  3:boom_tip  4:arm_base
      5:turret_center  6:frame_front_mid  7:frame_rear_mid
    """

    TURRET_IDX     = 5
    FRAME_FRONT    = 6
    FRAME_REAR     = 7
    ARM_KP_INDICES = [0, 1, 2, 3, 4]

    def __init__(self, skeleton: 'CustomURDFSkeleton', fixed_scale=None, facing=None):
        self.skeleton = skeleton
        self.joint_lower, self.joint_upper = skeleton.get_joint_limits()
        self.n_joints = len(skeleton.arm_joints)
        self.n_kp     = len(skeleton.keypoints)
        self.prev_angles = None
        self.fixed_scale = fixed_scale  # px/m; None = auto-estimate per sequence
        self.facing = facing            # +1=arm left, -1=arm right, None=auto-detect

    # ── scale estimation ──────────────────────────────────────────────────────

    def estimate_scale(self, kp_2d_ref: np.ndarray, cabin_yaw_ref: float = 0.0) -> float:
        zero_angles = np.zeros(self.n_joints)
        kp_3d_ref   = self.skeleton.forward_kinematics(zero_angles)
        proj_ref    = self._rotate_and_project(kp_3d_ref, cabin_yaw_ref)
        obs_dist    = np.linalg.norm(kp_2d_ref[self.FRAME_FRONT] - kp_2d_ref[self.FRAME_REAR])
        proj_dist   = np.linalg.norm(proj_ref[self.FRAME_FRONT]  - proj_ref[self.FRAME_REAR])
        self.fixed_scale = obs_dist / proj_dist if proj_dist > 1e-6 else None
        return self.fixed_scale

    # ── projection ────────────────────────────────────────────────────────────

    @staticmethod
    def detect_facing(kp_2d: np.ndarray) -> int:
        """+1 if arm/front on LEFT side of image, -1 if on RIGHT."""
        return +1 if kp_2d[6, 0] < kp_2d[7, 0] else -1

    def _rotate_and_project(self, kp_3d: np.ndarray, cabin_yaw_rad: float) -> np.ndarray:
        """R_z(cabin_yaw) then orthographic project → (N,2) unnormalized."""
        R_z = R.from_euler('z', cabin_yaw_rad).as_matrix()
        kp_world = (R_z @ kp_3d.T).T
        facing = self.facing if self.facing is not None else +1
        # compensate cos(θ) sign flip when |cabin_yaw| > 90°
        eff = facing * np.sign(np.cos(cabin_yaw_rad))
        if eff == 0:
            eff = facing
        return np.stack([eff * kp_world[:, 1], -kp_world[:, 2]], axis=1)

    def _projection_params(self, kp_3d: np.ndarray, cabin_yaw_rad: float,
                           kp_2d: np.ndarray):
        proj = self._rotate_and_project(kp_3d, cabin_yaw_rad)
        if self.fixed_scale is not None:
            scale = self.fixed_scale
        else:
            obs_dist  = np.linalg.norm(kp_2d[self.FRAME_FRONT] - kp_2d[self.FRAME_REAR])
            proj_dist = np.linalg.norm(proj[self.FRAME_FRONT]  - proj[self.FRAME_REAR])
            scale = obs_dist / proj_dist if proj_dist > 1e-6 else 100.0
        tx = kp_2d[self.TURRET_IDX, 0] - scale * proj[self.TURRET_IDX, 0]
        ty = kp_2d[self.TURRET_IDX, 1] - scale * proj[self.TURRET_IDX, 1]
        return scale, tx, ty

    def _apply_proj(self, proj_unnorm: np.ndarray, scale: float,
                    tx: float, ty: float) -> np.ndarray:
        return scale * proj_unnorm + np.array([tx, ty])

    # ── per-frame fit ─────────────────────────────────────────────────────────

    def fit_frame(self, kp_2d: np.ndarray, vis: np.ndarray,
                  cabin_yaw_rad: float, temporal_weight: float = 0.1) -> dict:
        init_angles = self.prev_angles if self.prev_angles is not None else np.zeros(self.n_joints)
        kp_3d_any = self.skeleton.forward_kinematics(init_angles)
        scale, tx, ty = self._projection_params(kp_3d_any, cabin_yaw_rad, kp_2d)

        arm_vis = vis[self.ARM_KP_INDICES]
        arm_2d  = kp_2d[self.ARM_KP_INDICES]

        def loss(angles):
            kp_3d   = self.skeleton.forward_kinematics(angles)
            proj    = self._rotate_and_project(kp_3d, cabin_yaw_rad)
            kp_proj = self._apply_proj(proj, scale, tx, ty)
            reproj  = np.sum((arm_2d - kp_proj[self.ARM_KP_INDICES]) ** 2
                             * arm_vis[:, np.newaxis])
            temporal = (temporal_weight * np.sum((angles - self.prev_angles) ** 2)
                        if self.prev_angles is not None and temporal_weight > 0 else 0.0)
            return reproj + temporal

        result = minimize(
            loss, init_angles, method='L-BFGS-B',
            bounds=[(self.joint_lower[i], self.joint_upper[i]) for i in range(self.n_joints)],
            options={'maxiter': 500},
        )
        final_angles = result.x

        kp_3d   = self.skeleton.forward_kinematics(final_angles)
        proj    = self._rotate_and_project(kp_3d, cabin_yaw_rad)
        kp_proj = self._apply_proj(proj, scale, tx, ty)
        self.prev_angles = final_angles.copy()

        vis_mask = vis > 0.5
        arm_mask = arm_vis > 0.5
        body_pts = [self.TURRET_IDX, self.FRAME_FRONT, self.FRAME_REAR]
        return {
            'joint_angles': final_angles,
            'scale':        scale,
            'translation':  np.array([tx, ty]),
            'projected_2d': kp_proj,
            'keypoints_3d': kp_3d,
            'error':     (np.mean(np.sum((kp_2d - kp_proj) ** 2, axis=1)[vis_mask])
                          if vis_mask.any() else 0.0),
            'arm_error': (np.mean(np.sum((arm_2d - kp_proj[self.ARM_KP_INDICES]) ** 2,
                                         axis=1)[arm_mask]) if arm_mask.any() else 0.0),
            'body_error': np.mean(np.sum((kp_2d[body_pts] - kp_proj[body_pts]) ** 2, axis=1)),
            'success':    result.success,
        }

    # ── sequence fit ──────────────────────────────────────────────────────────

    def fit_sequence(self, all_kp_2d: np.ndarray, all_vis: np.ndarray,
                     cabin_yaw_seq: np.ndarray, temporal_weight: float = 0.01,
                     fix_scale: bool = True, verbose: bool = True) -> tuple:
        """Fit skeleton to a full sequence. Returns (angles, errors, projected_2d)."""
        T = len(all_kp_2d)
        ref_t = int(np.argmin(np.abs(cabin_yaw_seq)))

        if self.facing is None:
            self.facing = self.detect_facing(all_kp_2d[ref_t])
            if verbose:
                side = 'LEFT' if self.facing == +1 else 'RIGHT'
                print(f'Facing: {side}  '
                      f'(frame_front.x={all_kp_2d[ref_t][6,0]:.0f}  '
                      f'frame_rear.x={all_kp_2d[ref_t][7,0]:.0f})')

        if fix_scale and self.fixed_scale is None:
            self.estimate_scale(all_kp_2d[ref_t], cabin_yaw_seq[ref_t])
            if verbose:
                print(f'Scale: {self.fixed_scale:.2f} px/m  '
                      f'(from frame {ref_t}, θ={np.degrees(cabin_yaw_seq[ref_t]):.1f}°)')

        angles_seq    = np.zeros((T, self.n_joints))
        errors        = np.zeros(T)
        projected_seq = np.zeros_like(all_kp_2d)
        self.prev_angles = None

        for t in range(T):
            res = self.fit_frame(all_kp_2d[t], all_vis[t], cabin_yaw_seq[t], temporal_weight)
            angles_seq[t]    = res['joint_angles']
            errors[t]        = res['error']
            projected_seq[t] = res['projected_2d']
            if verbose and (t % 20 == 0 or t == T - 1):
                print(f'  [{t:3d}/{T}] err={res["error"]:.1f}px  '
                      f'arm={res["arm_error"]:.1f}  body={res["body_error"]:.1f}  '
                      f'scale={res["scale"]:.1f}  yaw={np.degrees(cabin_yaw_seq[t]):.1f}°')

        if verbose:
            print(f'Done. Mean error: {np.mean(errors):.2f} px')
        return angles_seq, errors, projected_seq

    def reset(self):
        self.prev_angles = None
        self.fixed_scale = None
        self.facing = None


def build_excavator_skeleton(urdf_path: str = DEFAULT_URDF_PATH) -> CustomURDFSkeleton:
    """Return a ready-to-use CustomURDFSkeleton with the excavator keypoints baked in."""
    skeleton = CustomURDFSkeleton(urdf_path)

    arm_joints = ["lower_arm", "upperToLow", "scoop1"]

    # Local offsets computed by interactive_skeleton_builder.ipynb (world_to_local).
    # Order matches KEYPOINT_ORDER in ik_from_annotations_testing.ipynb:
    #   0:bucket_tip  1:stick_tip  2:bucket_floor  3:boom_tip
    #   4:arm_base    5:turret_center  6:frame_front_mid  7:frame_rear_mid
    # To update: re-run the notebook, copy the printed local arrays here.
    keypoints = [
        ("bucket_tip",       "bucketry",                              np.array([ 0.098565,  0.961117,  0.095708])),
        ("stick_tip",        "lower_boom",                            np.array([ 0.015942, -1.661787, -0.095708])),
        ("bucket_floor",     "bucketry",                              np.array([ 0.371847,  0.589880,  0.095708])),
        ("boom_tip",         "upper_boom",                            np.array([-0.817103,  2.330508, -0.039292])),
        ("arm_base",         "part01_pin_1",                          np.array([ 0.040708,  0.093264, -0.319600])),
        ("turret_center",    "compact_excavator_turret_cabin_roller",  np.array([-0.250000, -0.000000, -0.050000])),
        ("frame_front_mid",  "compact_excavator_frame_body",          np.array([ 0.800000,  0.000000, -0.000000])),
        ("frame_rear_mid",   "compact_excavator_frame_body",          np.array([-0.800000, -0.000000,  0.000000])),
    ]

    skeleton.define_skeleton(arm_joints, keypoints)
    return skeleton

# --- END ADDED ---


if __name__ == "__main__":
    skeleton = build_excavator_skeleton()

    # Test forward kinematics
    print("\n=== Testing Forward Kinematics ===")
    n_joints = len(skeleton.arm_joints)
    n_keypoints = len(skeleton.keypoints)
    test_angles = np.array([0.3, -0.1, 0.5])[:n_joints]
    keypoints_3d = skeleton.forward_kinematics(test_angles)

    print(f"Joint angles: {test_angles}")
    print(f"Keypoint 3D positions:")
    for i, (kp, pos) in enumerate(zip(skeleton.keypoints, keypoints_3d)):
        print(f"  {i}. {kp.name}: {pos}")

    # Test IK
    print("\n=== Testing IK ===")
    ik = CustomKeypointIK(skeleton)

    camera_params = np.array([150.0, 320.0, 240.0])
    keypoints_2d = ik.project_3d_to_2d(keypoints_3d, camera_params)

    keypoints_2d_noisy = keypoints_2d + np.random.randn(n_keypoints, 2) * 3.0
    visibility = np.ones(n_keypoints)
    visibility[5] = 0  # Occlude stick_tip
    visibility[6] = 0  # Occlude bucket_floor

    result = ik.fit_frame(keypoints_2d_noisy, visibility)

    print(f"True angles:      {test_angles}")
    print(f"Recovered angles: {result['joint_angles']}")
    print(f"Error (deg):      {np.rad2deg(np.abs(test_angles - result['joint_angles']))}")
    print(f"Reprojection error: {result['reprojection_error']:.2f} px")
