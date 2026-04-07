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

        # print(f"Loaded URDF: {urdf_path}")
        # print(f"Available joints: {list(self.joint_info.keys())}")

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

        # print(f"\n✓ Skeleton defined:")
        # print(f"  Arm joints ({len(self.arm_joints)}): {self.arm_joints}")
        # print(f"  Keypoints ({len(self.keypoints)}):")
        # for i, kp in enumerate(self.keypoints):
        #     print(f"    {i}. {kp.name} -> link '{kp.link}' + offset {kp.offset}")

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

        # Start from root (assume first joint's parent is root)
        root_link = self.joint_info[self.arm_joints[0]]['parent']
        link_transforms[root_link] = (np.array([0.0, 0.0, 0.0]), np.eye(3))

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
                # Link not in our kinematic chain, use identity
                # (This happens for base link keypoints)
                link_pos, link_rot = np.array([0.0, 0.0, 0.0]), np.eye(3)
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
            # print(keypoints_2d.shape, visibility.shape, keypoints_2d, visibility)
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


# Example: Define your excavator skeleton
if __name__ == "__main__":
    # urdf_path = "./excavatorURDF/excavator_lowpoly_locked_splitbucket.urdf"
    # skeleton = CustomURDFSkeleton(urdf_path)

    # # Define arm joints
    # arm_joints = ["lower_arm", "upperToLow", "scoop1"]

    # # Define keypoints (full body): (name, link, offset)
    # keypoints = [
    #     ("frame_front_mid", "compact_excavator_frame_body", np.array([0.800000, 0.000000, -0.000000])),
    #     ("frame_rear_mid", "compact_excavator_frame_body", np.array([-0.800000, -0.000000, 0.000000])),
    #     ("turret_center", "compact_excavator_turret_cabin_roller", np.array([-0.250000, -0.000000, -0.050000])),
    #     ("arm_base", "part01_pin_1", np.array([0.040708, 0.093264, -0.319600])),
    #     ("boom_tip", "upper_boom", np.array([-0.663317, 2.202642, -0.039292])),
    #     ("stick_tip", "lower_boom", np.array([-0.048634, -1.046702, -0.095708])),
    #     ("bucket_floor", "bucketry", np.array([0.010710, 0.293730, 0.095708])),
    #     ("bucket_tip", "bucketry", np.array([-0.379012, 0.804848, 0.095708])),
    # ]



    urdf_path = "./excavatorURDF/excavator_lowpoly_locked_splitbucket.urdf"

    skeleton = CustomURDFSkeleton(urdf_path)

    # Define your skeleton!
    # 4 arm joints (in kinematic order)
    arm_joints = [
        "lower_arm",          # Joint 1
        "upperToLow",         # Joint 2
        "scoop1"              # Joint 3
    ]

    # Define keypoints as (name, link, offset)
    # These should match what you can detect in your videos!
    keypoints = [
        # Keypoint 0: Cabin center (base reference)
        ("cabin_center", "compact_excavator_cabin_body_cmpl", np.array([0.0, 0.0, 0.5])),

        # Keypoint 1: Boom base joint (where boom attaches to cabin)
        ("boom_base", "part01_pin_1", np.array([0.0, 0.0, 0.0])),

        # Keypoint 2: Boom-stick joint (elbow) - midpoint of boom link
        ("boom_stick_joint", "upper_boom", np.array([0.0, 1.2, 0.0])),

        # Keypoint 3: Stick-bucket joint (wrist)
        ("stick_bucket_joint", "lower_boom", np.array([0.0, -0.6, 0.0])),

        # Keypoint 4: Bucket attachment point
        ("bucket_joint", "bucketry", np.array([0.1, 0.2, 0.0])),

        # Keypoint 5: Bucket tip (end effector)
        ("bucket_tip", "bucketry", np.array([0.2, 0.6, 0.0])),
    ]

    skeleton.define_skeleton(arm_joints, keypoints)

    # Test forward kinematics
    print("\n=== Testing Forward Kinematics ===")
    test_angles = np.array([0.3, -0.1, 0.5])
    keypoints_3d = skeleton.forward_kinematics(test_angles)

    print(f"Joint angles: {test_angles}")
    print(f"Keypoint 3D positions:")
    for i, (kp, pos) in enumerate(zip(skeleton.keypoints, keypoints_3d)):
        print(f"  {i}. {kp.name}: {pos}")

    # Test IK
    print("\n=== Testing IK ===")
    ik = CustomKeypointIK(skeleton)

    # Simulate 2D projection
    camera_params = np.array([150.0, 320.0, 240.0])
    keypoints_2d = ik.project_3d_to_2d(keypoints_3d, camera_params)

    import plotly.graph_objects as go
    fig = go.Figure()
    KEYPOINT_ORDER = [
        'frame_front_mid',
        'frame_rear_mid',
        'turret_center',
        'arm_base',
        'boom_tip',
        'stick_tip',
        'bucket_floor',
        'bucket_tip',
    ]
    kp = keypoints_3d

    fig.add_trace(go.Scatter3d(
        x=kp[:, 0], y=kp[:, 1], z=kp[:, 2], 
        mode='markers+text',
        marker=dict(size=8, color='red', line=dict(width=1, color='black')),
        text=KEYPOINT_ORDER,
        textposition='top center',
        textfont=dict(size=8),
        name='Keypoints',
        hovertemplate='<b>%{text}</b><br>X:%{x:.3f}<br>Y:%{y:.3f}<br>Z:%{z:.3f}<extra></extra>',
    ))
    fig.show()

    # Add noise and occlusions
    keypoints_2d_noisy = keypoints_2d + np.random.randn(keypoints_2d.shape[0], 2) * 3.0
    visibility = np.ones(keypoints_2d.shape[0])
    visibility[2] = 0  # Occlude wrist
    visibility[3] = 0  # Occlude bucket

    result = ik.fit_frame(keypoints_2d_noisy, visibility)

    print(f"True angles:      {test_angles}")
    print(f"Recovered angles: {result['joint_angles']}")
    print(f"Error (deg):      {np.rad2deg(np.abs(test_angles - result['joint_angles']))}")
    print(f"Reprojection error: {result['reprojection_error']:.2f} px")
