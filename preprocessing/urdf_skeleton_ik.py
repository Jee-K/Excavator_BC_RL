"""
URDF Skeleton IK Solver for Video Keypoint Fitting

This module provides tools to:
1. Extract skeleton keypoints from URDF
2. Forward kinematics: joint angles -> 3D keypoint positions
3. Inverse kinematics: 2D video keypoints -> joint angles (with occlusion handling)

Author: Claude
Date: 2026-02-12
"""

import numpy as np
import xml.etree.ElementTree as ET
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as R
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt


class ExcavatorURDFSkeleton:
    """Extract and work with excavator skeleton from URDF"""

    def __init__(self, urdf_path: str):
        """
        Args:
            urdf_path: Path to URDF file
        """
        self.urdf_path = urdf_path
        self.tree = ET.parse(urdf_path)
        self.root = self.tree.getroot()

        # Define main arm joints (in kinematic chain order)
        self.arm_joints = [
            "full_arm_rotation",  # Boom base
            "lower_arm",          # Boom/stick
            "upperToLow",         # Stick/arm
            "scoop1"              # Bucket
        ]

        # Parse URDF structure
        self.joint_info = self._parse_joints()
        self.link_info = self._parse_links()

        # Define skeleton keypoints
        self.keypoint_names = [
            "base",           # Base of excavator
            "boom_base",      # full_arm_rotation
            "boom_stick",     # lower_arm
            "stick_bucket",   # upperToLow
            "bucket_joint",   # scoop1
            "bucket_tip"      # End effector
        ]

        print(f"Loaded URDF with {len(self.arm_joints)} arm joints")
        print(f"Skeleton has {len(self.keypoint_names)} keypoints")

    def _parse_joints(self) -> Dict:
        """Parse all joints from URDF"""
        joints = {}
        for joint in self.root.findall('joint'):
            name = joint.get('name')
            joint_type = joint.get('type')

            # Parse origin
            origin_elem = joint.find('origin')
            xyz = [0, 0, 0]
            rpy = [0, 0, 0]
            if origin_elem is not None:
                if origin_elem.get('xyz'):
                    xyz = [float(x) for x in origin_elem.get('xyz').split()]
                if origin_elem.get('rpy'):
                    rpy = [float(x) for x in origin_elem.get('rpy').split()]

            # Parse limits
            limit_elem = joint.find('limit')
            lower, upper = -np.inf, np.inf
            if limit_elem is not None:
                if limit_elem.get('lower'):
                    lower = float(limit_elem.get('lower'))
                if limit_elem.get('upper'):
                    upper = float(limit_elem.get('upper'))

            # Parse axis
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

    def forward_kinematics(self, joint_angles: np.ndarray) -> np.ndarray:
        """
        Compute 3D positions of skeleton keypoints from joint angles.

        Args:
            joint_angles: Array of shape (4,) containing angles for the 4 arm joints
                         [full_arm_rotation, lower_arm, upperToLow, scoop1]

        Returns:
            keypoints: Array of shape (6, 3) containing 3D positions of keypoints
        """
        assert len(joint_angles) == 4, "Expected 4 joint angles"

        keypoints = np.zeros((6, 3))

        # Start at base (cabin center)
        current_pos = np.array([0.0, 0.0, 0.0])
        current_rot = np.eye(3)

        # Keypoint 0: Base
        keypoints[0] = current_pos

        # Build kinematic chain
        for i, joint_name in enumerate(self.arm_joints):
            joint = self.joint_info[joint_name]

            # Apply joint transform (from parent to child)
            # 1. Static transform from URDF origin
            rot_static = R.from_euler('xyz', joint['rpy']).as_matrix()
            pos_offset = joint['xyz']

            # Transform offset by current rotation
            current_pos = current_pos + current_rot @ pos_offset
            current_rot = current_rot @ rot_static

            # 2. Apply joint angle rotation
            joint_axis = joint['axis']
            angle = joint_angles[i]

            # Rotate around joint axis
            axis_rot = R.from_rotvec(angle * joint_axis).as_matrix()
            current_rot = current_rot @ axis_rot

            # Store keypoint at this joint
            keypoints[i + 1] = current_pos

        # Keypoint 5: Bucket tip (estimate as extension from bucket joint)
        # Typical bucket length ~0.4m
        bucket_extension = np.array([0.4, 0.0, 0.0])
        keypoints[5] = keypoints[4] + current_rot @ bucket_extension

        return keypoints

    def get_joint_limits(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get joint limits for the 4 main arm joints"""
        lower = np.zeros(4)
        upper = np.zeros(4)

        for i, joint_name in enumerate(self.arm_joints):
            joint = self.joint_info[joint_name]
            lower[i] = joint['lower']
            upper[i] = joint['upper']

        return lower, upper


class VideoToJointIK:
    """Inverse kinematics solver to fit URDF skeleton to video keypoints"""

    def __init__(self, skeleton: ExcavatorURDFSkeleton, camera_intrinsics: Optional[np.ndarray] = None):
        """
        Args:
            skeleton: ExcavatorURDFSkeleton instance
            camera_intrinsics: 3x3 camera intrinsic matrix (if None, uses weak perspective)
        """
        self.skeleton = skeleton
        self.camera_intrinsics = camera_intrinsics

        # Get joint limits
        self.joint_lower, self.joint_upper = skeleton.get_joint_limits()

        # Previous solution for temporal smoothing
        self.prev_angles = None
        self.prev_camera = None

    def project_3d_to_2d(self, points_3d: np.ndarray, camera_params: np.ndarray) -> np.ndarray:
        """
        Project 3D points to 2D using weak perspective or full perspective.

        Args:
            points_3d: (N, 3) array of 3D points
            camera_params: Camera parameters
                For weak perspective: [scale, tx, ty] (3,)
                For full perspective: [fx, fy, cx, cy, tx, ty, tz] (7,)

        Returns:
            points_2d: (N, 2) array of 2D pixel coordinates
        """
        if self.camera_intrinsics is None:
            # Weak perspective projection (scale + translation)
            scale, tx, ty = camera_params[:3]
            points_2d = points_3d[:, :2] * scale + np.array([tx, ty])
        else:
            # Full perspective projection
            fx, fy, cx, cy, tx, ty, tz = camera_params
            # Apply camera extrinsics (translation)
            points_cam = points_3d + np.array([tx, ty, tz])
            # Project using intrinsics
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
        Fit URDF skeleton to 2D video keypoints using IK.

        Args:
            keypoints_2d: (6, 2) array of 2D keypoint positions [x, y] in pixels
            visibility: (6,) array of visibility flags (1=visible, 0=occluded)
            init_angles: Initial guess for joint angles (if None, uses previous or zeros)
            temporal_weight: Weight for temporal smoothness term (penalizes change from previous frame)

        Returns:
            result: Dict containing:
                - joint_angles: (4,) optimized joint angles
                - camera_params: optimized camera parameters
                - reprojection_error: float
                - keypoints_3d: (6, 3) 3D keypoint positions
                - keypoints_2d_proj: (6, 2) reprojected 2D keypoints
        """
        # Initialize
        if init_angles is None:
            if self.prev_angles is not None:
                init_angles = self.prev_angles.copy()
            else:
                # Start with middle of joint ranges
                init_angles = (self.joint_lower + self.joint_upper) / 2

        # Initialize camera parameters
        if self.prev_camera is not None:
            init_camera = self.prev_camera.copy()
        else:
            if self.camera_intrinsics is None:
                # Weak perspective: [scale, tx, ty]
                # Estimate scale from keypoint spread
                visible_kpts = keypoints_2d[visibility > 0.5]
                if len(visible_kpts) > 0:
                    spread = np.std(visible_kpts)
                    init_camera = np.array([spread / 2.0,
                                           np.mean(visible_kpts[:, 0]),
                                           np.mean(visible_kpts[:, 1])])
                else:
                    init_camera = np.array([100.0, 320.0, 240.0])
            else:
                # Full perspective: [fx, fy, cx, cy, tx, ty, tz]
                init_camera = np.array([800, 800, 320, 240, 0, 0, 5.0])

        # Combine parameters
        if self.camera_intrinsics is None:
            x0 = np.concatenate([init_angles, init_camera[:3]])
            n_cam = 3
        else:
            x0 = np.concatenate([init_angles, init_camera])
            n_cam = 7

        # Define loss function
        def loss_fn(x):
            joint_angles = x[:4]
            camera_params = x[4:]

            # Forward kinematics
            keypoints_3d = self.skeleton.forward_kinematics(joint_angles)

            # Project to 2D
            keypoints_2d_proj = self.project_3d_to_2d(keypoints_3d, camera_params)

            # Reprojection error (only for visible keypoints)
            diff = keypoints_2d - keypoints_2d_proj
            weighted_diff = diff * visibility[:, np.newaxis]
            reprojection_loss = np.sum(weighted_diff ** 2)

            # Temporal smoothness (penalize large changes from previous frame)
            temporal_loss = 0.0
            if self.prev_angles is not None and temporal_weight > 0:
                temporal_loss = temporal_weight * np.sum((joint_angles - self.prev_angles) ** 2)

            return reprojection_loss + temporal_loss

        # Joint limits as bounds
        bounds = []
        for i in range(4):
            bounds.append((self.joint_lower[i], self.joint_upper[i]))

        # Camera bounds (loose)
        if self.camera_intrinsics is None:
            bounds.append((10.0, 500.0))  # scale
            bounds.append((0.0, 1000.0))  # tx
            bounds.append((0.0, 1000.0))  # ty
        else:
            bounds.append((500, 1500))   # fx
            bounds.append((500, 1500))   # fy
            bounds.append((0, 640))      # cx
            bounds.append((0, 480))      # cy
            bounds.append((-10, 10))     # tx
            bounds.append((-10, 10))     # ty
            bounds.append((1, 20))       # tz

        # Optimize
        result = minimize(
            loss_fn,
            x0,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 500, 'ftol': 1e-6}
        )

        # Extract results
        joint_angles = result.x[:4]
        camera_params = result.x[4:]

        # Update previous state
        self.prev_angles = joint_angles.copy()
        self.prev_camera = camera_params.copy()

        # Compute final 3D and 2D positions
        keypoints_3d = self.skeleton.forward_kinematics(joint_angles)
        keypoints_2d_proj = self.project_3d_to_2d(keypoints_3d, camera_params)

        # Compute reprojection error (only for visible points)
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
        """
        Fit skeleton to entire video sequence.

        Args:
            video_keypoints: (T, 6, 2) array of keypoints for T frames
            visibility: (T, 6) visibility flags
            temporal_weight: Weight for temporal smoothness
            verbose: Print progress

        Returns:
            results: Dict containing:
                - joint_angles_sequence: (T, 4) joint angles over time
                - camera_params_sequence: (T, n_cam) camera params over time
                - errors: (T,) reprojection errors
        """
        T = len(video_keypoints)
        joint_angles_seq = np.zeros((T, 4))
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
        """Reset temporal state (use when starting new video)"""
        self.prev_angles = None
        self.prev_camera = None


def visualize_fit(
    image: np.ndarray,
    keypoints_2d: np.ndarray,
    keypoints_2d_proj: np.ndarray,
    visibility: np.ndarray,
    joint_angles: np.ndarray
):
    """
    Visualize the skeleton fit on an image.

    Args:
        image: (H, W, 3) image
        keypoints_2d: (6, 2) detected keypoints
        keypoints_2d_proj: (6, 2) projected skeleton keypoints
        visibility: (6,) visibility flags
        joint_angles: (4,) joint angles in radians
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: detected keypoints
    axes[0].imshow(image)
    axes[0].set_title("Detected Keypoints (with occlusions)")

    for i, (kpt, vis) in enumerate(zip(keypoints_2d, visibility)):
        color = 'green' if vis > 0.5 else 'red'
        axes[0].plot(kpt[0], kpt[1], 'o', color=color, markersize=10)
        axes[0].text(kpt[0] + 5, kpt[1], str(i), color='white')

    # Draw skeleton connections (detected)
    skeleton_pairs = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]
    for i, j in skeleton_pairs:
        if visibility[i] > 0.5 and visibility[j] > 0.5:
            axes[0].plot([keypoints_2d[i, 0], keypoints_2d[j, 0]],
                        [keypoints_2d[i, 1], keypoints_2d[j, 1]], 'g-', linewidth=2)

    # Right: fitted skeleton
    axes[1].imshow(image)
    axes[1].set_title("Fitted URDF Skeleton")

    for i, kpt in enumerate(keypoints_2d_proj):
        axes[1].plot(kpt[0], kpt[1], 'bo', markersize=10)
        axes[1].text(kpt[0] + 5, kpt[1], str(i), color='white')

    # Draw skeleton connections (projected)
    for i, j in skeleton_pairs:
        axes[1].plot([keypoints_2d_proj[i, 0], keypoints_2d_proj[j, 0]],
                    [keypoints_2d_proj[i, 1], keypoints_2d_proj[j, 1]], 'b-', linewidth=2)

    # Print joint angles
    joint_names = ["full_arm_rot", "lower_arm", "upperToLow", "scoop1"]
    angles_deg = np.rad2deg(joint_angles)
    print("\nJoint Angles:")
    for name, angle in zip(joint_names, angles_deg):
        print(f"  {name}: {angle:.1f}°")

    plt.tight_layout()
    plt.show()


# Example usage
if __name__ == "__main__":
    # Load URDF
    urdf_path = "../excavatorURDF/robot_fixed_alternate.urdf"
    skeleton = ExcavatorURDFSkeleton(urdf_path)

    # Test forward kinematics
    print("\n=== Testing Forward Kinematics ===")
    test_angles = np.array([0.2, 0.3, -0.1, 0.5])  # radians
    keypoints_3d = skeleton.forward_kinematics(test_angles)
    print("Joint angles (rad):", test_angles)
    print("3D Keypoints:")
    for i, (name, kpt) in enumerate(zip(skeleton.keypoint_names, keypoints_3d)):
        print(f"  {i}. {name}: ({kpt[0]:.3f}, {kpt[1]:.3f}, {kpt[2]:.3f})")

    # Test IK with synthetic data
    print("\n=== Testing Inverse Kinematics ===")
    ik_solver = VideoToJointIK(skeleton)

    # Generate synthetic 2D keypoints by projecting
    camera_params = np.array([150.0, 320.0, 240.0])  # scale, tx, ty
    keypoints_2d_true = ik_solver.project_3d_to_2d(keypoints_3d, camera_params)

    # Add noise
    keypoints_2d_noisy = keypoints_2d_true + np.random.randn(6, 2) * 5.0

    # Simulate occlusions (keypoint 3 and 4 occluded)
    visibility = np.ones(6)
    visibility[3] = 0
    visibility[4] = 0

    print("Visibility:", visibility)
    print("True joint angles (rad):", test_angles)

    # Solve IK
    result = ik_solver.fit_frame(keypoints_2d_noisy, visibility, temporal_weight=0.0)

    print("Estimated joint angles (rad):", result['joint_angles'])
    print("Angle error (deg):", np.rad2deg(np.abs(test_angles - result['joint_angles'])))
    print("Reprojection error (px):", result['reprojection_error'])
    print("Success:", result['success'])
