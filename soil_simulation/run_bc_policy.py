#!/usr/bin/env python3
"""Run the trained BC policy in the MPM soil simulator.

Usage:
    python soil_simulation/run_bc_policy.py --checkpoint test_20/best_model.pt

    # Specify primitive sequence (default: dig,swing,dump,return repeating)
    python soil_simulation/run_bc_policy.py --checkpoint test_20/best_model.pt --sequence dig,swing,dump,return

    # Run a single primitive indefinitely
    python soil_simulation/run_bc_policy.py --checkpoint test_20/best_model.pt --sequence dig --steps-per-primitive 100
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np

# Add project root to path
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
SOIL_SIM_DIR = str(Path(__file__).resolve().parent)
sys.path.insert(0, PROJECT_ROOT)

# Use the global newton (/home/caee/Desktop/newton) instead of the local
# soil_simulation/newton/ copy, which requires a newer warp.
# The global newton is compatible with warp 1.9.0.
GLOBAL_NEWTON = "/home/caee/Desktop/newton"
if os.path.isdir(GLOBAL_NEWTON):
    sys.path.insert(0, os.path.dirname(GLOBAL_NEWTON))

from bc_component.bc_inference import BCPolicy, NAME_TO_ID

# Mapping from BC joint names to simulator joint_map keys
BC_TO_SIM_JOINT = {
    0: "swing",   # full_arm_rotation -> swing
    1: "arm",     # lower_arm -> arm
    2: "stick",   # upperToLow -> stick
    3: "bucket",  # scoop1 -> bucket
}


class BCControlledExcavator:
    """Wraps ExcavatorMPMExample and drives it with a trained BC policy."""

    def __init__(self, sim, policy: BCPolicy, sequence: list[str], steps_per_primitive: int):
        self.sim = sim
        self.policy = policy
        self.sequence = sequence
        self.steps_per_primitive = steps_per_primitive
        self.current_primitive_idx = 0
        self.steps_in_current = 0
        self.total_steps = 0

        # Read initial joint state from sim and seed the policy
        initial_obs = self._read_joint_state()
        self.policy.reset(initial_obs)

        print(f"\nBC Policy Controller initialized")
        print(f"  Primitive sequence: {self.sequence}")
        print(f"  Steps per primitive: {self.steps_per_primitive}")
        print(f"  Joint mapping (BC -> sim):")
        for bc_idx, sim_key in BC_TO_SIM_JOINT.items():
            sim_idx = self.sim.joint_map.get(sim_key)
            print(f"    BC[{bc_idx}] ({self.policy.obs_keys[bc_idx]}) -> sim[{sim_key}] (idx={sim_idx})")
        print()

    def _read_joint_state(self) -> np.ndarray:
        """Read the 4 arm joint positions from the simulator state."""
        q = self.sim.state_0.joint_q.numpy()
        obs = np.zeros(self.policy.obs_dim, dtype=np.float32)
        for bc_idx, sim_key in BC_TO_SIM_JOINT.items():
            sim_idx = self.sim.joint_map.get(sim_key)
            if sim_idx is not None and sim_idx < len(q):
                obs[bc_idx] = float(q[sim_idx])
        return obs

    @property
    def current_primitive_name(self) -> str:
        return self.sequence[self.current_primitive_idx % len(self.sequence)]

    @property
    def current_primitive_id(self) -> int:
        return NAME_TO_ID[self.current_primitive_name]

    def apply_control(self) -> None:
        """Replace the simulator's apply_control with BC policy inference."""
        if self.sim.control_size == 0:
            return

        # Check if we should advance to next primitive
        if self.steps_in_current >= self.steps_per_primitive:
            self.current_primitive_idx += 1
            self.steps_in_current = 0
            if self.current_primitive_idx >= len(self.sequence):
                self.current_primitive_idx = 0
            print(f"\n>>> Switching to primitive: {self.current_primitive_name} "
                  f"(step {self.total_steps})")

        # Read current joint state from sim
        current_obs = self._read_joint_state()
        self.policy.push_obs(current_obs)

        # Predict next action
        pred = self.policy.predict(self.current_primitive_id)
        target_joints = pred[0]  # First horizon step

        # Write targets to simulator control
        targets = self.sim._joint_target_host.copy()
        for bc_idx, sim_key in BC_TO_SIM_JOINT.items():
            sim_idx = self.sim.joint_map.get(sim_key)
            if sim_idx is not None and sim_idx < len(targets):
                value = float(target_joints[bc_idx])
                targets[sim_idx] = self.sim._clip_target(sim_idx, value)

        self.sim.control.joint_target_pos.assign(targets)

        # Logging
        if self.total_steps % 30 == 0:  # Every ~1 second at 30fps
            obs_deg = np.rad2deg(current_obs)
            tgt_deg = np.rad2deg(target_joints)
            print(f"[step={self.total_steps:4d} t={self.sim.sim_time:.1f}s] "
                  f"primitive={self.current_primitive_name:6s} | "
                  f"obs(deg)=[{', '.join(f'{x:6.1f}' for x in obs_deg)}] | "
                  f"target(deg)=[{', '.join(f'{x:6.1f}' for x in tgt_deg)}]")

        self.steps_in_current += 1
        self.total_steps += 1


def main():
    parser = argparse.ArgumentParser(description="Run BC policy in excavator MPM simulator.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to best_model.pt")
    parser.add_argument("--sequence", type=str, default="dig,swing,dump,return",
                        help="Comma-separated primitive sequence (default: dig,swing,dump,return)")
    parser.add_argument("--steps-per-primitive", type=int, default=60,
                        help="Simulation steps per primitive before switching (default: 60 = ~2s at 30fps)")
    parser.add_argument("--preset", type=str, default="balanced", help="Simulation preset name")
    parser.add_argument("--device", type=str, default="cpu", help="Torch device for BC model")
    args = parser.parse_args()

    # Parse primitive sequence
    sequence = [s.strip() for s in args.sequence.split(",")]
    for name in sequence:
        if name not in NAME_TO_ID:
            parser.error(f"Unknown primitive '{name}'. Choose from: {list(NAME_TO_ID.keys())}")

    # Load BC policy
    print(f"Loading BC policy from: {args.checkpoint}")
    policy = BCPolicy(args.checkpoint, device=args.device)
    print(f"  Joints: {policy.obs_keys}")
    print(f"  Primitives: {policy.primitive_names}")

    # Launch simulator
    import newton
    import newton.examples
    from excavator_mpm_dp_soil_sticky import ExcavatorMPMExample

    viewer, sim_args = newton.examples.init()
    sim = ExcavatorMPMExample(viewer, preset_name=args.preset)

    # Create BC controller
    controller = BCControlledExcavator(
        sim=sim,
        policy=policy,
        sequence=sequence,
        steps_per_primitive=args.steps_per_primitive,
    )

    # Override the sim's apply_control with our BC version
    sim.apply_control = controller.apply_control

    print("\n" + "=" * 60)
    print("BC POLICY SIMULATION RUNNING")
    print(f"  Sequence: {' -> '.join(sequence)} (repeating)")
    print(f"  Steps per primitive: {args.steps_per_primitive} (~{args.steps_per_primitive/30:.1f}s)")
    print("=" * 60 + "\n")

    try:
        while viewer.is_running():
            sim.step()
    except KeyboardInterrupt:
        print(f"\nStopped at step {controller.total_steps} (t={sim.sim_time:.1f}s)")


if __name__ == "__main__":
    main()
