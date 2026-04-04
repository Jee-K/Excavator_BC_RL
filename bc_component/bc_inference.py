"""Inference script for trained BC excavator policy.

Usage:
    # Roll out all 4 primitives from their mean starting state
    python bc_component/bc_inference.py --checkpoint test_20/best_model.pt

    # Roll out a specific primitive for N steps
    python bc_component/bc_inference.py --checkpoint test_20/best_model.pt --primitive dig --steps 30

    # Start from specific joint angles (radians)
    python bc_component/bc_inference.py --checkpoint test_20/best_model.pt --primitive dig --start 0.8,0.1,-0.5,0.0
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import torch
except ImportError:
    raise RuntimeError("PyTorch is required for inference.")

# Add parent directory so we can import the trainer module
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bc_component.excavator_bc_trainer import build_model


PRIMITIVE_NAMES = {0: "dig", 1: "dump", 2: "swing", 3: "return"}
NAME_TO_ID = {v: k for k, v in PRIMITIVE_NAMES.items()}


class BCPolicy:
    """Wraps a trained BC checkpoint for easy inference."""

    def __init__(self, checkpoint_path: str, device: str = "cpu"):
        self.device = torch.device(device)
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        self.obs_dim = ckpt["obs_dim"]
        self.act_dim = ckpt["act_dim"]
        self.num_primitives = ckpt["num_primitives"]
        self.primitive_names = ckpt["primitive_names"]
        self.obs_keys = ckpt["obs_keys"]
        self.act_keys = ckpt["act_keys"]

        self.obs_mean = ckpt["obs_norm_mean"]
        self.obs_std = ckpt["obs_norm_std"]
        self.act_mean = ckpt["act_norm_mean"]
        self.act_std = ckpt["act_norm_std"]

        args_dict = ckpt["args"]
        self.context = args_dict["context"]
        self.horizon = args_dict["horizon"]

        # Rebuild model
        args = argparse.Namespace(**args_dict)
        self.model = build_model(args, self.obs_dim, self.act_dim, self.num_primitives)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.to(self.device)
        self.model.eval()

        # Observation history buffer
        self.obs_history = np.zeros((self.context, self.obs_dim), dtype=np.float32)
        self.valid_count = 0

    def reset(self, initial_obs: Optional[np.ndarray] = None):
        """Reset observation history. Optionally seed with an initial observation."""
        self.obs_history = np.zeros((self.context, self.obs_dim), dtype=np.float32)
        self.valid_count = 0
        if initial_obs is not None:
            self.push_obs(initial_obs)

    def push_obs(self, obs_raw: np.ndarray):
        """Add a raw (unnormalized) observation to the history buffer."""
        obs_norm = ((obs_raw - self.obs_mean) / self.obs_std).astype(np.float32)
        self.obs_history = np.roll(self.obs_history, -1, axis=0)
        self.obs_history[-1] = obs_norm
        self.valid_count = min(self.valid_count + 1, self.context)

    @torch.no_grad()
    def predict(self, primitive_id: int) -> np.ndarray:
        """Predict next action(s) given current history and primitive.

        Returns:
            np.ndarray of shape (horizon, act_dim) in raw joint angle space (radians).
        """
        obs_seq = torch.from_numpy(self.obs_history).unsqueeze(0).to(self.device)
        phase_seq = torch.zeros(1, self.context, 1, device=self.device)

        valid_seq = torch.zeros(1, self.context, device=self.device)
        if self.valid_count > 0:
            valid_seq[0, -self.valid_count:] = 1.0

        prim = torch.tensor([primitive_id], dtype=torch.long, device=self.device)

        pred_norm = self.model(obs_seq, phase_seq, valid_seq, prim)
        pred_np = pred_norm[0].cpu().numpy()

        # Denormalize
        return pred_np * self.act_std + self.act_mean

    def rollout(self, primitive_id: int, start_obs: np.ndarray, steps: int) -> np.ndarray:
        """Auto-regressively roll out a trajectory.

        Args:
            primitive_id: Which primitive (0=dig, 1=dump, 2=swing, 3=return).
            start_obs: Initial joint angles in radians, shape (act_dim,).
            steps: Number of steps to roll out.

        Returns:
            np.ndarray of shape (steps+1, act_dim) — the full trajectory including start.
        """
        self.reset(start_obs)
        trajectory = [start_obs.copy()]

        for _ in range(steps):
            pred = self.predict(primitive_id)
            next_obs = pred[0]  # Use first horizon step
            trajectory.append(next_obs.copy())
            self.push_obs(next_obs)

        return np.array(trajectory)


def main():
    parser = argparse.ArgumentParser(description="BC policy inference and rollout visualization.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to best_model.pt")
    parser.add_argument("--primitive", type=str, default=None,
                        help="Primitive name (dig/dump/swing/return). If not set, rolls out all 4.")
    parser.add_argument("--steps", type=int, default=20, help="Number of rollout steps")
    parser.add_argument("--start", type=str, default=None,
                        help="Comma-separated starting joint angles in radians (e.g. 0.8,0.1,-0.5,0.0)")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    policy = BCPolicy(args.checkpoint, device=args.device)

    print(f"Loaded model: {args.checkpoint}")
    print(f"  Architecture: {policy.model.__class__.__name__}")
    print(f"  Obs dim: {policy.obs_dim}, Act dim: {policy.act_dim}")
    print(f"  Context: {policy.context}, Horizon: {policy.horizon}")
    print(f"  Primitives: {policy.primitive_names}")
    print(f"  Joint names: {policy.obs_keys}")
    print()

    # Determine which primitives to run
    if args.primitive:
        if args.primitive not in NAME_TO_ID:
            parser.error(f"Unknown primitive '{args.primitive}'. Choose from: {list(NAME_TO_ID.keys())}")
        prim_ids = [NAME_TO_ID[args.primitive]]
    else:
        prim_ids = sorted(policy.primitive_names.keys())

    # Determine starting state
    if args.start:
        start = np.array([float(x) for x in args.start.split(",")], dtype=np.float32)
        if len(start) != policy.obs_dim:
            parser.error(f"Expected {policy.obs_dim} values, got {len(start)}")
    else:
        start = None  # Will use per-primitive mean

    # Run rollouts
    all_trajectories = {}
    for pid in prim_ids:
        pname = policy.primitive_names[pid]
        if start is not None:
            s = start
        else:
            # Use the normalization mean as a reasonable starting point
            s = policy.obs_mean.copy()

        traj = policy.rollout(pid, s, args.steps)
        all_trajectories[pname] = traj

        print(f"--- {pname} (primitive_id={pid}) ---")
        print(f"  Start (deg): [{', '.join(f'{np.rad2deg(x):7.1f}' for x in traj[0])}]")
        print(f"  End   (deg): [{', '.join(f'{np.rad2deg(x):7.1f}' for x in traj[-1])}]")
        print(f"  Total motion (deg): [{', '.join(f'{np.rad2deg(x):7.1f}' for x in traj[-1] - traj[0])}]")
        print()

    # Plot
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available — skipping plots.")
        return

    joint_names = policy.obs_keys
    n_joints = len(joint_names)
    n_prims = len(all_trajectories)

    fig, axes = plt.subplots(n_joints, 1, figsize=(12, 3 * n_joints), sharex=True)
    if n_joints == 1:
        axes = [axes]

    colors = {"dig": "tab:red", "dump": "tab:blue", "swing": "tab:green", "return": "tab:orange"}

    for pname, traj in all_trajectories.items():
        traj_deg = np.rad2deg(traj)
        t = np.arange(len(traj))
        for j, ax in enumerate(axes):
            ax.plot(t, traj_deg[:, j], color=colors.get(pname, "gray"),
                    linewidth=2, label=pname)
            ax.set_ylabel(f"{joint_names[j]}\n(degrees)")
            ax.grid(True, alpha=0.3)

    axes[0].legend(loc="upper right")
    axes[-1].set_xlabel("Step")
    fig.suptitle(f"BC Policy Rollout ({args.steps} steps)", fontsize=14)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
