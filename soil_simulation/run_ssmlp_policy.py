"""
Run a trained single-step goal-conditioned MLP policy against a simulator.

This file is matched to the training setup in `single_step_mlp_bc_train_joint_scaled.py`:
- current 4-D joint state is scaled from raw joint limits into [-1, 1]
- the model predicts the next 4-D joint target in the same scaled space
- predictions are mapped back to raw joint positions before `apply_control(...)`
- the goal is normalized in grouped (3, 3) form using the saved train-split stats

Expected checkpoint layout:
    /outputs/ssmlp/best.pt
or:
    /outputs/ssmlp/latest.pt

The checkpoint must contain:
    ckpt["model_state_dict"]
    ckpt["config"]
where ckpt["config"]["model"] and ckpt["config"]["normalization"] come from
`single_step_mlp_bc_train_joint_scaled.py`.

Typical use inside your simulator code:

    import numpy as np
    from run_ssmlp_policy import run_policy_loop

    HARDCODED_GOAL = np.array([
        [x0, y0, z0],
        [x1, y1, z1],
        [x2, y2, z2],
    ], dtype=np.float32)

    run_policy_loop(
        sim_env=sim_env,
        viewer=viewer,
        preset=preset,
        goal=HARDCODED_GOAL,
        checkpoint_dir="/outputs/ssmlp",
        which="best.pt",
        device="auto",
        command_hz=10,
        joint_indices=[0, 1, 2, 3],  # only if your simulator exposes more than 4 joints
    )
"""
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn
from torchvision.transforms import Normalize

# from single_step_mlp_bc_train_joint_scaled import SingleStepGoalConditionedMLP

class SingleStepGoalConditionedMLP(nn.Module):
    """Same architecture used by the single-step trainer."""

    def __init__(
        self,
        state_dim: int,
        goal_dim: int,
        action_dim: int,
        hidden_dims: Sequence[int] = (256, 256),
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        dims = [state_dim + goal_dim, *hidden_dims, action_dim]
        layers: list[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
                if dropout > 0.0:
                    layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, state: Tensor, goal: Tensor) -> Tensor:
        x = torch.cat([state, goal], dim=-1)
        return self.net(x)


def choose_device(requested: str = "auto") -> torch.device:
    requested = requested.lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        return torch.device("cuda")
    if requested == "cpu":
        return torch.device("cpu")
    raise ValueError(f"Unsupported device option: {requested!r}")


def _as_float_tensor(x: Any, device: torch.device | None = None) -> Tensor:
    t = torch.as_tensor(x, dtype=torch.float32)
    if device is not None:
        t = t.to(device)
    return t


def scale_joint_to_unit(raw: Tensor, raw_min: Tensor, raw_max: Tensor) -> Tensor:
    """Map raw joint values from [raw_min, raw_max] to [-1, 1]."""
    return 2.0 * (raw - raw_min) / (raw_max - raw_min) - 1.0


def unscale_joint_from_unit(unit: Tensor, raw_min: Tensor, raw_max: Tensor) -> Tensor:
    """Map joint values from [-1, 1] back to [raw_min, raw_max]."""
    return 0.5 * (unit + 1.0) * (raw_max - raw_min) + raw_min


class JointScaledSSMLPPolicy:
    """
    Thin inference wrapper around a trained single-step MLP policy.

    The policy expects:
        raw current joint positions -> scaled into [-1, 1]
        raw goal in shape (3, 3) or flattenable to 9 values -> grouped-normalized

    It returns:
        raw next-step target joint positions in shape (4,)
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        joint_raw_min: Tensor,
        joint_raw_max: Tensor,
        goal_mean_grouped: Tensor,
        goal_std_grouped: Tensor,
        joint_indices: Sequence[int] | None = None,
        clamp_output: bool = True,
    ) -> None:
        self.model = model
        self.device = device
        self.joint_raw_min = _as_float_tensor(joint_raw_min, device)
        self.joint_raw_max = _as_float_tensor(joint_raw_max, device)
        self.goal_norm = Normalize(
            mean=_as_float_tensor(goal_mean_grouped).tolist(),
            std=_as_float_tensor(goal_std_grouped).tolist(),
        )
        self.joint_indices = list(joint_indices) if joint_indices is not None else None
        self.clamp_output = clamp_output

        if self.joint_raw_min.shape != (4,) or self.joint_raw_max.shape != (4,):
            raise ValueError(
                f"Expected joint_raw_min and joint_raw_max to each have shape (4,), got "
                f"{tuple(self.joint_raw_min.shape)} and {tuple(self.joint_raw_max.shape)}"
            )
        if not torch.all(self.joint_raw_max > self.joint_raw_min):
            raise ValueError("Every entry in joint_raw_max must be strictly greater than joint_raw_min.")

    def preprocess_goal(self, goal: np.ndarray | Sequence[float] | Tensor) -> Tensor:
        """
        Normalize a goal using the grouped (3, 3) coordinate-wise stats from training.

        Accepts:
            goal shaped (3, 3) or flattenable to 9 values.

        Returns:
            Tensor[1, 9]
        """
        goal_t = _as_float_tensor(goal, self.device)
        if goal_t.numel() != 9:
            raise ValueError(f"Expected goal to contain 9 values, got shape {tuple(goal_t.shape)}")

        goal_t = goal_t.reshape(3, 3)
        # Match training dataset logic:
        # (entries=3, coords=3) -> (coords=3, entries=3, width=1)
        g = goal_t.t().unsqueeze(-1)
        g = self.goal_norm(g)
        g = g.squeeze(-1).t().reshape(1, 9)
        return g

    def preprocess_state(self, current_joint_positions: np.ndarray | Sequence[float] | Tensor) -> Tensor:
        """
        Select the 4 policy joints if needed and scale them to [-1, 1].

        Returns:
            Tensor[1, 4]
        """
        if isinstance(current_joint_positions, Tensor):
            q = current_joint_positions.detach().cpu().numpy().astype(np.float32)
        else:
            q = np.asarray(current_joint_positions, dtype=np.float32)

        if self.joint_indices is not None:
            q = q[self.joint_indices]
        if q.shape != (4,):
            raise ValueError(
                f"Expected selected current_joint_positions to have shape (4,), got {q.shape}. "
                "Pass joint_indices if the simulator exposes more than the 4 policy joints."
            )

        q_t = _as_float_tensor(q, self.device)
        q_t = scale_joint_to_unit(q_t, self.joint_raw_min, self.joint_raw_max)
        return q_t.unsqueeze(0)

    @torch.no_grad()
    def act(self, current_joint_positions: np.ndarray | Sequence[float] | Tensor, goal: np.ndarray | Sequence[float] | Tensor) -> np.ndarray:
        """Return the next raw joint-position target as a NumPy array of shape (4,)."""
        state_t = self.preprocess_state(current_joint_positions)
        goal_t = self.preprocess_goal(goal)

        pred_unit = self.model(state_t, goal_t).squeeze(0)
        pred_raw = unscale_joint_from_unit(pred_unit, self.joint_raw_min, self.joint_raw_max)
        if self.clamp_output:
            pred_raw = torch.clamp(pred_raw, min=self.joint_raw_min, max=self.joint_raw_max)
        return pred_raw.detach().cpu().numpy()


def _extract_norm_config(config: Mapping[str, Any]) -> Mapping[str, Any]:
    norm_cfg = config.get("normalization")
    if norm_cfg is None:
        raise KeyError("Checkpoint config does not contain a 'normalization' section.")

    required_keys = [
        "joint_raw_min",
        "joint_raw_max",
        "goal_mean_grouped",
        "goal_std_grouped",
    ]
    missing = [k for k in required_keys if k not in norm_cfg]
    if missing:
        raise KeyError(f"Checkpoint normalization config is missing keys: {missing}")
    return norm_cfg


def load_ssmlp_policy(
    checkpoint_dir: str | Path = "/outputs/ssmlp",
    which: str = "best.pt",
    device: str = "auto",
    joint_indices: Sequence[int] | None = None,
    clamp_output: bool = True,
) -> JointScaledSSMLPPolicy:
    """
    Load the trained policy from a checkpoint directory.

    Args:
        checkpoint_dir: Directory containing best.pt or latest.pt.
        which: Usually "best.pt".
        device: "auto", "cuda", or "cpu".
        joint_indices: Optional indices selecting the 4 controlled joints from the simulator state.
        clamp_output: Whether to clamp raw predicted joint targets to the saved raw limits.
    """
    device_obj = choose_device(device)
    checkpoint_path = Path(checkpoint_dir) / which
    import os.path
    print(os.path.abspath(checkpoint_path))
    ckpt = torch.load(checkpoint_path, map_location=device_obj)

    if "config" not in ckpt or "model_state_dict" not in ckpt:
        raise KeyError(
            f"Checkpoint at {checkpoint_path} must contain 'config' and 'model_state_dict'."
        )

    config = ckpt["config"]
    model_cfg = config["model"]
    norm_cfg = _extract_norm_config(config)

    model = SingleStepGoalConditionedMLP(
        state_dim=int(model_cfg["state_dim"]),
        goal_dim=int(model_cfg["goal_dim"]),
        action_dim=int(model_cfg["action_dim"]),
        hidden_dims=model_cfg["hidden_dims"],
        dropout=float(model_cfg.get("dropout", 0.0)),
    ).to(device_obj)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    return JointScaledSSMLPPolicy(
        model=model,
        device=device_obj,
        joint_raw_min=_as_float_tensor(norm_cfg["joint_raw_min"]),
        joint_raw_max=_as_float_tensor(norm_cfg["joint_raw_max"]),
        goal_mean_grouped=_as_float_tensor(norm_cfg["goal_mean_grouped"]),
        goal_std_grouped=_as_float_tensor(norm_cfg["goal_std_grouped"]),
        joint_indices=joint_indices,
        clamp_output=clamp_output,
    )


@torch.no_grad()
def run_policy_loop(
    sim_env: Any,
    viewer: Any,
    preset: Any,
    goal: np.ndarray | Sequence[float] | Tensor,
    checkpoint_dir: str | Path = "/outputs/ssmlp",
    which: str = "best.pt",
    device: str = "auto",
    command_hz: int = 10,
    joint_indices: Sequence[int] | None = None,
    clamp_output: bool = True,
) -> None:
    """
    Run the trained best model against the simulator.

    This matches the simulator interaction pattern:

        while viewer.is_running():
            for _ in range(preset.fps // COMMAND_HZ):
                sim_env.step()
            current_joint_positions = sim_env.state_0.joint_q.numpy()
            target_joint_positions = policy.act(current_joint_positions, goal)
            sim_env.apply_control(target_joint_positions.tolist())
    """
    if command_hz <= 0:
        raise ValueError(f"command_hz must be positive, got {command_hz}")
    if preset.fps < command_hz:
        raise ValueError(
            f"preset.fps ({preset.fps}) must be >= command_hz ({command_hz}) so the control loop can step the sim."
        )

    policy = load_ssmlp_policy(
        checkpoint_dir=checkpoint_dir,
        which=which,
        device=device,
        joint_indices=joint_indices,
        clamp_output=clamp_output,
    )

    steps_per_command = preset.fps // command_hz

    while viewer.is_running():
        for _ in range(steps_per_command):
            sim_env.step()

        current_joint_positions = sim_env.state_0.joint_q.numpy()
        target_joint_positions = policy.act(current_joint_positions, goal)
        sim_env.apply_control(target_joint_positions.tolist())


if __name__ == "__main__":
    from excavator_mpm_soil import *
    viewer, args = newton.examples.init()

    preset = SIM_PRESETS["experimental"]

    sim_env = ExcavatorMPM(
        viewer,
        fidelity=preset,
        debug=True
    )

    COMMAND_HZ = 10

    HARDCODED_GOAL = np.array([[0,0,0],[1,0,2],[0,0,0]], dtype=np.float32)

    run_policy_loop(
        sim_env=sim_env,
        viewer=viewer,
        preset=preset,
        goal=HARDCODED_GOAL,
        checkpoint_dir='./bc_component/outputs/ssmlp_js',
        which='best.pt',
        device='auto',
        command_hz=10,
        joint_indices=[0, 1, 2, 3],
    )
