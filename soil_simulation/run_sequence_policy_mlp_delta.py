"""
Run a trained delta-target sequence MLP policy against a simulator.

This is the delta-action analogue of the absolute-position sequence runtime:
- current 4-D joint states are scaled from raw joint limits into [-1, 1]
- the goal is normalized in grouped (3, 3) form using the saved train-split stats
- the model consumes a fixed-length state-history window and predicts a chunk of
  cumulative deltas in normalized joint space, relative to the current/start pose
- those deltas are added back to the current normalized state and mapped to raw joint
  positions before apply_control(...)

Inference behavior:
- maintain a rolling command-time history of observed states
- update that history on every command tick, even when not replanning
- predict a future chunk when the pending command queue is empty
- execute the first `execute_steps_per_inference` steps from the chunk before replanning
"""
from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import newton
import torch
from torch import Tensor, nn
from torchvision.transforms import Normalize

Batch = dict[str, Tensor]


class MLPBlock(nn.Module):
    def __init__(self, dims: Sequence[int], dropout: float = 0.0) -> None:
        super().__init__()
        if len(dims) < 2:
            raise ValueError("dims must contain at least input and output sizes")

        layers: list[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
                if dropout > 0.0:
                    layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class DeltaGoalConditionedMLPPolicy(nn.Module):
    def __init__(
        self,
        state_dim: int,
        goal_dim: int,
        action_dim: int,
        history_len: int,
        action_horizon: int,
        hidden_dims: Sequence[int] = (512, 512),
        dropout: float = 0.0,
        loss_type: str = "smooth_l1",
    ) -> None:
        super().__init__()
        self.state_dim = int(state_dim)
        self.goal_dim = int(goal_dim)
        self.action_dim = int(action_dim)
        self.history_len = int(history_len)
        self.action_horizon = int(action_horizon)
        self.hidden_dims = list(hidden_dims)
        self.dropout = float(dropout)
        self.loss_type = str(loss_type)

        input_dim = self.history_len * self.state_dim + self.goal_dim
        output_dim = self.action_horizon * self.action_dim
        self.net = MLPBlock([input_dim, *self.hidden_dims, output_dim], dropout=self.dropout)
        self.loss_fn = self._get_loss_fn(self.loss_type)

        self.config = dict(
            policy_type=self.__class__.__name__,
            state_dim=self.state_dim,
            goal_dim=self.goal_dim,
            action_dim=self.action_dim,
            history_len=self.history_len,
            action_horizon=self.action_horizon,
            hidden_dims=list(self.hidden_dims),
            dropout=self.dropout,
            loss_type=self.loss_type,
            target_representation="cumulative_delta_from_start_action_in_normalized_joint_space",
        )

    @staticmethod
    def _get_loss_fn(loss_type: str):
        loss_type = loss_type.lower()
        if loss_type == "mse":
            return torch.nn.functional.mse_loss
        if loss_type in {"smooth_l1", "huber"}:
            return torch.nn.functional.smooth_l1_loss
        raise ValueError(f"Unsupported loss_type={loss_type!r}")

    def forward(self, batch: Batch) -> dict[str, Tensor]:
        state_history = batch["state_history"]
        goal = batch["goal"]

        if state_history.ndim != 3:
            raise ValueError(
                f"state_history must have shape [B, T, state_dim], got {tuple(state_history.shape)}"
            )
        if state_history.shape[1] != self.history_len:
            raise ValueError(
                f"Expected state history length {self.history_len}, got {state_history.shape[1]}"
            )

        flat_history = state_history.reshape(state_history.shape[0], self.history_len * self.state_dim)
        x = torch.cat([flat_history, goal], dim=-1)
        predicted_deltas = self.net(x).view(-1, self.action_horizon, self.action_dim)

        out: dict[str, Tensor] = {"predicted_deltas": predicted_deltas}
        action_targets = batch.get("action_targets")
        if action_targets is not None:
            if action_targets.shape[1] != self.action_horizon:
                raise ValueError(
                    f"Expected action horizon {self.action_horizon}, got {action_targets.shape[1]}"
                )
            out["loss"] = self.loss_fn(predicted_deltas, action_targets)
        return out



def choose_device(requested: str | torch.device = "auto") -> torch.device:
    if isinstance(requested, torch.device):
        return requested
    requested_l = str(requested).lower()
    if requested_l == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested_l == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Requested device='cuda' but CUDA is not available.")
        return torch.device("cuda")
    if requested_l == "cpu":
        return torch.device("cpu")
    raise ValueError(f"Unsupported device request: {requested!r}. Use 'auto', 'cuda', or 'cpu'.")


def _as_float_tensor(x: Any, device: torch.device | None = None) -> Tensor:
    t = torch.as_tensor(x, dtype=torch.float32)
    if device is not None:
        t = t.to(device)
    return t


def scale_joint_to_unit(raw: Tensor, raw_min: Tensor, raw_max: Tensor) -> Tensor:
    return 2.0 * (raw - raw_min) / (raw_max - raw_min) - 1.0


def unscale_joint_from_unit(unit: Tensor, raw_min: Tensor, raw_max: Tensor) -> Tensor:
    return 0.5 * (unit + 1.0) * (raw_max - raw_min) + raw_min


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


def build_model_from_config(config: Mapping[str, Any]) -> DeltaGoalConditionedMLPPolicy:
    model_cfg = config.get("model")
    if model_cfg is None:
        raise KeyError("Checkpoint config does not contain a 'model' section.")
    policy_type = str(model_cfg.get("policy_type", ""))
    if policy_type != "DeltaGoalConditionedMLPPolicy":
        raise ValueError(
            f"This runtime expects a DeltaGoalConditionedMLPPolicy checkpoint, got policy_type={policy_type!r}"
        )

    return DeltaGoalConditionedMLPPolicy(
        state_dim=int(model_cfg["state_dim"]),
        goal_dim=int(model_cfg["goal_dim"]),
        action_dim=int(model_cfg["action_dim"]),
        history_len=int(model_cfg["history_len"]),
        action_horizon=int(model_cfg["action_horizon"]),
        hidden_dims=model_cfg["hidden_dims"],
        dropout=float(model_cfg.get("dropout", 0.0)),
        loss_type=str(model_cfg.get("loss_type", "smooth_l1")),
    )


class DeltaSequenceChunkMLPPolicy:
    """
    Inference wrapper for a trained delta-target sequence MLP policy.

    The policy predicts cumulative deltas relative to the most recent observed state.
    """

    def __init__(
        self,
        model: DeltaGoalConditionedMLPPolicy,
        device: torch.device,
        history_len: int,
        action_horizon: int,
        joint_raw_min: Tensor,
        joint_raw_max: Tensor,
        goal_mean_grouped: Tensor,
        goal_std_grouped: Tensor,
        joint_indices: Sequence[int] | None = None,
        clamp_output: bool = True,
    ) -> None:
        self.model = model
        self.device = device
        self.history_len = int(history_len)
        self.action_horizon = int(action_horizon)
        self.joint_raw_min = _as_float_tensor(joint_raw_min, device)
        self.joint_raw_max = _as_float_tensor(joint_raw_max, device)
        self.goal_norm = Normalize(
            mean=_as_float_tensor(goal_mean_grouped).tolist(),
            std=_as_float_tensor(goal_std_grouped).tolist(),
        )
        self.joint_indices = list(joint_indices) if joint_indices is not None else None
        self.clamp_output = bool(clamp_output)
        self.history: deque[Tensor] = deque(maxlen=self.history_len)
        self.last_observed_state_unit: Tensor | None = None

        if self.joint_raw_min.shape != (4,) or self.joint_raw_max.shape != (4,):
            raise ValueError(
                f"Expected joint_raw_min and joint_raw_max to each have shape (4,), got "
                f"{tuple(self.joint_raw_min.shape)} and {tuple(self.joint_raw_max.shape)}"
            )
        if not torch.all(self.joint_raw_max > self.joint_raw_min):
            raise ValueError("Every entry in joint_raw_max must be strictly greater than joint_raw_min.")
        if self.history_len <= 0:
            raise ValueError(f"history_len must be positive, got {self.history_len}")
        if self.action_horizon <= 0:
            raise ValueError(f"action_horizon must be positive, got {self.action_horizon}")

    def reset_history(self) -> None:
        self.history.clear()
        self.last_observed_state_unit = None

    def preprocess_goal(self, goal: np.ndarray | Sequence[float] | Tensor) -> Tensor:
        goal_t = _as_float_tensor(goal, self.device)
        if goal_t.numel() != 9:
            raise ValueError(f"Expected goal to contain 9 values, got shape {tuple(goal_t.shape)}")

        goal_t = goal_t.reshape(3, 3)
        g = goal_t.t().unsqueeze(-1)
        g = self.goal_norm(g)
        g = g.squeeze(-1).t().reshape(1, 9)
        return g

    def preprocess_state(self, current_joint_positions: np.ndarray | Sequence[float] | Tensor) -> Tensor:
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
        return scale_joint_to_unit(q_t, self.joint_raw_min, self.joint_raw_max)

    def observe(self, current_joint_positions: np.ndarray | Sequence[float] | Tensor) -> None:
        state_t = self.preprocess_state(current_joint_positions)
        self.last_observed_state_unit = state_t
        self.history.append(state_t.detach().clone())

    def _build_history_tensors(self) -> tuple[Tensor, Tensor]:
        state_dim = int(self.joint_raw_min.numel())
        history = torch.zeros((self.history_len, state_dim), dtype=torch.float32, device=self.device)
        history_mask = torch.zeros((self.history_len,), dtype=torch.bool, device=self.device)

        valid_len = len(self.history)
        if valid_len > 0:
            stacked = torch.stack(list(self.history), dim=0)
            history[-valid_len:] = stacked
            history_mask[-valid_len:] = True

        return history.unsqueeze(0), history_mask.unsqueeze(0)

    @torch.no_grad()
    def predict_chunk(self, goal: np.ndarray | Sequence[float] | Tensor) -> np.ndarray:
        if self.last_observed_state_unit is None:
            raise RuntimeError("No observation has been provided yet. Call observe(...) before predict_chunk(...).")

        state_history, history_mask = self._build_history_tensors()
        goal_t = self.preprocess_goal(goal)
        start_action = self.last_observed_state_unit.unsqueeze(0)

        batch = {
            "state_history": state_history,
            "history_mask": history_mask,
            "goal": goal_t,
            "start_action": start_action,
        }
        out = self.model(batch)
        pred_delta_unit = out["predicted_deltas"].squeeze(0)
        pred_abs_unit = start_action.squeeze(0).unsqueeze(0) + pred_delta_unit
        pred_raw = unscale_joint_from_unit(pred_abs_unit, self.joint_raw_min, self.joint_raw_max)
        if self.clamp_output:
            pred_raw = torch.clamp(pred_raw, min=self.joint_raw_min, max=self.joint_raw_max)
        return pred_raw.detach().cpu().numpy()

    @torch.no_grad()
    def act(self, goal: np.ndarray | Sequence[float] | Tensor) -> np.ndarray:
        pred_chunk = self.predict_chunk(goal)
        return pred_chunk[0]


def load_delta_sequence_policy(
    checkpoint_dir: str | Path,
    which: str = "best.pt",
    device: str | torch.device = "auto",
    joint_indices: Sequence[int] | None = None,
    clamp_output: bool = True,
) -> DeltaSequenceChunkMLPPolicy:
    device_obj = choose_device(device)
    checkpoint_path = Path(checkpoint_dir) / which
    ckpt = torch.load(checkpoint_path, map_location=device_obj)

    if "config" not in ckpt or "model_state_dict" not in ckpt:
        raise KeyError(
            f"Checkpoint at {checkpoint_path} must contain 'config' and 'model_state_dict'."
        )

    config = ckpt["config"]
    norm_cfg = _extract_norm_config(config)
    model = build_model_from_config(config).to(device_obj)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    model_cfg = config["model"]
    return DeltaSequenceChunkMLPPolicy(
        model=model,
        device=device_obj,
        history_len=int(model_cfg["history_len"]),
        action_horizon=int(model_cfg["action_horizon"]),
        joint_raw_min=_as_float_tensor(norm_cfg["joint_raw_min"]),
        joint_raw_max=_as_float_tensor(norm_cfg["joint_raw_max"]),
        goal_mean_grouped=_as_float_tensor(norm_cfg["goal_mean_grouped"]),
        goal_std_grouped=_as_float_tensor(norm_cfg["goal_std_grouped"]),
        joint_indices=joint_indices,
        clamp_output=clamp_output,
    )


@torch.no_grad()
def run_policy_loop(
    sim_env,
    viewer,
    preset: "SimulationFidelity",
    goal: np.ndarray | Tensor,
    checkpoint_dir: str,
    which: str = "best.pt",
    device: str | torch.device = "auto",
    command_hz: int = 10,
    joint_indices: Sequence[int] | None = None,
    clamp_output: bool = True,
    execute_steps_per_inference: int = 1,
) -> None:
    """
    Run the trained delta-sequence MLP against the simulator.

    `execute_steps_per_inference` controls how many leading steps from the predicted
    chunk are executed before replanning from a fresh observation.
    """
    if command_hz <= 0:
        raise ValueError(f"command_hz must be positive, got {command_hz}")
    if preset.fps < command_hz:
        raise ValueError(f"preset.fps ({preset.fps}) must be >= command_hz ({command_hz})")
    if execute_steps_per_inference <= 0:
        raise ValueError(
            f"execute_steps_per_inference must be positive, got {execute_steps_per_inference}"
        )

    policy = load_delta_sequence_policy(
        checkpoint_dir=checkpoint_dir,
        which=which,
        device=device,
        joint_indices=joint_indices,
        clamp_output=clamp_output,
    )
    policy.reset_history()

    steps_per_command = preset.fps // command_hz
    pending_targets: deque[np.ndarray] = deque()

    while viewer.is_running():
        for _ in range(steps_per_command):
            sim_env.step()

        current_joint_positions = sim_env.state_0.joint_q.numpy()
        policy.observe(current_joint_positions)

        if not pending_targets:
            pred_chunk = np.asarray(policy.predict_chunk(goal), dtype=np.float32)
            if pred_chunk.ndim != 2 or pred_chunk.shape[1] != 4:
                raise ValueError(f"Predicted chunk has shape {pred_chunk.shape}, expected (K, 4)")
            run_len = min(int(execute_steps_per_inference), int(pred_chunk.shape[0]))
            for step_target in pred_chunk[:run_len]:
                pending_targets.append(np.asarray(step_target, dtype=np.float32))

        target_joint_positions = pending_targets.popleft()
        sim_env.apply_control(target_joint_positions.tolist())


if __name__ == "__main__":
    from excavator_mpm_soil import *

    viewer, args = newton.examples.init()

    preset = SIM_PRESETS["experimental"]
    HARDCODED_GOAL = np.array(
        [
            [0.8, 7.4940055485687e-17, 0.0],
            [0.8, 7.4940055485687e-17, 2.0],
            [0.8, 7.4940055485687e-17, 0.0],
        ],
        dtype=np.float32,
    )

    sim_env = ExcavatorMPM(
        viewer,
        fidelity=preset,
        task=TaskInfo((3.0, -0.8, 0), (4.0, 0.8, 0.5 * np.pi), (0.0, 0.55, 0.10, 0.55)),
        debug=False,
    )

    run_policy_loop(
        sim_env=sim_env,
        viewer=viewer,
        preset=preset,
        goal=HARDCODED_GOAL,
        checkpoint_dir="./bc_component/outputs/emlp_deltas_small_mse",
        which="best.pt",
        command_hz=10,
        joint_indices=[6, 7, 8, 9],
        execute_steps_per_inference=1,
    )
