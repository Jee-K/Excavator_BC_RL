"""
Run a trained sequence goal-conditioned policy against a simulator.

This file is matched to the training setup in `train_sequence_policies.py`:
- current 4-D joint states are scaled from raw joint limits into [-1, 1]
- the goal is normalized in grouped (3, 3) form using the saved train-split stats
- the model consumes a fixed-length state-history window and predicts a chunk of future
  4-D joint targets in the same normalized space
- predictions are mapped back to raw joint positions before `apply_control(...)`

Expected checkpoint layout:
    /outputs/sequence_policy/best.pt
or:
    /outputs/sequence_policy/latest.pt

The checkpoint must contain:
    ckpt["model_state_dict"]
    ckpt["config"]
where ckpt["config"]["model"] and ckpt["config"]["normalization"] come from
`train_sequence_policies.py`.

Inference policy used here:
- maintain a rolling fixed-length history of observed command-time states
- left-pad the history with zeros and emit a history mask until enough states are observed
- run the policy to predict an action chunk
- apply only the FIRST predicted action, then replan at the next control tick

That last choice keeps the loop closed and matches the example single-step runner's
observe -> predict -> apply pattern. If you want open-loop execution of the full predicted
chunk later, change the selection in `SequenceChunkPolicy.predict_chunk(...)` or
`SequenceChunkPolicy.act(...)`.

Typical use inside your simulator code:

    import numpy as np
    from run_sequence_policy import run_policy_loop

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
        checkpoint_dir="/outputs/sequence_policy",
        which="best.pt",
        device="auto",
        command_hz=10,
        joint_indices=[0, 1, 2, 3],  # only if your simulator exposes more than 4 joints
    )
"""

from collections import deque
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn
from torchvision.transforms import Normalize

from models import ACTStylePolicy, GoalConditionedMLPPolicy, LSTMSeq2SeqPolicy


POLICY_TYPE_TO_NAME = {
    "GoalConditionedMLPPolicy": "mlp",
    "LSTMSeq2SeqPolicy": "lstm",
    "ACTStylePolicy": "act",
}


def choose_device(requested: str | torch.device = "auto") -> torch.device:
    if isinstance(requested, torch.device):
        return requested

    requested = str(requested).lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Requested device='cuda' but CUDA is not available.")
        return torch.device("cuda")
    if requested == "cpu":
        return torch.device("cpu")
    raise ValueError(f"Unsupported device request: {requested!r}. Use 'auto', 'cuda', or 'cpu'.")


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


def _extract_model_choice(config: Mapping[str, Any], model_cfg: Mapping[str, Any]) -> str:
    policy_type = model_cfg.get("policy_type")
    if policy_type in POLICY_TYPE_TO_NAME:
        return POLICY_TYPE_TO_NAME[str(policy_type)]

    model_choice = config.get("model_choice")
    if model_choice in {"mlp", "lstm", "act"}:
        return str(model_choice)

    raise KeyError(
        "Could not infer model type from checkpoint config. Expected model['policy_type'] "
        "or config['model_choice'] to identify one of: mlp, lstm, act."
    )


def build_model_from_config(config: Mapping[str, Any]) -> nn.Module:
    model_cfg = config.get("model")
    if model_cfg is None:
        raise KeyError("Checkpoint config does not contain a 'model' section.")

    model_name = _extract_model_choice(config, model_cfg)

    common = dict(
        state_dim=int(model_cfg["state_dim"]),
        goal_dim=int(model_cfg["goal_dim"]),
        action_dim=int(model_cfg["action_dim"]),
        history_len=int(model_cfg["history_len"]),
        action_horizon=int(model_cfg["action_horizon"]),
    )

    if model_name == "mlp":
        return GoalConditionedMLPPolicy(
            **common,
            hidden_dims=model_cfg["hidden_dims"],
            dropout=float(model_cfg.get("dropout", 0.0)),
            loss_type=str(model_cfg.get("loss_type", "smooth_l1")),
        )

    if model_name == "lstm":
        return LSTMSeq2SeqPolicy(
            **common,
            encoder_hidden_dim=int(model_cfg.get("encoder_hidden_dim", 256)),
            decoder_hidden_dim=int(model_cfg.get("decoder_hidden_dim", 256)),
            num_layers=int(model_cfg.get("num_layers", 2)),
            dropout=float(model_cfg.get("dropout", 0.1)),
            default_teacher_forcing_ratio=float(model_cfg.get("default_teacher_forcing_ratio", 0.0)),
            loss_type=str(model_cfg.get("loss_type", "smooth_l1")),
        )

    if model_name == "act":
        return ACTStylePolicy(
            **common,
            d_model=int(model_cfg.get("d_model", 256)),
            latent_dim=int(model_cfg.get("latent_dim", 32)),
            nhead=int(model_cfg.get("nhead", 8)),
            num_encoder_layers=int(model_cfg.get("num_encoder_layers", 4)),
            num_decoder_layers=int(model_cfg.get("num_decoder_layers", 3)),
            ff_dim=int(model_cfg.get("ff_dim", 512)),
            dropout=float(model_cfg.get("dropout", 0.1)),
            kl_weight=float(model_cfg.get("kl_weight", 1e-4)),
            recon_loss_type=str(model_cfg.get("recon_loss_type", "smooth_l1")),
            sample_prior_at_inference=bool(model_cfg.get("sample_prior_at_inference", False)),
        )

    raise ValueError(f"Unsupported model type: {model_name!r}")


class SequenceChunkPolicy:
    """
    Thin inference wrapper around a trained sequence goal-conditioned policy.

    The policy expects:
        raw current joint positions -> selected down to the 4 policy joints if needed,
        then scaled to [-1, 1]
        raw goal in shape (3, 3) or flattenable to 9 values -> grouped-normalized

    It returns:
        one raw next-step target joint position vector in shape (4,), chosen as the first
        step from the predicted action chunk.
    """

    def __init__(
        self,
        model: nn.Module,
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
        self.clamp_output = clamp_output
        self.history: deque[Tensor] = deque(maxlen=self.history_len)

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
        """Clear the rolling observation history. Call this after environment resets."""
        self.history.clear()

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
        g = goal_t.t().unsqueeze(-1)
        g = self.goal_norm(g)
        g = g.squeeze(-1).t().reshape(1, 9)
        return g

    def preprocess_state(self, current_joint_positions: np.ndarray | Sequence[float] | Tensor) -> Tensor:
        """
        Select the policy joints if needed and scale them to [-1, 1].

        Returns:
            Tensor[4]
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
        return q_t

    def _append_state(self, state_unit: Tensor) -> None:
        self.history.append(state_unit.detach().clone())

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
    def predict_chunk(
        self,
        current_joint_positions: np.ndarray | Sequence[float] | Tensor,
        goal: np.ndarray | Sequence[float] | Tensor,
    ) -> np.ndarray:
        """
        Return the full predicted raw action chunk as a NumPy array of shape (K, 4).

        The internal rolling history is updated with the current observation before prediction.
        """
        state_t = self.preprocess_state(current_joint_positions)
        self._append_state(state_t)
        state_history, history_mask = self._build_history_tensors()
        goal_t = self.preprocess_goal(goal)

        batch = {
            "state_history": state_history,
            "history_mask": history_mask,
            "goal": goal_t,
            "start_action": state_t.unsqueeze(0),
        }
        out = self.model(batch)
        pred_unit = out["predicted_actions"].squeeze(0)
        pred_raw = unscale_joint_from_unit(pred_unit, self.joint_raw_min, self.joint_raw_max)
        if self.clamp_output:
            pred_raw = torch.clamp(pred_raw, min=self.joint_raw_min, max=self.joint_raw_max)
        return pred_raw.detach().cpu().numpy()

    @torch.no_grad()
    def act(
        self,
        current_joint_positions: np.ndarray | Sequence[float] | Tensor,
        goal: np.ndarray | Sequence[float] | Tensor,
    ) -> np.ndarray:
        """Return the first raw joint-position target from the predicted chunk as shape (4,)."""
        pred_chunk = self.predict_chunk(current_joint_positions, goal)
        return pred_chunk[0]


def load_sequence_policy(
    checkpoint_dir: str | Path,
    which: str = "best.pt",
    device: str = "auto",
    joint_indices: Sequence[int] | None = None,
    clamp_output: bool = True,
) -> SequenceChunkPolicy:
    """
    Load a trained sequence policy from a checkpoint directory.

    Args:
        checkpoint_dir: Directory containing best.pt or latest.pt.
        which: Usually "best.pt".
        device: "auto", "cuda", or "cpu".
        joint_indices: Optional indices selecting the 4 controlled joints from the simulator state.
        clamp_output: Whether to clamp raw predicted joint targets to the saved raw limits.
    """
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
    return SequenceChunkPolicy(
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
    device: str = "auto",
    command_hz: int = 10,
    joint_indices: Sequence[int] | None = None,
    clamp_output: bool = True,
) -> None:
    """
    Run the trained sequence model against the simulator.

    This matches the single-step runner's interaction pattern:

        while viewer.is_running():
            for _ in range(preset.fps // command_hz):
                sim_env.step()
            current_joint_positions = sim_env.state_0.joint_q.numpy()
            target_joint_positions = policy.act(current_joint_positions, goal)
            sim_env.apply_control(target_joint_positions.tolist())

    The policy itself internally maintains the rolling history needed by the sequence model.
    """
    assert command_hz > 0
    assert preset.fps >= command_hz

    policy = load_sequence_policy(
        checkpoint_dir=checkpoint_dir,
        which=which,
        device=device,
        joint_indices=joint_indices,
        clamp_output=clamp_output,
    )
    policy.reset_history()

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
        checkpoint_dir="./bc_component/outputs/emlp_large_batch",
        which="best.pt",
        command_hz=10,
        joint_indices=[6, 7, 8, 9],
    )
