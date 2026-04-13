"""
Offline goal-conditioned policy classes in PyTorch.

This module contains four separate policy classes for the setting:
    (goal, present/past joint-state history) -> future joint targets

Classes:
    - ACTStylePolicy
    - GoalConditionedMLPPolicy
    - LSTMSeq2SeqPolicy

Shared batch convention for `step(...)` / training calls:
    batch = {
        "state_history": Tensor[B, T, state_dim],
        "goal": Tensor[B, goal_dim],
        # optional during inference, required for supervised training
        "action_targets": Tensor[B, K, action_dim],
        # optional boolean padding mask for state history
        "history_mask": BoolTensor[B, T],  # True = valid token, False = padding
        # optional for autoregressive decoder warm start
        "start_action": Tensor[B, action_dim],
        # optional override for seq2seq teacher forcing
        "teacher_forcing_ratio": float,
    }
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, MutableMapping, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor, nn
import torch.nn.functional as F

# !!! models will be cast to the proper device at a later time


Batch = MutableMapping[str, Any]


@dataclass
class StepOutput:
    predicted_actions: Tensor
    loss: Optional[Tensor] = None
    metrics: Optional[Dict[str, Tensor]] = None


class SingleStepMLP(nn.Module):
    """
    Simple pointwise MLP baseline.

    Maps:
        (goal, current_state) -> next_state

    Batch convention:
        batch = {
            "state": Tensor[B, state_dim],
            "goal": Tensor[B, goal_dim],
            "action_targets": Tensor[B, action_dim],
        }
    """

    def __init__(
        self,
        state_dim: int,
        goal_dim: int,
        action_dim: int,
        hidden_dims: Sequence[int],
        dropout: float = 0.0,
        loss_func=F.mse_loss,
    ) -> None:
        super().__init__()

        self.state_dim = state_dim
        self.goal_dim = goal_dim
        self.action_dim = action_dim
        self.hidden_dims = list(hidden_dims)
        self.dropout = dropout
        self.loss_func = loss_func

        dims = [state_dim + goal_dim, *hidden_dims, action_dim]
        layers: List[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
                if dropout > 0.0:
                    layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def train_step(self, batch: Dict[str, Tensor]) -> Tensor:
        states = batch["state"]
        goals = batch["goal"]

        predicted_actions = self.forward(states, goals)
        action_targets = batch["action_targets"]
        loss = self.loss_func(predicted_actions, action_targets)
        return loss

    def forward(self, state: Tensor, goal: Tensor) -> Tensor:
        x = torch.cat([state, goal], dim=-1)
        predicted_actions = self.net(x)
        return predicted_actions


class PolicyBase(nn.Module):
    """Shared helpers for the policy classes."""

    def __init__(self) -> None:
        super().__init__()
        self.config: Dict[str, Any] = {}

    def _get_loss_fn(self, loss_type: str):
        loss_type = loss_type.lower()
        if loss_type == "mse":
            return F.mse_loss
        if loss_type in {"smooth_l1", "huber"}:
            return F.smooth_l1_loss
        raise ValueError(f"Unsupported loss_type={loss_type!r}")

    def _history_valid_mask(self, batch: Batch, expected_len: int) -> Optional[Tensor]:
        history_mask = batch.get("history_mask")
        if history_mask is None:
            return None
        if history_mask.ndim != 2 or history_mask.shape[1] != expected_len:
            raise ValueError(
                f"history_mask must have shape [B, {expected_len}], got {tuple(history_mask.shape)}"
            )
        return history_mask.bool()

    def forward(self, batch: Batch) -> Dict[str, Tensor]:
        raise NotImplementedError

    def train_step(
        self,
        batch: Optional[Batch] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        clip_grad_norm: Optional[float] = None,
    ):
        """
        Two supported usages:

        1) Standard nn.Module behavior:
              model.train_step()              # set train mode
              model.train_step(False)         # set eval mode

        2) One supervised optimization step:
              out = model.train_step(batch=batch, optimizer=optimizer)

        Returns `self` in mode-switch mode, and a detached metrics dict in batch-update mode.
        """
        assert self.training # thus, we are in train mode
        
        optimizer.zero_grad(set_to_none=True)
        out = self.forward(batch)
        loss = out.get("loss")
        if loss is None:
            raise RuntimeError("step(batch) did not return a training loss.")
        loss.backward()
        if clip_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.parameters(), clip_grad_norm)
        optimizer.step()

        detached: Dict[str, Tensor] = {}
        for key, value in out.items():
            if isinstance(value, torch.Tensor):
                detached[key] = value.detach()
        return detached


class MLPBlock(nn.Module):
    def __init__(self, dims: Sequence[int], dropout: float = 0.0) -> None:
        super().__init__()
        if len(dims) < 2:
            raise ValueError("dims must contain at least input and output sizes")
        layers: List[nn.Module] = []
        for i in range(len(dims) - 1):
            in_dim = dims[i]
            out_dim = dims[i + 1]
            layers.append(nn.Linear(in_dim, out_dim))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
                if dropout > 0.0:
                    layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class ACTStylePolicy(PolicyBase):
    """
    Action-Chunking-Transformer-style policy with a simple CVAE latent.

    This captures the important structure of ACT-style policies for your non-visual case:
      - encode goal + state history,
      - decode a chunk of future joint targets,
      - use a latent variable during training to model demonstration variability.

    Notes:
    - This is not a line-by-line reproduction of the original visual ACT implementation.
    - The prior is taken to be a standard normal N(0, I).
    - The posterior network here summarizes the target action chunk with an MLP; if you
      need a closer match to a specific ACT variant, replace that summary module yourself.
    """

    def __init__(
        self,
        state_dim: int,
        goal_dim: int,
        action_dim: int,
        history_len: int,
        action_horizon: int,
        d_model: int = 256,
        latent_dim: int = 32,
        nhead: int = 8,
        num_encoder_layers: int = 4,
        num_decoder_layers: int = 3,
        ff_dim: int = 512,
        dropout: float = 0.1,
        kl_weight: float = 1e-4,
        recon_loss_type: str = "smooth_l1",
        sample_prior_at_inference: bool = False,
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.goal_dim = goal_dim
        self.action_dim = action_dim
        self.history_len = history_len
        self.action_horizon = action_horizon
        self.d_model = d_model
        self.latent_dim = latent_dim
        self.kl_weight = kl_weight
        self.recon_loss_fn = self._get_loss_fn(recon_loss_type)
        self.sample_prior_at_inference = sample_prior_at_inference

        self.state_proj = nn.Linear(state_dim, d_model)
        self.goal_proj = nn.Linear(goal_dim, d_model)
        self.history_pos = nn.Embedding(history_len + 1, d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.context_encoder = nn.TransformerEncoder(enc_layer, num_layers=num_encoder_layers)

        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=num_decoder_layers)

        self.action_chunk_encoder = MLPBlock(
            [action_horizon * action_dim, ff_dim, d_model],
            dropout=dropout,
        )
        self.posterior_net = MLPBlock([2 * d_model, ff_dim, 2 * latent_dim], dropout=dropout)
        self.latent_to_query = nn.Linear(latent_dim, d_model)
        self.future_queries = nn.Parameter(torch.randn(action_horizon, d_model) * 0.02)
        self.future_query_pos = nn.Embedding(action_horizon, d_model)
        self.output_head = nn.Linear(d_model, action_dim)

        self.config = dict(
            policy_type=self.__class__.__name__,
            state_dim=state_dim,
            goal_dim=goal_dim,
            action_dim=action_dim,
            history_len=history_len,
            action_horizon=action_horizon,
            d_model=d_model,
            latent_dim=latent_dim,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            ff_dim=ff_dim,
            dropout=dropout,
            kl_weight=kl_weight,
            recon_loss_type=recon_loss_type,
            sample_prior_at_inference=sample_prior_at_inference,
        )

    def _encode_context(self, state_history: Tensor, goal: Tensor, history_mask: Optional[Tensor]) -> Tensor:
        batch_size, seq_len, _ = state_history.shape
        state_tokens = self.state_proj(state_history)
        goal_token = self.goal_proj(goal).unsqueeze(1)
        context = torch.cat([goal_token, state_tokens], dim=1)

        positions = torch.arange(seq_len + 1, device=state_history.device)
        context = context + self.history_pos(positions).unsqueeze(0)

        src_key_padding_mask = None
        if history_mask is not None:
            src_key_padding_mask = torch.cat(
                [torch.zeros(batch_size, 1, dtype=torch.bool, device=history_mask.device), ~history_mask],
                dim=1,
            )
        return self.context_encoder(context, src_key_padding_mask=src_key_padding_mask)

    def _posterior(self, memory: Tensor, action_targets: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        context_summary = memory.mean(dim=1)
        action_summary = self.action_chunk_encoder(action_targets.reshape(action_targets.shape[0], -1))
        stats = self.posterior_net(torch.cat([context_summary, action_summary], dim=-1))
        mu, logvar = torch.chunk(stats, chunks=2, dim=-1)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z, mu, logvar

    def _prior_sample(self, batch_size: int, device: torch.device) -> Tensor:
        if self.sample_prior_at_inference:
            return torch.randn(batch_size, self.latent_dim, device=device)
        return torch.zeros(batch_size, self.latent_dim, device=device)

    def _decode(self, memory: Tensor, z: Tensor) -> Tensor:
        batch_size = memory.shape[0]
        query_latent = self.latent_to_query(z).unsqueeze(1)
        queries = self.future_queries.unsqueeze(0).expand(batch_size, -1, -1) + query_latent
        positions = torch.arange(self.action_horizon, device=memory.device)
        queries = queries + self.future_query_pos(positions).unsqueeze(0)
        decoded = self.decoder(tgt=queries, memory=memory)
        return self.output_head(decoded)

    @staticmethod
    def _kl_standard_normal(mu: Tensor, logvar: Tensor) -> Tensor:
        return 0.5 * torch.mean(torch.exp(logvar) + mu.pow(2) - 1.0 - logvar)

    def forward(self, batch: Batch) -> Dict[str, Tensor]:
        state_history = batch["state_history"]
        goal = batch["goal"]
        if state_history.shape[1] != self.history_len:
            raise ValueError(
                f"Expected state history length {self.history_len}, got {state_history.shape[1]}"
            )

        history_mask = self._history_valid_mask(batch, self.history_len)
        memory = self._encode_context(state_history, goal, history_mask)

        action_targets = batch.get("action_targets")
        if action_targets is not None:
            if action_targets.shape[1] != self.action_horizon:
                raise ValueError(
                    f"Expected action horizon {self.action_horizon}, got {action_targets.shape[1]}"
                )
            z, mu, logvar = self._posterior(memory, action_targets)
            predicted_actions = self._decode(memory, z)
            recon_loss = self.recon_loss_fn(predicted_actions, action_targets)
            kl_loss = self._kl_standard_normal(mu, logvar)
            loss = recon_loss + self.kl_weight * kl_loss
            return {
                "predicted_actions": predicted_actions,
                "loss": loss,
                "recon_loss": recon_loss,
                "kl_loss": kl_loss,
                "latent_mu": mu,
                "latent_logvar": logvar,
            }

        z = self._prior_sample(batch_size=state_history.shape[0], device=state_history.device)
        predicted_actions = self._decode(memory, z)
        return {"predicted_actions": predicted_actions}


class GoalConditionedMLPPolicy(PolicyBase):
    """
    Simple goal-conditioned MLP baseline.

    Encodes a fixed window of state history by flattening it, concatenates the goal,
    and predicts a future chunk of joint targets.
    """

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
        self.state_dim = state_dim
        self.goal_dim = goal_dim
        self.action_dim = action_dim
        self.history_len = history_len
        self.action_horizon = action_horizon
        self.loss_fn = self._get_loss_fn(loss_type)

        input_dim = history_len * state_dim + goal_dim
        output_dim = action_horizon * action_dim
        dims = [input_dim, *hidden_dims, output_dim]
        self.net = MLPBlock(dims, dropout=dropout)

        self.config = dict(
            policy_type=self.__class__.__name__,
            state_dim=state_dim,
            goal_dim=goal_dim,
            action_dim=action_dim,
            history_len=history_len,
            action_horizon=action_horizon,
            hidden_dims=list(hidden_dims),
            dropout=dropout,
            loss_type=loss_type,
        )

    def forward(self, batch: Batch) -> Dict[str, Tensor]:
        state_history = batch["state_history"]
        goal = batch["goal"]
        if state_history.shape[1] != self.history_len:
            raise ValueError(
                f"Expected state history length {self.history_len}, got {state_history.shape[1]}"
            )
        flat_history = state_history.reshape(state_history.shape[0], self.history_len * self.state_dim)
        x = torch.cat([flat_history, goal], dim=-1)
        predicted_actions = self.net(x).view(-1, self.action_horizon, self.action_dim)

        out: Dict[str, Tensor] = {"predicted_actions": predicted_actions}
        action_targets = batch.get("action_targets")
        if action_targets is not None:
            if action_targets.shape[1] != self.action_horizon:
                raise ValueError(
                    f"Expected action horizon {self.action_horizon}, got {action_targets.shape[1]}"
                )
            loss = self.loss_fn(predicted_actions, action_targets)
            out["loss"] = loss
        return out


class LSTMSeq2SeqPolicy(PolicyBase):
    """
    Goal-conditioned LSTM encoder / autoregressive decoder.

    Design choices made here:
    - Encoder input at each step is [state_t, goal].
    - Decoder input at each step is [previous_action, goal].
    - A learnable start token is used for the first decoder step unless the batch supplies
      `start_action`.

    If your application requires a different warm-start signal for the decoder (e.g. the
    last commanded action, a current reference target, or a phase variable), replace that
    input explicitly rather than assuming the learned start token is always appropriate.
    """

    def __init__(
        self,
        state_dim: int,
        goal_dim: int,
        action_dim: int,
        history_len: int,
        action_horizon: int,
        encoder_hidden_dim: int = 256,
        decoder_hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1,
        default_teacher_forcing_ratio: float = 0.5,
        loss_type: str = "smooth_l1",
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.goal_dim = goal_dim
        self.action_dim = action_dim
        self.history_len = history_len
        self.action_horizon = action_horizon
        self.encoder_hidden_dim = encoder_hidden_dim
        self.decoder_hidden_dim = decoder_hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.default_teacher_forcing_ratio = default_teacher_forcing_ratio
        self.loss_fn = self._get_loss_fn(loss_type)

        enc_dropout = dropout if num_layers > 1 else 0.0
        self.encoder = nn.LSTM(
            input_size=state_dim + goal_dim,
            hidden_size=encoder_hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=enc_dropout,
        )

        self.decoder_cells = nn.ModuleList()
        for layer_idx in range(num_layers):
            in_dim = action_dim + goal_dim if layer_idx == 0 else decoder_hidden_dim
            self.decoder_cells.append(nn.LSTMCell(in_dim, decoder_hidden_dim))

        self.init_h = nn.ModuleList([nn.Linear(encoder_hidden_dim, decoder_hidden_dim) for _ in range(num_layers)])
        self.init_c = nn.ModuleList([nn.Linear(encoder_hidden_dim, decoder_hidden_dim) for _ in range(num_layers)])
        self.output_head = nn.Linear(decoder_hidden_dim, action_dim)
        self.start_action = nn.Parameter(torch.zeros(action_dim))

        self.config = dict(
            policy_type=self.__class__.__name__,
            state_dim=state_dim,
            goal_dim=goal_dim,
            action_dim=action_dim,
            history_len=history_len,
            action_horizon=action_horizon,
            encoder_hidden_dim=encoder_hidden_dim,
            decoder_hidden_dim=decoder_hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            default_teacher_forcing_ratio=default_teacher_forcing_ratio,
            loss_type=loss_type,
        )

    def _initialize_decoder_state(self, enc_h_n: Tensor, enc_c_n: Tensor) -> Tuple[List[Tensor], List[Tensor]]:
        # enc_h_n / enc_c_n: [num_layers, B, encoder_hidden_dim]
        h_list: List[Tensor] = []
        c_list: List[Tensor] = []
        for layer in range(self.num_layers):
            h_list.append(torch.tanh(self.init_h[layer](enc_h_n[layer])))
            c_list.append(torch.tanh(self.init_c[layer](enc_c_n[layer])))
        return h_list, c_list

    def _decode(
        self,
        goal: Tensor,
        init_state: Tuple[List[Tensor], List[Tensor]],
        action_targets: Optional[Tensor],
        teacher_forcing_ratio: float,
        start_action: Optional[Tensor],
    ) -> Tensor:
        batch_size = goal.shape[0]
        h_list, c_list = init_state
        outputs: List[Tensor] = []

        if start_action is None:
            prev_action = self.start_action.unsqueeze(0).expand(batch_size, -1)
        else:
            prev_action = start_action

        for t in range(self.action_horizon):
            x = torch.cat([prev_action, goal], dim=-1)
            for layer_idx, cell in enumerate(self.decoder_cells):
                h_new, c_new = cell(x, (h_list[layer_idx], c_list[layer_idx]))
                if self.training and self.dropout > 0.0 and layer_idx < self.num_layers - 1:
                    x = F.dropout(h_new, p=self.dropout, training=True)
                else:
                    x = h_new
                h_list[layer_idx] = h_new
                c_list[layer_idx] = c_new

            action_t = self.output_head(h_list[-1])
            outputs.append(action_t)

            use_teacher = (
                action_targets is not None
                and self.training
                and torch.rand(1, device=goal.device).item() < teacher_forcing_ratio
            )
            prev_action = action_targets[:, t] if use_teacher else action_t

        return torch.stack(outputs, dim=1)

    def forward(self, batch: Batch) -> Dict[str, Tensor]:
        state_history = batch["state_history"]
        goal = batch["goal"]
        if state_history.shape[1] != self.history_len:
            raise ValueError(
                f"Expected state history length {self.history_len}, got {state_history.shape[1]}"
            )

        goal_seq = goal.unsqueeze(1).expand(-1, self.history_len, -1)
        encoder_inputs = torch.cat([state_history, goal_seq], dim=-1)
        _, (enc_h_n, enc_c_n) = self.encoder(encoder_inputs)
        init_state = self._initialize_decoder_state(enc_h_n, enc_c_n)

        action_targets = batch.get("action_targets")
        start_action = batch.get("start_action")
        teacher_forcing_ratio = float(batch.get("teacher_forcing_ratio", self.default_teacher_forcing_ratio))
        predicted_actions = self._decode(
            goal=goal,
            init_state=init_state,
            action_targets=action_targets,
            teacher_forcing_ratio=teacher_forcing_ratio,
            start_action=start_action,
        )

        out: Dict[str, Tensor] = {"predicted_actions": predicted_actions}
        if action_targets is not None:
            if action_targets.shape[1] != self.action_horizon:
                raise ValueError(
                    f"Expected action horizon {self.action_horizon}, got {action_targets.shape[1]}"
                )
            out["loss"] = self.loss_fn(predicted_actions, action_targets)
        return out


__all__ = [
    "ACTStylePolicy",
    "GoalConditionedMLPPolicy",
    "LSTMSeq2SeqPolicy",
]
