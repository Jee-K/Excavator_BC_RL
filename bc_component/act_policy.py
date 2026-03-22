import numpy as np
from dataclasses import dataclass, asdict
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


PHASE_TO_ID = {"dig": 0, "dump": 1, "swing": 2, "return": 3}
ID_TO_PHASE = {v: k for k, v in PHASE_TO_ID.items()}


@dataclass
class ACTConfig:
    """Configuration for a phase-conditioned ACT-style policy.

    This version is adapted to excavator joint-trajectory imitation first,
    while leaving hooks for later image conditioning.
    """

    obs_dim: int
    action_dim: int
    phase_dim: int = 4
    obs_horizon: int = 8
    action_chunk: int = 16
    latent_dim: int = 32
    d_model: int = 256
    nhead: int = 8
    num_encoder_layers: int = 4
    num_decoder_layers: int = 4
    dim_feedforward: int = 512
    dropout: float = 0.1
    use_vision: bool = False
    image_channels: int = 3
    image_size: int = 128
    max_views: int = 1
    action_mode: str = "delta"  # {delta, absolute}
    kl_weight: float = 10.0


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        return x + self.pe[:, : x.size(1)]


class SmallVisionEncoder(nn.Module):
    """A lightweight image encoder placeholder.

    This is intentionally small so the starter implementation is dependency-light.
    Replace with ResNet/ViT for real image-conditioned training.
    """

    def __init__(self, in_channels: int, out_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.proj = nn.Linear(128, out_dim)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # images: [B, V, C, H, W]
        b, v, c, h, w = images.shape
        x = images.reshape(b * v, c, h, w)
        x = self.net(x).flatten(1)
        x = self.proj(x)
        x = x.reshape(b, v, -1)
        return x


class MLPBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ACTCVAEEncoder(nn.Module):
    """BERT-style encoder over [CLS], observation, phase, future actions."""

    def __init__(self, cfg: ACTConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.cls = nn.Parameter(torch.zeros(1, 1, cfg.d_model))
        self.obs_proj = nn.Linear(cfg.obs_horizon * cfg.obs_dim, cfg.d_model)
        self.phase_emb = nn.Embedding(cfg.phase_dim, cfg.d_model)
        self.action_proj = nn.Linear(cfg.action_dim, cfg.d_model)
        self.pos = PositionalEncoding(cfg.d_model, max_len=cfg.action_chunk + 8)
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=cfg.num_encoder_layers)
        self.mu = nn.Linear(cfg.d_model, cfg.latent_dim)
        self.logvar = nn.Linear(cfg.d_model, cfg.latent_dim)

    def forward(
        self,
        obs_history: torch.Tensor,
        phase: torch.Tensor,
        action_chunk: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        b = obs_history.size(0)
        cls = self.cls.expand(b, -1, -1)
        obs_token = self.obs_proj(obs_history.flatten(1)).unsqueeze(1)
        phase_token = self.phase_emb(phase).unsqueeze(1)
        action_tokens = self.action_proj(action_chunk)
        x = torch.cat([cls, obs_token, phase_token, action_tokens], dim=1)
        x = self.pos(x)
        x = self.encoder(x)
        pooled = x[:, 0]
        return self.mu(pooled), self.logvar(pooled)


class ACTDecoder(nn.Module):
    def __init__(self, cfg: ACTConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.obs_proj = nn.Linear(cfg.obs_horizon * cfg.obs_dim, cfg.d_model)
        self.phase_emb = nn.Embedding(cfg.phase_dim, cfg.d_model)
        self.latent_proj = nn.Linear(cfg.latent_dim, cfg.d_model)
        self.vision_encoder = SmallVisionEncoder(cfg.image_channels, cfg.d_model) if cfg.use_vision else None
        self.memory_pos = PositionalEncoding(cfg.d_model, max_len=32)
        self.query_embed = nn.Parameter(torch.randn(1, cfg.action_chunk, cfg.d_model) * 0.02)
        self.query_pos = PositionalEncoding(cfg.d_model, max_len=cfg.action_chunk + 1)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        dec_layer = nn.TransformerDecoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.memory_encoder = nn.TransformerEncoder(enc_layer, num_layers=max(1, cfg.num_encoder_layers // 2))
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=cfg.num_decoder_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(cfg.d_model),
            nn.Linear(cfg.d_model, cfg.action_dim),
        )

    def forward(
        self,
        obs_history: torch.Tensor,
        phase: torch.Tensor,
        z: torch.Tensor,
        images: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        b = obs_history.size(0)
        memory_tokens = [
            self.obs_proj(obs_history.flatten(1)).unsqueeze(1),
            self.phase_emb(phase).unsqueeze(1),
            self.latent_proj(z).unsqueeze(1),
        ]
        if self.vision_encoder is not None and images is not None:
            vision_tokens = self.vision_encoder(images)
            memory_tokens.append(vision_tokens)
        memory = torch.cat(memory_tokens, dim=1)
        memory = self.memory_pos(memory)
        memory = self.memory_encoder(memory)

        queries = self.query_embed.expand(b, -1, -1)
        queries = self.query_pos(queries)
        decoded = self.decoder(tgt=queries, memory=memory)
        return self.head(decoded)


class ACTPolicy(nn.Module):
    """ACT-style policy for excavator phases.

    The policy predicts a chunk of future actions conditioned on a recent window of
    observations and a discrete phase token: dig / dump / swing / return.
    """

    def __init__(self, cfg: ACTConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.cvae_encoder = ACTCVAEEncoder(cfg)
        self.decoder = ACTDecoder(cfg)

    @staticmethod
    def _reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(
        self,
        obs_history: torch.Tensor,
        phase: torch.Tensor,
        target_actions: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ) -> Dict[str, torch.Tensor]:
        if target_actions is not None:
            mu, logvar = self.cvae_encoder(obs_history, phase, target_actions)
            z = mu if deterministic else self._reparameterize(mu, logvar)
        else:
            batch = obs_history.size(0)
            mu = torch.zeros(batch, self.cfg.latent_dim, device=obs_history.device)
            logvar = torch.zeros_like(mu)
            z = mu

        pred = self.decoder(obs_history, phase, z, images=images)
        kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=-1).mean()
        return {
            "pred_actions": pred,
            "mu": mu,
            "logvar": logvar,
            "kl_loss": kl,
        }

    def bc_loss(
        self,
        batch: Dict[str, torch.Tensor],
        deterministic_latent: bool = False,
    ) -> Dict[str, torch.Tensor]:
        out = self.forward(
            obs_history=batch["obs_history"],
            phase=batch["phase"],
            target_actions=batch["target_actions"],
            images=batch.get("images"),
            deterministic=deterministic_latent,
        )
        pred = out["pred_actions"]
        target = batch["target_actions"]
        recon_l1 = F.l1_loss(pred, target)
        recon_l2 = F.mse_loss(pred, target)
        loss = recon_l1 + 0.1 * recon_l2 + self.cfg.kl_weight * out["kl_loss"]
        return {
            "loss": loss,
            "recon_l1": recon_l1.detach(),
            "recon_l2": recon_l2.detach(),
            "kl_loss": out["kl_loss"].detach(),
        }

    @torch.no_grad()
    def predict_chunk(
        self,
        obs_history: torch.Tensor,
        phase: torch.Tensor,
        images: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        out = self.forward(
            obs_history=obs_history,
            phase=phase,
            target_actions=None,
            images=images,
            deterministic=True,
        )
        return out["pred_actions"]

    def export_config(self) -> Dict[str, object]:
        return asdict(self.cfg)


class TemporalEnsembler:
    """ACT-style temporal ensembling over overlapping chunks.

    Each call adds a predicted chunk for the current step and returns the ensembled
    action for the current time index. Exponential weights favor more recent chunk
    predictions by default.
    """

    def __init__(self, action_dim: int, chunk_size: int, decay: float = 0.6) -> None:
        self.action_dim = action_dim
        self.chunk_size = chunk_size
        self.decay = decay
        self.reset()

    def reset(self) -> None:
        self.t = 0
        self.buffer: Dict[int, list[Tuple[torch.Tensor, float]]] = {}

    def add_chunk(self, chunk: torch.Tensor) -> None:
        # chunk: [K, A]
        if chunk.dim() != 2:
            raise ValueError(f"Expected chunk with shape [K, A], got {tuple(chunk.shape)}")
        for offset in range(chunk.size(0)):
            t_abs = self.t + offset
            weight = self.decay ** offset
            self.buffer.setdefault(t_abs, []).append((chunk[offset], weight))

    def pop_action(self) -> torch.Tensor:
        if self.t not in self.buffer:
            raise RuntimeError("No chunk has been added for the current time step.")
        entries = self.buffer.pop(self.t)
        weights = torch.tensor([w for _, w in entries], dtype=entries[0][0].dtype, device=entries[0][0].device)
        actions = torch.stack([a for a, _ in entries], dim=0)
        action = (weights[:, None] * actions).sum(dim=0) / weights.sum()
        self.t += 1
        return action


def build_phase_tensor(phase_names: list[str], device: Optional[torch.device] = None) -> torch.Tensor:
    ids = [PHASE_TO_ID[p] for p in phase_names]
    return torch.tensor(ids, dtype=torch.long, device=device)
