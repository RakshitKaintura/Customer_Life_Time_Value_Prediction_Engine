"""
Multi-Head Transformer LTV Model — PyTorch implementation.

Architecture:
    Input: purchase sequence of max length 50
    Each token: [cat_id, amount_bucket, days_delta, channel_id]

    Embedding Layer  → token embedding (vocab → 64) + positional encoding
    Transformer Encoder → 4 layers, 8 heads, FFN dim 256, dropout 0.1
    CLS token pooling  → 64-dim customer embedding (stored in pgvector)
    3 Prediction Heads → Linear(64→1) for LTV_12m / LTV_24m / LTV_36m

    Loss: Huber loss (robust to outlier LTV values)

Exports to ONNX for < 200ms inference in FastAPI (Phase 6).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger


# ─────────────────────────────────────────────────────────────
# Token feature dimensions
# ─────────────────────────────────────────────────────────────

TOKEN_FEATURES = {
    "cat_id":        {"vocab_size": 13,  "embed_dim": 16},
    "amount_bucket": {"vocab_size": 6,   "embed_dim": 8},   # 1–5 + PAD=0
    "days_delta":    {"max_val":    365, "embed_dim": 16},   # continuous → linear proj
    "channel_id":    {"vocab_size": 8,   "embed_dim": 8},
}
COMBINED_TOKEN_DIM = 16 + 8 + 16 + 8   # = 48  → projected to model_dim


# ─────────────────────────────────────────────────────────────
# Positional Encoding
# ─────────────────────────────────────────────────────────────

class PositionalEncoding(nn.Module):
    """
    Standard sinusoidal positional encoding (Vaswani et al. 2017).
    Adds sequence position information to token embeddings.
    """

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)   # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            x + positional encoding, shape unchanged
        """
        x = x + self.pe[:, : x.size(1), :]  # type: ignore[index]
        return self.dropout(x)


# ─────────────────────────────────────────────────────────────
# Token Embedding
# ─────────────────────────────────────────────────────────────

class PurchaseTokenEmbedding(nn.Module):
    """
    Embeds each purchase sequence token into a dense vector.

    Each token = [cat_id, amount_bucket, days_delta, channel_id]
    Categorical fields: nn.Embedding
    Continuous field (days_delta): linear projection after normalisation
    All field embeddings concatenated → linear projection to model_dim
    """

    def __init__(
        self,
        model_dim: int = 64,
        cat_vocab_size: int = 13,
        bucket_vocab_size: int = 6,
        channel_vocab_size: int = 8,
        cat_embed_dim: int = 16,
        bucket_embed_dim: int = 8,
        days_embed_dim: int = 16,
        channel_embed_dim: int = 8,
        max_days_delta: int = 365,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.max_days_delta = float(max_days_delta)
        combined_dim = cat_embed_dim + bucket_embed_dim + days_embed_dim + channel_embed_dim

        self.cat_embed     = nn.Embedding(cat_vocab_size,    cat_embed_dim,     padding_idx=0)
        self.bucket_embed  = nn.Embedding(bucket_vocab_size, bucket_embed_dim,  padding_idx=0)
        self.channel_embed = nn.Embedding(channel_vocab_size, channel_embed_dim, padding_idx=0)

        # days_delta is continuous — project to days_embed_dim
        self.days_proj = nn.Sequential(
            nn.Linear(1, days_embed_dim),
            nn.ReLU(),
        )

        # Project combined embedding to model_dim
        self.proj = nn.Sequential(
            nn.Linear(combined_dim, model_dim),
            nn.LayerNorm(model_dim),
            nn.Dropout(dropout),
        )

    def forward(self, tokens: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            tokens: dict with keys 'cat_id', 'amount_bucket', 'days_delta', 'channel_id'
                    each shape (batch, seq_len) except days_delta which is float

        Returns:
            Tensor of shape (batch, seq_len, model_dim)
        """
        cat     = self.cat_embed(tokens["cat_id"])        # (B, L, 16)
        bucket  = self.bucket_embed(tokens["amount_bucket"])  # (B, L, 8)
        channel = self.channel_embed(tokens["channel_id"])   # (B, L, 8)

        # days_delta: normalise to [0, 1] and project
        days_norm = tokens["days_delta"].float() / self.max_days_delta  # (B, L)
        days_feat = self.days_proj(days_norm.unsqueeze(-1))             # (B, L, 16)

        combined = torch.cat([cat, bucket, days_feat, channel], dim=-1)  # (B, L, 48)
        return self.proj(combined)                                         # (B, L, model_dim)


# ─────────────────────────────────────────────────────────────
# Main Transformer Model
# ─────────────────────────────────────────────────────────────

class LTVTransformer(nn.Module):
    """
    Multi-head Transformer Encoder for LTV prediction.

    The [CLS] token prepended to each sequence acts as a global
    representation of the customer's purchase history.
    Its final hidden state is used for:
      1. The 3 LTV prediction heads (12m / 24m / 36m)
      2. pgvector storage for lookalike similarity search

    Monte Carlo Dropout: during inference, call .enable_mc_dropout()
    and run N forward passes to estimate uncertainty.
    """

    def __init__(
        self,
        model_dim: int = 64,
        n_heads: int = 8,
        n_layers: int = 4,
        ffn_dim: int = 256,
        dropout: float = 0.1,
        max_seq_len: int = 50,
        cat_vocab_size: int = 13,
        bucket_vocab_size: int = 6,
        channel_vocab_size: int = 8,
    ) -> None:
        super().__init__()

        self.model_dim = model_dim

        # [CLS] token embedding (learnable)
        self.cls_token = nn.Parameter(torch.randn(1, 1, model_dim))

        # Token + positional embedding
        self.token_embedding = PurchaseTokenEmbedding(
            model_dim=model_dim,
            cat_vocab_size=cat_vocab_size,
            bucket_vocab_size=bucket_vocab_size,
            channel_vocab_size=channel_vocab_size,
        )
        self.pos_encoding = PositionalEncoding(
            d_model=model_dim,
            max_len=max_seq_len + 1,  # +1 for CLS
            dropout=dropout,
        )

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=n_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="relu",
            batch_first=True,
            norm_first=True,   # Pre-LN for stability
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=n_layers,
            norm=nn.LayerNorm(model_dim),
        )

        # 3 separate prediction heads
        self.head_12m = nn.Sequential(
            nn.Linear(model_dim, model_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim // 2, 1),
        )
        self.head_24m = nn.Sequential(
            nn.Linear(model_dim, model_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim // 2, 1),
        )
        self.head_36m = nn.Sequential(
            nn.Linear(model_dim, model_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim // 2, 1),
        )

        self._mc_dropout_enabled = False
        self._init_weights()

    def _init_weights(self) -> None:
        """Xavier uniform initialisation for linear layers."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.02)
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()

    def enable_mc_dropout(self) -> None:
        """Enable Monte Carlo Dropout for uncertainty estimation."""
        self._mc_dropout_enabled = True
        for module in self.modules():
            if isinstance(module, nn.Dropout):
                module.train()

    def disable_mc_dropout(self) -> None:
        self._mc_dropout_enabled = False

    def get_padding_mask(
        self, tokens: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Build padding mask — True where cat_id == 0 (PAD token).
        Shape: (batch, seq_len + 1) — +1 for CLS.
        """
        pad_mask = tokens["cat_id"] == 0   # (B, L)
        # CLS token is never masked
        cls_mask = torch.zeros(
            pad_mask.size(0), 1, dtype=torch.bool, device=pad_mask.device
        )
        return torch.cat([cls_mask, pad_mask], dim=1)   # (B, L+1)

    def forward(
        self,
        tokens: dict[str, torch.Tensor],
        return_embedding: bool = False,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            tokens:           dict with keys 'cat_id', 'amount_bucket',
                              'days_delta', 'channel_id' — each (B, L)
            return_embedding: if True, also return the CLS embedding

        Returns:
            dict with 'ltv_12m', 'ltv_24m', 'ltv_36m' (each (B,))
            and optionally 'embedding' (B, model_dim)
        """
        B = tokens["cat_id"].size(0)

        # Token + positional embedding
        x = self.token_embedding(tokens)        # (B, L, model_dim)
        cls = self.cls_token.expand(B, -1, -1)  # (B, 1, model_dim)
        x = torch.cat([cls, x], dim=1)          # (B, L+1, model_dim)
        x = self.pos_encoding(x)                # (B, L+1, model_dim)

        # Padding mask
        src_key_padding_mask = self.get_padding_mask(tokens)  # (B, L+1)

        # Transformer encoder
        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)  # (B, L+1, model_dim)

        # CLS token = customer embedding
        cls_out = x[:, 0, :]    # (B, model_dim)

        # LTV predictions — clamp to non-negative
        ltv_12m = F.softplus(self.head_12m(cls_out)).squeeze(-1)  # (B,)
        ltv_24m = F.softplus(self.head_24m(cls_out)).squeeze(-1)
        ltv_36m = F.softplus(self.head_36m(cls_out)).squeeze(-1)

        output: dict[str, torch.Tensor] = {
            "ltv_12m": ltv_12m,
            "ltv_24m": ltv_24m,
            "ltv_36m": ltv_36m,
        }
        if return_embedding:
            output["embedding"] = cls_out

        return output

    def predict_with_uncertainty(
        self,
        tokens: dict[str, torch.Tensor],
        n_samples: int = 50,
    ) -> dict[str, torch.Tensor]:
        """
        Monte Carlo Dropout uncertainty estimation.
        Runs n_samples forward passes with dropout enabled.

        Returns:
            dict with mean, std, lower (5th pct), upper (95th pct) for each horizon
        """
        self.enable_mc_dropout()
        samples: dict[str, list[torch.Tensor]] = {
            "ltv_12m": [], "ltv_24m": [], "ltv_36m": []
        }

        with torch.no_grad():
            for _ in range(n_samples):
                out = self.forward(tokens)
                for k in samples:
                    samples[k].append(out[k])

        self.disable_mc_dropout()

        results: dict[str, torch.Tensor] = {}
        for k in ["ltv_12m", "ltv_24m", "ltv_36m"]:
            stack = torch.stack(samples[k], dim=0)   # (n_samples, B)
            results[f"{k}_mean"]  = stack.mean(dim=0)
            results[f"{k}_std"]   = stack.std(dim=0)
            results[f"{k}_lower"] = torch.quantile(stack, 0.05, dim=0)
            results[f"{k}_upper"] = torch.quantile(stack, 0.95, dim=0)

        return results


# ─────────────────────────────────────────────────────────────
# Loss Function
# ─────────────────────────────────────────────────────────────

class MultiHorizonHuberLoss(nn.Module):
    """
    Combined Huber loss across 3 LTV horizons.
    Uses log1p transform to handle heavy right skew in LTV values.
    """

    def __init__(
        self,
        delta: float = 1.0,
        weights: tuple[float, float, float] = (0.3, 0.3, 0.4),
        positive_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.delta = delta
        self.w12, self.w24, self.w36 = weights
        self.positive_weight = max(float(positive_weight), 1.0)

    def _weighted_huber(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Optionally up-weight non-zero targets in zero-inflated LTV training."""
        per_sample = F.huber_loss(pred, target, delta=self.delta, reduction="none")
        if self.positive_weight <= 1.0:
            return per_sample.mean()

        weights = torch.where(
            target > 0,
            torch.full_like(target, self.positive_weight),
            torch.ones_like(target),
        )
        return (per_sample * weights).sum() / torch.clamp(weights.sum(), min=1e-8)

    def forward(
        self,
        pred: dict[str, torch.Tensor],
        target: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Args:
            pred:   {'ltv_12m': ..., 'ltv_24m': ..., 'ltv_36m': ...}
            target: same keys with ground-truth LTV values
        """
        # Log1p transform to stabilise training on skewed LTV
        p12 = torch.log1p(pred["ltv_12m"])
        p24 = torch.log1p(pred["ltv_24m"])
        p36 = torch.log1p(pred["ltv_36m"])

        t12 = torch.log1p(target["ltv_12m"])
        t24 = torch.log1p(target["ltv_24m"])
        t36 = torch.log1p(target["ltv_36m"])

        loss = (
            self.w12 * self._weighted_huber(p12, t12)
            + self.w24 * self._weighted_huber(p24, t24)
            + self.w36 * self._weighted_huber(p36, t36)
        )
        return loss


# ─────────────────────────────────────────────────────────────
# Model factory
# ─────────────────────────────────────────────────────────────

def build_model(config: dict[str, Any]) -> LTVTransformer:
    """Build LTVTransformer from a config dict (used by Optuna)."""
    return LTVTransformer(
        model_dim   = config.get("model_dim",   64),
        n_heads     = config.get("n_heads",     8),
        n_layers    = config.get("n_layers",    4),
        ffn_dim     = config.get("ffn_dim",     256),
        dropout     = config.get("dropout",     0.1),
        max_seq_len = config.get("max_seq_len", 50),
    )


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
