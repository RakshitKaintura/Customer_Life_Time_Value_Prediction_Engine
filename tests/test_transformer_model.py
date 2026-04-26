"""Unit tests for LTVTransformer."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from backend.ml.transformer_model import (
    LTVTransformer,
    MultiHorizonHuberLoss,
    PositionalEncoding,
    PurchaseTokenEmbedding,
    build_model,
    count_parameters,
)


def _dummy_tokens(batch: int = 4, seq_len: int = 10) -> dict[str, torch.Tensor]:
    rng = np.random.default_rng(0)
    return {
        "cat_id":        torch.from_numpy(rng.integers(0, 13,  (batch, seq_len)).astype(np.int64)),
        "amount_bucket": torch.from_numpy(rng.integers(0, 6,   (batch, seq_len)).astype(np.int64)),
        "days_delta":    torch.from_numpy(rng.integers(0, 366, (batch, seq_len)).astype(np.int64)),
        "channel_id":    torch.from_numpy(rng.integers(0, 8,   (batch, seq_len)).astype(np.int64)),
    }


@pytest.fixture
def model() -> LTVTransformer:
    return LTVTransformer(model_dim=64, n_heads=4, n_layers=2, ffn_dim=128, max_seq_len=10)


def test_forward_output_keys(model: LTVTransformer) -> None:
    tokens = _dummy_tokens()
    out = model(tokens)
    assert "ltv_12m" in out
    assert "ltv_24m" in out
    assert "ltv_36m" in out


def test_forward_output_shape(model: LTVTransformer) -> None:
    B = 4
    tokens = _dummy_tokens(batch=B)
    out = model(tokens)
    for k in ["ltv_12m", "ltv_24m", "ltv_36m"]:
        assert out[k].shape == (B,), f"{k} shape mismatch"


def test_forward_non_negative(model: LTVTransformer) -> None:
    tokens = _dummy_tokens()
    out = model(tokens)
    for k in ["ltv_12m", "ltv_24m", "ltv_36m"]:
        assert (out[k] >= 0).all(), f"{k} has negative values"


def test_forward_with_embedding(model: LTVTransformer) -> None:
    tokens = _dummy_tokens()
    out = model(tokens, return_embedding=True)
    assert "embedding" in out
    assert out["embedding"].shape == (4, 64)


def test_mc_dropout_produces_variation(model: LTVTransformer) -> None:
    tokens = _dummy_tokens(batch=8)
    result = model.predict_with_uncertainty(tokens, n_samples=20)
    assert "ltv_12m_std" in result
    # Std should be positive for most samples
    assert (result["ltv_12m_std"] > 0).any()


def test_mc_dropout_ci_ordering(model: LTVTransformer) -> None:
    tokens = _dummy_tokens()
    result = model.predict_with_uncertainty(tokens, n_samples=30)
    assert (result["ltv_36m_upper"] >= result["ltv_36m_lower"]).all()


def test_padding_mask_cls_never_masked(model: LTVTransformer) -> None:
    tokens = _dummy_tokens()
    mask = model.get_padding_mask(tokens)
    # First column (CLS) must be all False (not masked)
    assert not mask[:, 0].any()


def test_positional_encoding_shape() -> None:
    pe = PositionalEncoding(d_model=64, max_len=50, dropout=0.0)
    x = torch.zeros(4, 10, 64)
    out = pe(x)
    assert out.shape == x.shape


def test_token_embedding_shape() -> None:
    emb = PurchaseTokenEmbedding(model_dim=64)
    tokens = _dummy_tokens(batch=4, seq_len=8)
    out = emb(tokens)
    assert out.shape == (4, 8, 64)


def test_loss_forward() -> None:
    loss_fn = MultiHorizonHuberLoss()
    pred    = {"ltv_12m": torch.rand(8), "ltv_24m": torch.rand(8), "ltv_36m": torch.rand(8)}
    target  = {"ltv_12m": torch.rand(8), "ltv_24m": torch.rand(8), "ltv_36m": torch.rand(8)}
    loss = loss_fn(pred, target)
    assert loss.item() >= 0
    assert not torch.isnan(loss)


def test_build_model_from_config() -> None:
    config = {"n_layers": 2, "n_heads": 4, "ffn_dim": 128, "dropout": 0.1, "max_seq_len": 10}
    m = build_model(config)
    assert isinstance(m, LTVTransformer)


def test_count_parameters() -> None:
    model = LTVTransformer(model_dim=64, n_heads=4, n_layers=2, ffn_dim=128)
    n = count_parameters(model)
    assert n > 0
    assert n < 10_000_000   # should be small


def test_no_grad_in_eval() -> None:
    model = LTVTransformer(model_dim=64, n_heads=4, n_layers=2, ffn_dim=128, max_seq_len=10)
    model.eval()
    tokens = _dummy_tokens()
    with torch.no_grad():
        out = model(tokens)
    assert out["ltv_36m"].requires_grad is False