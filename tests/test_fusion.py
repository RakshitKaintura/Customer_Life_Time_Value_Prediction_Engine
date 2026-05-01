"""Unit tests for XGBoostMetaLearner fusion model."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from backend.ml.fusion import XGBoostMetaLearner, build_meta_features, META_FEATURES
from backend.ml.segmentation import assign_segment, compute_max_cac, assign_segments_batch


# ─────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────

def _make_meta_df(n: int = 200, seed: int = 42) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    return pl.DataFrame({
        "customer_id":              [f"C{i:05d}" for i in range(n)],
        "bgnbd_ltv_12m":            rng.uniform(100, 5000, n).tolist(),
        "bgnbd_ltv_36m":            rng.uniform(200, 12000, n).tolist(),
        "transformer_ltv_12m":      rng.uniform(100, 5000, n).tolist(),
        "transformer_ltv_36m":      rng.uniform(200, 12000, n).tolist(),
        "probability_alive":        rng.uniform(0, 1, n).tolist(),
        "frequency":                rng.integers(0, 10, n).astype(float).tolist(),
        "monetary_avg":             rng.uniform(10, 500, n).tolist(),
        "recency_days":             rng.uniform(0, 300, n).tolist(),
        "t_days":                   rng.integers(180, 730, n).astype(float).tolist(),
        "purchase_variance":        rng.uniform(0, 10000, n).tolist(),
        "orders_count":             rng.integers(1, 15, n).astype(float).tolist(),
        "avg_days_between_orders":  rng.uniform(5, 120, n).tolist(),
        "unique_categories":        rng.integers(1, 6, n).astype(float).tolist(),
    })


def _make_targets(n: int = 200) -> pl.DataFrame:
    rng = np.random.default_rng(99)
    return pl.DataFrame({
        "customer_id":    [f"C{i:05d}" for i in range(n)],
        "actual_ltv_12m": rng.uniform(0, 5000, n).tolist(),
        "actual_ltv_36m": rng.uniform(0, 12000, n).tolist(),
    })


@pytest.fixture
def fitted_fusion() -> XGBoostMetaLearner:
    meta   = _make_meta_df(200)
    targets = _make_targets(200)
    model  = XGBoostMetaLearner(model_version="test_fusion_v1")
    model.fit(meta, targets)
    return model


# ─────────────────────────────────────────────────────────────
# Fitting tests
# ─────────────────────────────────────────────────────────────

def test_fusion_fits_without_error() -> None:
    meta    = _make_meta_df(100)
    targets = _make_targets(100)
    model   = XGBoostMetaLearner()
    model.fit(meta, targets)
    assert model._is_fitted


def test_fusion_fit_stores_feature_names(fitted_fusion: XGBoostMetaLearner) -> None:
    assert len(fitted_fusion._feature_names) > 0
    for f in fitted_fusion._feature_names:
        assert f in META_FEATURES


def test_fusion_raises_if_not_fitted() -> None:
    model = XGBoostMetaLearner()
    meta  = _make_meta_df(10)
    with pytest.raises(RuntimeError, match="fitted"):
        model.predict(meta)


# ─────────────────────────────────────────────────────────────
# Prediction tests
# ─────────────────────────────────────────────────────────────

def test_fusion_predict_returns_polars(fitted_fusion: XGBoostMetaLearner) -> None:
    meta   = _make_meta_df(50)
    result = fitted_fusion.predict(meta)
    assert isinstance(result, pl.DataFrame)


def test_fusion_predict_row_count(fitted_fusion: XGBoostMetaLearner) -> None:
    meta   = _make_meta_df(50)
    result = fitted_fusion.predict(meta)
    assert len(result) == 50


def test_fusion_predict_non_negative(fitted_fusion: XGBoostMetaLearner) -> None:
    meta   = _make_meta_df(50)
    result = fitted_fusion.predict(meta)
    assert (result["ltv_12m"] >= 0).all()
    assert (result["ltv_36m"] >= 0).all()


def test_fusion_predict_columns(fitted_fusion: XGBoostMetaLearner) -> None:
    meta   = _make_meta_df(50)
    result = fitted_fusion.predict(meta)
    for col in ["customer_id", "ltv_12m", "ltv_24m", "ltv_36m",
                "meta_weight_bgnbd", "meta_weight_transformer"]:
        assert col in result.columns, f"Missing column: {col}"


def test_fusion_predict_weights_sum_approx_one(fitted_fusion: XGBoostMetaLearner) -> None:
    meta    = _make_meta_df(50)
    result  = fitted_fusion.predict(meta)
    w_sum   = (result["meta_weight_bgnbd"] + result["meta_weight_transformer"])
    assert (w_sum >= 0.9).all()
    assert (w_sum <= 1.1).all()


def test_fusion_predict_single_returns_dict(fitted_fusion: XGBoostMetaLearner) -> None:
    meta_dict = {f: 100.0 for f in fitted_fusion._feature_names}
    result = fitted_fusion.predict_single(meta_dict)
    assert "ltv_12m" in result
    assert "ltv_24m" in result
    assert "ltv_36m" in result
    assert result["ltv_12m"] >= 0


# ─────────────────────────────────────────────────────────────
# Serialisation tests
# ─────────────────────────────────────────────────────────────

def test_fusion_save_and_load(fitted_fusion: XGBoostMetaLearner) -> None:
    meta = _make_meta_df(30)
    pred_before = fitted_fusion.predict(meta)["ltv_36m"].to_numpy()

    with tempfile.TemporaryDirectory() as tmpdir:
        fitted_fusion.save_to_disk(tmpdir)
        loaded = XGBoostMetaLearner.load_from_disk(tmpdir, fitted_fusion.model_version)

    pred_after = loaded.predict(meta)["ltv_36m"].to_numpy()
    np.testing.assert_allclose(pred_before, pred_after, rtol=1e-5)


# ─────────────────────────────────────────────────────────────
# Segmentation tests
# ─────────────────────────────────────────────────────────────

def test_segment_champions() -> None:
    assert assign_segment(15_000) == "champions"


def test_segment_high_value() -> None:
    assert assign_segment(7_500) == "high_value"


def test_segment_medium_value() -> None:
    assert assign_segment(2_000) == "medium_value"


def test_segment_low_value() -> None:
    assert assign_segment(500) == "low_value"


def test_max_cac_champions() -> None:
    cac = compute_max_cac(15_000)
    assert cac == pytest.approx(15_000 * 0.50)


def test_max_cac_low_value() -> None:
    cac = compute_max_cac(800)
    assert cac == pytest.approx(800 * 0.20)


def test_assign_segments_batch_adds_columns() -> None:
    preds = pl.DataFrame({
        "customer_id": [f"C{i}" for i in range(10)],
        "ltv_12m":     [100.0 + i * 500 for i in range(10)],
        "ltv_24m":     [200.0 + i * 1000 for i in range(10)],
        "ltv_36m":     [300.0 + i * 2000 for i in range(10)],
    })
    result = assign_segments_batch(preds)
    assert "segment" in result.columns
    assert "recommended_max_cac" in result.columns
    assert "ltv_percentile" in result.columns


def test_assign_segments_batch_percentile_range() -> None:
    preds = pl.DataFrame({
        "customer_id": [f"C{i}" for i in range(50)],
        "ltv_12m":     [float(i * 100) for i in range(50)],
        "ltv_24m":     [float(i * 200) for i in range(50)],
        "ltv_36m":     [float(i * 300) for i in range(50)],
    })
    result = assign_segments_batch(preds)
    assert (result["ltv_percentile"] >= 0).all()
    assert (result["ltv_percentile"] <= 100).all()