"""Unit tests for ColdStartScorer."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from backend.ml.cold_start import ColdStartScorer


def _make_lookup_table() -> pl.DataFrame:
    """Minimal firmographic lookup table for testing."""
    return pl.DataFrame({
        "vertical":          ["healthcare", "ecommerce", "fintech", "retail", "other"],
        "company_size":      ["enterprise", "smb",       "mid_market","smb",   "smb"],
        "channel":           ["paid_search","organic",   "email",    "organic","organic"],
        "plan_tier":         ["enterprise_trial","free", "professional","starter","free"],
        "ltv_36m_estimate":  [12000.0, 800.0, 5000.0, 600.0, 400.0],
        "ci_lower":          [8000.0,  400.0, 3000.0, 300.0, 200.0],
        "ci_upper":          [18000.0, 1600.0,8000.0, 1000.0,800.0],
        "cate_effect":       [500.0, 50.0, 200.0, 30.0, 20.0],
        "n_customers":       [50, 200, 80, 150, 100],
        "causal_model_version": ["causal_v1"] * 5,
    })


def test_scorer_exact_match() -> None:
    scorer = ColdStartScorer(None)
    scorer._table = _make_lookup_table()
    result = scorer.score("healthcare", "enterprise", "paid_search", "enterprise_trial")
    assert result["ltv_36m"] == pytest.approx(12000.0)
    assert result["match_quality"] == "exact"
    assert result["ltv_source"] == "firmographic_prior"


def test_scorer_partial_match() -> None:
    scorer = ColdStartScorer(None)
    scorer._table = _make_lookup_table()
    # Use a plan_tier not in table → falls back to partial
    result = scorer.score("healthcare", "enterprise", "paid_search", "unknown_tier")
    assert result["ltv_36m"] > 0
    assert result["match_quality"] in ["partial_channel", "partial_size", "vertical_only", "global_average", "exact"]


def test_scorer_global_fallback() -> None:
    scorer = ColdStartScorer(None)
    scorer._table = _make_lookup_table()
    result = scorer.score("unknown_vertical", "unknown_size", "unknown_channel", "unknown_tier")
    assert result["ltv_36m"] > 0
    assert result["ltv_source"] == "firmographic_prior"


def test_scorer_segment_assignment() -> None:
    scorer = ColdStartScorer(None)
    scorer._table = _make_lookup_table()
    result = scorer.score("healthcare", "enterprise", "paid_search", "enterprise_trial")
    assert result["segment"] == "champions"  # 12000 > 10000


def test_scorer_cac_recommendation() -> None:
    scorer = ColdStartScorer(None)
    scorer._table = _make_lookup_table()
    result = scorer.score("ecommerce", "smb", "organic", "free")
    # Max CAC should be 40% of LTV
    assert result["recommended_max_cac"] == pytest.approx(0.40 * result["ltv_36m"], rel=0.01)


def test_scorer_empty_table_fallback() -> None:
    scorer = ColdStartScorer(None)
    scorer._table = pl.DataFrame()
    result = scorer.score("healthcare", "enterprise", "paid_search", "enterprise_trial")
    assert result["ltv_36m"] > 0
    assert result["ltv_source"] == "firmographic_prior"


def test_scorer_ci_lower_leq_upper() -> None:
    scorer = ColdStartScorer(None)
    scorer._table = _make_lookup_table()
    result = scorer.score("fintech", "mid_market", "email", "professional")
    assert result["ci_lower_36m"] <= result["ci_upper_36m"]


def test_scorer_firmographic_inputs_in_response() -> None:
    scorer = ColdStartScorer(None)
    scorer._table = _make_lookup_table()
    result = scorer.score("retail", "smb", "organic", "starter")
    assert "firmographic_inputs" in result
    assert result["firmographic_inputs"]["vertical"] == "retail"