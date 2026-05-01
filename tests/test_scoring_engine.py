"""Unit tests for LTVScoringEngine cold-start path."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from backend.ml.cold_start import ColdStartScorer
from backend.ml.segmentation import assign_segment, compute_max_cac


def _make_lookup() -> pl.DataFrame:
    return pl.DataFrame({
        "vertical":          ["healthcare", "ecommerce", "fintech"],
        "company_size":      ["enterprise", "smb",       "mid_market"],
        "channel":           ["paid_search","organic",   "email"],
        "plan_tier":         ["enterprise_trial","free", "professional"],
        "ltv_36m_estimate":  [12000.0, 800.0, 5000.0],
        "ci_lower":          [8000.0,  400.0, 3000.0],
        "ci_upper":          [18000.0, 1600.0,8000.0],
        "cate_effect":       [500.0,    50.0,  200.0],
        "n_customers":       [50,       200,    80],
        "causal_model_version": ["causal_v1"] * 3,
    })


def test_cold_start_ltv_positive() -> None:
    scorer = ColdStartScorer(None)
    scorer._table = _make_lookup()
    result = scorer.score("healthcare", "enterprise", "paid_search", "enterprise_trial")
    assert result["ltv_36m"] > 0


def test_cold_start_has_all_required_keys() -> None:
    scorer = ColdStartScorer(None)
    scorer._table = _make_lookup()
    result = scorer.score("ecommerce", "smb", "organic", "free")
    required_keys = [
        "ltv_source", "ltv_36m", "ltv_12m", "ci_lower_36m", "ci_upper_36m",
        "segment", "recommended_max_cac", "match_quality", "firmographic_inputs"
    ]
    for k in required_keys:
        assert k in result, f"Missing key: {k}"


def test_cold_start_ltv_source() -> None:
    scorer = ColdStartScorer(None)
    scorer._table = _make_lookup()
    result = scorer.score("fintech", "mid_market", "email", "professional")
    assert result["ltv_source"] == "firmographic_prior"


def test_segment_boundary_values() -> None:
    assert assign_segment(10_001) == "champions"
    assert assign_segment(10_000) == "high_value"
    assert assign_segment(5_001)  == "high_value"
    assert assign_segment(5_000)  == "medium_value"
    assert assign_segment(1_001)  == "medium_value"
    assert assign_segment(1_000)  == "low_value"
    assert assign_segment(0)      == "low_value"


def test_cac_strictly_positive() -> None:
    for ltv in [0, 100, 500, 1000, 5000, 10000, 50000]:
        cac = compute_max_cac(float(ltv))
        assert cac >= 0