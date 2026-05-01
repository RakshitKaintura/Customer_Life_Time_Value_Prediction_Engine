"""
LTV Segmentation and CAC Recommendation Engine.

Assigns customers to LTV segments based on predicted 36m LTV
and computes recommended max Customer Acquisition Cost (CAC).

Segment thresholds from the project plan:
  Champions:    LTV_36m > $10,000  → max CAC = 50% of LTV
  High Value:   LTV_36m > $5,000   → max CAC = 40% of LTV
  Medium Value: LTV_36m > $1,000   → max CAC = 30% of LTV
  Low Value:    LTV_36m ≤ $1,000   → max CAC = 20% of LTV
"""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl
from loguru import logger


# ─────────────────────────────────────────────────────────────
# Segment definitions
# ─────────────────────────────────────────────────────────────

SEGMENT_CONFIG = {
    "champions":    {"min_ltv": 10_000, "max_cac_pct": 0.50},
    "high_value":   {"min_ltv":  5_000, "max_cac_pct": 0.40},
    "medium_value": {"min_ltv":  1_000, "max_cac_pct": 0.30},
    "low_value":    {"min_ltv":      0, "max_cac_pct": 0.20},
}


def assign_segment(ltv_36m: float) -> str:
    """Return segment label for a single LTV_36m value."""
    if ltv_36m > 10_000: return "champions"
    if ltv_36m > 5_000:  return "high_value"
    if ltv_36m > 1_000:  return "medium_value"
    return "low_value"


def compute_max_cac(ltv_36m: float) -> float:
    """Return recommended max CAC for a given LTV_36m."""
    seg = assign_segment(ltv_36m)
    return ltv_36m * SEGMENT_CONFIG[seg]["max_cac_pct"]


def assign_segments_batch(predictions: pl.DataFrame) -> pl.DataFrame:
    """
    Add segment, recommended_max_cac, and ltv_percentile columns to predictions DataFrame.

    Args:
        predictions: Polars DataFrame with ltv_36m column

    Returns:
        DataFrame with added columns
    """
    ltv_vals = predictions["ltv_36m"].to_numpy()

    # Percentile rank
    n = len(ltv_vals)
    ranks = np.argsort(np.argsort(ltv_vals))
    percentiles = np.round(ranks / max(n - 1, 1) * 100).astype(int)

    # Segment assignment
    segments = pl.Series([assign_segment(float(v)) for v in ltv_vals])

    # Max CAC
    max_cac = pl.Series([compute_max_cac(float(v)) for v in ltv_vals])

    result = predictions.with_columns([
        segments.alias("segment"),
        max_cac.alias("recommended_max_cac"),
        pl.Series(percentiles.tolist()).alias("ltv_percentile"),
    ])

    logger.info(
        "Segments assigned — champions:{} high:{} medium:{} low:{}",
        (segments == "champions").sum(),
        (segments == "high_value").sum(),
        (segments == "medium_value").sum(),
        (segments == "low_value").sum(),
    )
    return result


def compute_segment_boundaries(
    predictions: pl.DataFrame,
    model_version: str,
) -> dict:
    """
    Compute and return LTV distribution statistics for the segment boundaries table.
    """
    ltv = predictions["ltv_36m"].drop_nulls()
    return {
        "model_version":   model_version,
        "champions_min":   10_000.0,
        "high_value_min":  5_000.0,
        "medium_value_min":1_000.0,
        "low_value_min":   0.0,
        "p25":             float(ltv.quantile(0.25)),
        "p50":             float(ltv.quantile(0.50)),
        "p75":             float(ltv.quantile(0.75)),
        "p90":             float(ltv.quantile(0.90)),
        "p99":             float(ltv.quantile(0.99)),
        "mean_ltv":        float(ltv.mean()),
    }