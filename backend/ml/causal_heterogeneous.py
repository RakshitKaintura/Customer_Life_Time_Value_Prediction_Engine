"""
Heterogeneous Treatment Effects (CATE) Analysis.

Segments customers by CATE magnitude to identify:
  1. Who benefits MOST from each intervention
  2. Which customer profiles are "high-leverage" targets
  3. CATE visualisation data for the dashboard

Also implements the "causal segmentation" concept:
  - Beyond RFM segments, define customer segments by
    which causal lever drives the most uplift for them.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import polars as pl
from loguru import logger


def compute_cate_segments(
    cate_df: pl.DataFrame,
    treatment_name: str,
    n_segments: int = 4,
) -> pl.DataFrame:
    """
    Assign customers to CATE quartiles for a given treatment.

    Returns DataFrame with:
        customer_id | treatment_name | cate_estimate | cate_segment (1=low, 4=high)
    """
    filtered = cate_df.filter(pl.col("treatment_name") == treatment_name)
    if len(filtered) == 0:
        return filtered

    filtered = filtered.with_columns(
        pl.col("cate_estimate")
            .qcut(n_segments, labels=[str(i) for i in range(1, n_segments + 1)])
            .cast(pl.Int32)
            .alias("cate_segment")
    )
    return filtered


def compute_cate_by_rfm_segment(
    cate_df: pl.DataFrame,
    rfm_df: pl.DataFrame,
    treatment_name: str,
) -> pl.DataFrame:
    """
    Average CATE per RFM segment for a given treatment.
    Used in the Causal Insights dashboard page.
    """
    # Join CATE with RFM cohort month as proxy segment
    joined = (
        cate_df.filter(pl.col("treatment_name") == treatment_name)
        .join(
            rfm_df.select(["customer_id", "cohort_month", "frequency",
                           "monetary_avg", "actual_ltv_12m"]),
            on="customer_id",
            how="left",
        )
    )

    # Assign RFM quartile
    joined = joined.with_columns(
        pl.col("monetary_avg")
            .qcut(4, labels=["Q1", "Q2", "Q3", "Q4"])
            .alias("monetary_quartile")
    )

    summary = (
        joined.group_by("monetary_quartile")
        .agg([
            pl.col("cate_estimate").mean().alias("mean_cate"),
            pl.col("cate_estimate").std().alias("std_cate"),
            pl.col("cate_estimate").median().alias("median_cate"),
            pl.len().alias("n_customers"),
            pl.col("actual_ltv_12m").mean().alias("mean_actual_ltv"),
        ])
        .sort("monetary_quartile")
    )
    return summary


def compute_heterogeneity_report(
    cate_results: dict[str, np.ndarray],
    rfm_df: pl.DataFrame,
) -> pl.DataFrame:
    """
    Build a comprehensive heterogeneity report across all treatments.

    For each treatment:
      - ATE (mean CATE)
      - Who benefits most (top 25% of CATE)
      - Heterogeneity index (std / |mean|)

    Returns a Polars DataFrame for the dashboard.
    """
    rows = []
    for treatment_name, cate in cate_results.items():
        n = len(cate)
        top_25_mask = cate >= np.percentile(cate, 75)

        rows.append({
            "treatment_name":          treatment_name,
            "ate":                     float(np.mean(cate)),
            "ate_std":                 float(np.std(cate)),
            "ate_median":              float(np.median(cate)),
            "ate_p25":                 float(np.percentile(cate, 25)),
            "ate_p75":                 float(np.percentile(cate, 75)),
            "pct_positive_cate":       float(np.mean(cate > 0) * 100),
            "mean_cate_top25":         float(np.mean(cate[top_25_mask])),
            "heterogeneity_index":     float(np.std(cate) / max(abs(np.mean(cate)), 1e-6)),
            "n_customers":             n,
        })

    return pl.DataFrame(rows).sort("ate", descending=True)


def find_high_leverage_customers(
    cate_results: dict[str, np.ndarray],
    customer_ids: list[str],
    min_total_uplift: float = 500.0,
    top_n: int = 100,
) -> pl.DataFrame:
    """
    Find customers with high total potential uplift across all treatments.

    These are the customers product and marketing should focus on.
    """
    cate_matrix = np.column_stack(list(cate_results.values()))  # (n, k)
    positive_cate = np.clip(cate_matrix, 0, None)               # only positive effects
    total_uplift  = positive_cate.sum(axis=1)

    df = pl.DataFrame({
        "customer_id":   customer_ids,
        "total_uplift":  total_uplift.tolist(),
        **{
            f"cate_{t}": cate_results[t].tolist()
            for t in cate_results
        },
    })

    return (
        df.filter(pl.col("total_uplift") >= min_total_uplift)
        .sort("total_uplift", descending=True)
        .head(top_n)
    )