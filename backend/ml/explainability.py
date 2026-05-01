"""
SHAP Explainability Module.

Provides per-customer SHAP explanations and global feature importance
for the XGBoost fusion model.

Also generates human-readable LTV driver descriptions
for the API response top_ltv_drivers field.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl
from loguru import logger


FEATURE_DESCRIPTIONS = {
    "bgnbd_ltv_12m":           "BG/NBD model's 12-month LTV prediction",
    "bgnbd_ltv_36m":           "BG/NBD model's 36-month LTV prediction",
    "transformer_ltv_12m":     "Transformer model's 12-month LTV prediction",
    "transformer_ltv_36m":     "Transformer model's 36-month LTV prediction",
    "probability_alive":       "Probability the customer is still active",
    "frequency":               "Number of repeat purchases in observation window",
    "monetary_avg":            "Average order value",
    "recency_days":            "Days from first to last purchase",
    "t_days":                  "Customer age in days",
    "purchase_variance":       "Variability in order values",
    "orders_count":            "Total number of orders placed",
    "avg_days_between_orders": "Average days between consecutive purchases",
    "unique_categories":       "Number of distinct product categories purchased",
}

FEATURE_THRESHOLDS = {
    "frequency": {
        "high":   5,
        "medium": 2,
        "description_high":   "High purchase frequency (5+ orders)",
        "description_medium": "Moderate purchase frequency",
        "description_low":    "Low purchase frequency (1 order)",
    },
    "monetary_avg": {
        "high":   200,
        "medium": 60,
        "description_high":   "High average order value (£200+)",
        "description_medium": "Moderate average order value",
        "description_low":    "Low average order value",
    },
    "probability_alive": {
        "high":   0.80,
        "medium": 0.50,
        "description_high":   "High probability of remaining active",
        "description_medium": "Moderate activity probability",
        "description_low":    "At risk of churning",
    },
}


def generate_driver_narratives(
    shap_contributions: list[dict],
    customer_meta: dict[str, float],
    top_n: int = 3,
) -> list[str]:
    """
    Convert SHAP contributions into human-readable driver narratives.

    Returns list of strings like:
        "High purchase frequency (8 orders) drives LTV up by £340"
    """
    narratives = []

    for item in shap_contributions[:top_n]:
        feature   = item["feature"]
        value     = item["value"]
        shap_val  = item["shap_contribution"]
        direction = "increases" if shap_val > 0 else "decreases"
        magnitude = abs(shap_val)

        # Build readable description
        desc = FEATURE_DESCRIPTIONS.get(feature, feature.replace("_", " ").title())

        if feature == "frequency":
            desc = f"{int(value)} repeat purchases"
        elif feature == "monetary_avg":
            desc = f"Average order value £{value:.0f}"
        elif feature == "probability_alive":
            desc = f"{100*value:.0f}% probability of remaining active"
        elif feature in ("bgnbd_ltv_12m", "transformer_ltv_12m"):
            desc = f"Model consensus on high LTV potential"
        elif feature == "unique_categories":
            desc = f"{int(value)} product categories purchased"
        elif feature == "recency_days":
            desc = f"Recent purchase ({int(value)} days since last order)"

        narrative = f"{desc} {direction} LTV by £{magnitude:.0f}"
        narratives.append(narrative)

    return narratives


def compute_global_shap_importance(
    fusion_model: Any,
    meta_features: pl.DataFrame,
    max_samples: int = 500,
) -> pl.DataFrame:
    """
    Compute global SHAP importance and return sorted DataFrame.
    Also generates bar chart data for the dashboard.
    """
    try:
        importance_df = fusion_model.get_global_feature_importance(
            meta_features, max_samples=max_samples
        )
        logger.info("SHAP global importance computed for {} features", len(importance_df))
        return importance_df
    except Exception as exc:
        logger.warning("SHAP computation failed: {}", exc)
        # Fallback: XGBoost native feature importance
        try:
            feat_imp = fusion_model.model_12m.feature_importances_
            feat_names = fusion_model._feature_names
            return pl.DataFrame({
                "feature_name":  feat_names,
                "mean_abs_shap": feat_imp.tolist(),
            }).sort("mean_abs_shap", descending=True).with_columns(
                pl.Series("rank", list(range(1, len(feat_names) + 1)))
            )
        except Exception:
            return pl.DataFrame()


def build_top_drivers_for_customer(
    customer_id: str,
    customer_meta: dict[str, float],
    fusion_model: Any,
    causal_levers: list[str] | None = None,
    top_n: int = 3,
) -> tuple[list[str], list[str]]:
    """
    Build top_ltv_drivers and causal_levers for a single customer.

    Returns:
        (top_drivers list, causal_levers list)
    """
    # SHAP drivers
    try:
        shap_contributions = fusion_model.get_customer_shap_explanation(
            customer_meta, top_n=top_n
        )
        top_drivers = generate_driver_narratives(
            shap_contributions, customer_meta, top_n=top_n
        )
    except Exception:
        top_drivers = []

    # Causal levers
    levers = causal_levers or []

    return top_drivers, levers