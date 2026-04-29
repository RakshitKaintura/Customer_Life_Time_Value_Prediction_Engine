"""
Causal DAG Definition — DoWhy integration.

Defines the causal graph for the customer LTV journey.
Used to:
  1. Document our causal assumptions
  2. Test d-separation / backdoor criteria
  3. Identify valid adjustment sets

DAG structure (simplified):
  acquisition_channel → onboarding_completed → LTV
  plan_tier           → onboarding_completed → LTV
  onboarding_completed → fast_repeat_buyer   → LTV
  rfm_features (X)    → LTV  (confounders)
  rfm_features (X)    → treatments           (confounders)
"""

from __future__ import annotations

from typing import Any

from loguru import logger

try:
    import dowhy
    from dowhy import CausalModel
    DOWHY_AVAILABLE = True
except ImportError:
    DOWHY_AVAILABLE = False
    logger.warning("dowhy not installed. Run: pip install dowhy")


# ─────────────────────────────────────────────────────────────
# DAG definition
# ─────────────────────────────────────────────────────────────

CAUSAL_DAG_GML = """
graph [
  node [id "acquisition_channel"     label "acquisition_channel"]
  node [id "plan_tier"               label "plan_tier"]
  node [id "frequency"               label "frequency"]
  node [id "monetary_avg"            label "monetary_avg"]
  node [id "recency_days"            label "recency_days"]
  node [id "t_days"                  label "t_days"]
  node [id "unique_categories"       label "unique_categories"]
  node [id "onboarding_completed"    label "onboarding_completed"]
  node [id "high_value_first_purchase" label "high_value_first_purchase"]
  node [id "fast_repeat_buyer"       label "fast_repeat_buyer"]
  node [id "multi_category_buyer"    label "multi_category_buyer"]
  node [id "high_frequency"          label "high_frequency"]
  node [id "ltv"                     label "ltv"]

  edge [source "acquisition_channel"     target "onboarding_completed"]
  edge [source "acquisition_channel"     target "ltv"]
  edge [source "plan_tier"               target "onboarding_completed"]
  edge [source "plan_tier"               target "ltv"]
  edge [source "frequency"               target "ltv"]
  edge [source "frequency"               target "high_frequency"]
  edge [source "monetary_avg"            target "ltv"]
  edge [source "monetary_avg"            target "high_value_first_purchase"]
  edge [source "recency_days"            target "ltv"]
  edge [source "t_days"                  target "ltv"]
  edge [source "unique_categories"       target "multi_category_buyer"]
  edge [source "unique_categories"       target "ltv"]
  edge [source "onboarding_completed"    target "fast_repeat_buyer"]
  edge [source "onboarding_completed"    target "ltv"]
  edge [source "high_value_first_purchase" target "ltv"]
  edge [source "fast_repeat_buyer"       target "ltv"]
  edge [source "multi_category_buyer"    target "ltv"]
  edge [source "high_frequency"          target "ltv"]
]
"""

# Node metadata for the dashboard
DAG_NODE_METADATA = {
    "acquisition_channel":       {"type": "confounder",  "description": "How customer was acquired"},
    "plan_tier":                 {"type": "confounder",  "description": "Signup plan tier"},
    "frequency":                 {"type": "confounder",  "description": "Purchase frequency"},
    "monetary_avg":              {"type": "confounder",  "description": "Average order value"},
    "recency_days":              {"type": "confounder",  "description": "Days since last purchase"},
    "t_days":                    {"type": "confounder",  "description": "Customer age in days"},
    "unique_categories":         {"type": "confounder",  "description": "Product category diversity"},
    "onboarding_completed":      {"type": "treatment",   "description": "Completed onboarding flow"},
    "high_value_first_purchase": {"type": "treatment",   "description": "First purchase in top 40%"},
    "fast_repeat_buyer":         {"type": "treatment",   "description": "2nd purchase within 30 days"},
    "multi_category_buyer":      {"type": "treatment",   "description": "Bought from 3+ categories"},
    "high_frequency":            {"type": "treatment",   "description": "5+ purchases in obs window"},
    "ltv":                       {"type": "outcome",     "description": "Customer Lifetime Value"},
}


def build_dowhy_model(
    df: Any,  # pandas DataFrame
    treatment: str,
    outcome: str = "log_ltv",
    common_causes: list[str] | None = None,
) -> Any:
    """
    Build a DoWhy CausalModel for a given treatment.

    Args:
        df:            pandas DataFrame with treatment + outcome + controls
        treatment:     treatment column name (e.g. 't_onboarding_completed')
        outcome:       outcome column name
        common_causes: list of confounders (controls)

    Returns:
        DoWhy CausalModel (or None if dowhy not available)
    """
    if not DOWHY_AVAILABLE:
        logger.warning("DoWhy not available — skipping causal model building")
        return None

    if common_causes is None:
        from backend.ml.causal_model import CONTROL_FEATURES
        common_causes = [c for c in CONTROL_FEATURES if c in df.columns]

    try:
        model = CausalModel(
            data=df,
            treatment=treatment,
            outcome=outcome,
            common_causes=common_causes,
            graph=CAUSAL_DAG_GML,
        )
        logger.info(
            "DoWhy model built — treatment={}, outcome={}, n_confounders={}",
            treatment, outcome, len(common_causes),
        )
        return model
    except Exception as exc:
        logger.warning("DoWhy model build failed for {}: {}", treatment, exc)
        return None


def get_dag_records(model_version: str) -> tuple[list[dict], list[dict]]:
    """
    Return node and edge records for storing in causal_dag_nodes / causal_dag_edges.
    """
    nodes = [
        {
            "model_version": model_version,
            "node_name":     name,
            "node_type":     meta["type"],
            "description":   meta["description"],
        }
        for name, meta in DAG_NODE_METADATA.items()
    ]

    edges = [
        {"model_version": model_version, "from_node": "acquisition_channel", "to_node": "onboarding_completed"},
        {"model_version": model_version, "from_node": "acquisition_channel", "to_node": "ltv"},
        {"model_version": model_version, "from_node": "plan_tier",           "to_node": "onboarding_completed"},
        {"model_version": model_version, "from_node": "plan_tier",           "to_node": "ltv"},
        {"model_version": model_version, "from_node": "onboarding_completed","to_node": "fast_repeat_buyer"},
        {"model_version": model_version, "from_node": "onboarding_completed","to_node": "ltv"},
        {"model_version": model_version, "from_node": "fast_repeat_buyer",   "to_node": "ltv"},
        {"model_version": model_version, "from_node": "high_value_first_purchase","to_node": "ltv"},
        {"model_version": model_version, "from_node": "multi_category_buyer","to_node": "ltv"},
        {"model_version": model_version, "from_node": "high_frequency",      "to_node": "ltv"},
        {"model_version": model_version, "from_node": "frequency",           "to_node": "high_frequency"},
        {"model_version": model_version, "from_node": "frequency",           "to_node": "ltv"},
        {"model_version": model_version, "from_node": "monetary_avg",        "to_node": "ltv"},
        {"model_version": model_version, "from_node": "unique_categories",   "to_node": "multi_category_buyer"},
    ]

    return nodes, edges