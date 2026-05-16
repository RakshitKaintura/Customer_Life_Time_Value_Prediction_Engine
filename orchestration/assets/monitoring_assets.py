"""
Monitoring Assets — drift detection, rolling validation, Arize Phoenix sync.
"""

from __future__ import annotations

from datetime import date, datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import polars as pl
from dagster import (
    AssetExecutionContext,
    AssetIn,
    Output,
    asset,
    MetadataValue,
)


# ─────────────────────────────────────────────────────────────
# LTV distribution drift check
# ─────────────────────────────────────────────────────────────

@asset(
    group_name="monitoring",
    description="Detect LTV distribution drift vs baseline (PSI + KS test)",
    compute_kind="scipy",
    ins={"fusion_model": AssetIn()},
)
def ltv_drift_check(
    context,
    fusion_model: dict,
) -> Output[dict]:
    """Run PSI and KS drift tests on current vs baseline LTV distribution."""
    from backend.monitoring.drift import DriftDetector
    from backend.db.supabase_client import SupabaseClient

    context.log.info("Running LTV distribution drift check")

    db     = SupabaseClient(use_service_role=True)
    result = {"alerts": [], "drift_detected": False}

    try:
        detector = DriftDetector(db_client=db)
        drift_results = detector.run_full_drift_check(
            model_version   = fusion_model.get("model_version", "fusion_v1"),
            baseline_days   = 60,
            monitoring_days = 30,
        )
        result = drift_results

        n_alerts = len([a for a in drift_results.get("alerts", []) if a.get("threshold_exceeded")])
        context.log.info(
            f"Drift check complete — {n_alerts} alerts, "
            f"drift_detected={drift_results.get('drift_detected', False)}"
        )
    except Exception as exc:
        context.log.warning(f"Drift check failed: {exc}")

    return Output(
        result,
        metadata={
            "drift_detected": MetadataValue.bool(result.get("drift_detected", False)),
            "n_alerts":       MetadataValue.int(len(result.get("alerts", []))),
            "psi_score":      MetadataValue.float(float(result.get("psi_score", 0) or 0)),
        },
    )


# ─────────────────────────────────────────────────────────────
# Rolling validation
# ─────────────────────────────────────────────────────────────

@asset(
    group_name="monitoring",
    description="Validate 12m predictions as cohorts mature",
    compute_kind="python",
)
def rolling_validation(context) -> Output[dict]:
    """
    As each cohort's 12m holdout window elapses,
    compare predicted vs actual LTV and log to model_performance_history.
    """
    from backend.monitoring.rolling_validator import RollingValidator
    from backend.db.supabase_client import SupabaseClient

    context.log.info("Running rolling validation")

    db      = SupabaseClient(use_service_role=True)
    result  = {}

    try:
        validator = RollingValidator(db_client=db)
        result    = validator.run()
        context.log.info(
            f"Rolling validation: MAE={result.get('mae_ltv_12m', 0):.2f}, "
            f"Gini={result.get('gini_coefficient', 0):.4f}"
        )
    except Exception as exc:
        context.log.warning(f"Rolling validation failed: {exc}")

    return Output(
        result,
        metadata={
            "mae_ltv_12m":      MetadataValue.float(float(result.get("mae_ltv_12m", 0) or 0)),
            "gini_coefficient": MetadataValue.float(float(result.get("gini_coefficient", 0) or 0)),
            "n_customers":      MetadataValue.int(int(result.get("n_customers", 0) or 0)),
        },
    )


# ─────────────────────────────────────────────────────────────
# Arize Phoenix sync
# ─────────────────────────────────────────────────────────────

@asset(
    group_name="monitoring",
    description="Sync LTV predictions to Arize Phoenix for observability",
    compute_kind="arize",
    ins={"fusion_model": AssetIn()},
)
def arize_phoenix_sync(
    context,
    fusion_model: dict,
) -> Output[dict]:
    """Send predictions to Arize Phoenix for drift monitoring."""
    from backend.monitoring.arize_integration import ArizePhoenixClient
    from backend.db.supabase_client import SupabaseClient

    context.log.info("Syncing to Arize Phoenix")

    db     = SupabaseClient(use_service_role=True)
    result = {"status": "skipped", "n_records": 0}

    try:
        client = ArizePhoenixClient()
        result = client.sync_predictions(
            db_client     = db,
            model_version = fusion_model.get("model_version", "fusion_v1"),
            limit         = 5000,
        )
        context.log.info(f"Arize sync: {result.get('n_records', 0)} records sent")
    except Exception as exc:
        context.log.warning(f"Arize sync failed (non-critical): {exc}")
        result = {"status": "failed", "error": str(exc)}

    return Output(
        result,
        metadata={
            "status":    MetadataValue.text(result.get("status", "unknown")),
            "n_records": MetadataValue.int(int(result.get("n_records", 0) or 0)),
        },
    )