"""
/health and /model-performance endpoints.
"""

from __future__ import annotations

from fastapi import APIRouter
from loguru import logger

from backend.api.config import api_settings
from backend.api.dependencies import DBClient, get_scoring_engine
from backend.api.schemas import HealthResponse, ModelPerformanceResponse, SegmentStatsResponse

router = APIRouter(tags=["Health & Monitoring"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Service health check",
)
async def health_check(db: DBClient) -> HealthResponse:
    """Returns OK if service is running and DB is reachable."""
    db_ok = db.health_check()

    try:
        engine = get_scoring_engine()
        engine_ok = engine is not None
    except Exception:
        engine_ok = False

    try:
        models_ok = engine_ok
    except Exception:
        models_ok = False

    return HealthResponse(
        status         = "ok" if db_ok else "degraded",
        environment    = api_settings.ENVIRONMENT,
        models_loaded  = models_ok,
        db_connected   = db_ok,
        scoring_engine = engine_ok,
    )


@router.get(
    "/model-performance",
    response_model=ModelPerformanceResponse,
    summary="Current model MAE, calibration, and segment metrics",
)
async def model_performance(db: DBClient) -> ModelPerformanceResponse:
    """Return live model performance metrics from the DB."""
    # Fusion metrics
    fusion_rows = db.execute_sql(
        """
        SELECT model_version, mae_ltv_12m, gini_coefficient,
               top_decile_lift, calibration_error
        FROM fusion_model_registry
        ORDER BY trained_at DESC LIMIT 1
        """
    )
    fusion = fusion_rows[0] if fusion_rows else {}

    # BG/NBD metrics
    bgnbd_rows = db.execute_sql(
        """
        SELECT model_version, r2_frequency, mae_ltv_12m
        FROM bgnbd_model_params
        ORDER BY fitted_at DESC LIMIT 1
        """
    )
    bgnbd = bgnbd_rows[0] if bgnbd_rows else {}

    # Transformer metrics
    trans_rows = db.execute_sql(
        """
        SELECT model_version, mae_ltv_12m, gini_coefficient
        FROM transformer_model_registry
        ORDER BY trained_at DESC LIMIT 1
        """
    )
    trans = trans_rows[0] if trans_rows else {}

    # Segment distribution
    seg_rows = db.execute_sql(
        """
        SELECT segment, COUNT(*) AS n
        FROM final_ltv_scores
        GROUP BY segment
        """
    )
    seg_dist = {r["segment"]: int(r["n"]) for r in seg_rows}

    # Coverage
    cov_rows = db.execute_sql(
        "SELECT COUNT(*) AS n, MAX(scored_at) AS last FROM final_ltv_scores"
    )
    cov = cov_rows[0] if cov_rows else {}

    return ModelPerformanceResponse(
        fusion_model_version      = fusion.get("model_version"),
        bgnbd_model_version       = bgnbd.get("model_version"),
        transformer_model_version = trans.get("model_version"),
        fusion_mae_ltv_12m        = fusion.get("mae_ltv_12m"),
        fusion_gini               = fusion.get("gini_coefficient"),
        fusion_top_decile_lift    = fusion.get("top_decile_lift"),
        fusion_calibration_error  = fusion.get("calibration_error"),
        bgnbd_r2_frequency        = bgnbd.get("r2_frequency"),
        bgnbd_mae_ltv_12m         = bgnbd.get("mae_ltv_12m"),
        transformer_mae_ltv_12m   = trans.get("mae_ltv_12m"),
        transformer_gini          = trans.get("gini_coefficient"),
        segment_distribution      = seg_dist,
        n_customers_scored        = int(cov.get("n", 0)),
        last_scored_at            = cov.get("last"),
    )


@router.get(
    "/segment-stats",
    response_model=SegmentStatsResponse,
    summary="Live segment LTV and CAC statistics",
)
async def segment_stats(db: DBClient) -> SegmentStatsResponse:
    rows = db.execute_sql(
        """
        SELECT segment, pct_customers, avg_ltv_36m, avg_max_cac
        FROM v_segment_revenue_concentration
        ORDER BY avg_ltv_36m DESC
        """
    )

    normalized = []
    for row in rows:
        avg_ltv = float(row.get("avg_ltv_36m") or 0)
        avg_max_cac = float(row.get("avg_max_cac") or 0)
        pct_customers = float(row.get("pct_customers") or 0) / 100
        max_cac_pct = avg_max_cac / avg_ltv if avg_ltv > 0 else 0

        normalized.append(
            {
                "segment": str(row.get("segment") or "unknown"),
                "avg_ltv": avg_ltv,
                "avg_max_cac": avg_max_cac,
                "pct_customers": pct_customers,
                "max_cac_pct": max_cac_pct,
            }
        )

    return SegmentStatsResponse(data=normalized)