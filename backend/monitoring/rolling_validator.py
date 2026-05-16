"""
Rolling Validation.

As cohorts' 12-month windows elapse, compare predicted LTV
(from final_ltv_scores) against actual revenue (from transactions).

This is the ground-truth accuracy check — the only way to know
if the model's 12m predictions were actually correct.

Results are stored in model_performance_history.
"""

from __future__ import annotations

from datetime import date, datetime, timezone, timedelta
from typing import Any

import numpy as np
from loguru import logger

from backend.ml.bgnbd_model import (
    compute_gini,
    compute_top_decile_lift,
    compute_calibration_error,
)


class RollingValidator:
    """
    Rolling validation engine.

    For each customer whose first purchase was ≥ 13 months ago,
    compare their predicted LTV_12m vs actual 12m revenue.
    """

    def __init__(self, db_client: Any) -> None:
        self.db = db_client

    def run(
        self,
        min_months_since_first: int = 13,
        model_version: str | None = None,
    ) -> dict[str, float]:
        """
        Run rolling validation and persist metrics.

        Returns dict with MAE, RMSE, Gini, top_decile_lift, calibration_error.
        """
        logger.info("Running rolling validation (min_months={})", min_months_since_first)

        # Load customers where first purchase was ≥ 13 months ago
        rows = self.db.execute_sql(
            """
            SELECT
                f.customer_id,
                f.ltv_12m                   AS predicted_ltv_12m,
                COALESCE(
                    (
                        SELECT SUM(t.quantity * t.unit_price)
                        FROM transactions t
                        WHERE t.customer_id = f.customer_id
                          AND t.invoice_date BETWEEN c.first_purchase_date
                              AND c.first_purchase_date + INTERVAL '12 months'
                    ), 0
                )                           AS actual_ltv_12m
            FROM final_ltv_scores f
            JOIN customers c USING (customer_id)
            WHERE c.first_purchase_date <= NOW() - INTERVAL '13 months'
              AND f.ltv_source = 'full_model'
              AND f.ltv_12m IS NOT NULL
            LIMIT 5000
            """
        )

        if not rows:
            logger.warning("No customers available for rolling validation")
            return {}

        y_pred = np.array([float(r["predicted_ltv_12m"] or 0) for r in rows])
        y_true = np.array([float(r["actual_ltv_12m"] or 0) for r in rows])

        mean_ltv = float(y_true.mean()) if y_true.mean() > 0 else 1.0

        metrics = {
            "mae_ltv_12m":       float(np.mean(np.abs(y_true - y_pred))),
            "rmse_ltv_12m":      float(np.sqrt(np.mean((y_true - y_pred) ** 2))),
            "mae_pct":           float(np.mean(np.abs(y_true - y_pred)) / mean_ltv),
            "gini_coefficient":  compute_gini(y_true, y_pred),
            "top_decile_lift":   compute_top_decile_lift(y_true, y_pred),
            "calibration_error": compute_calibration_error(y_true, y_pred),
            "n_customers":       len(rows),
            "mean_actual_ltv":   mean_ltv,
        }

        logger.info(
            "Rolling validation: MAE=£{:.2f} ({:.1f}%), "
            "Gini={:.4f}, Lift={:.2f}×, n={}",
            metrics["mae_ltv_12m"],
            metrics["mae_pct"] * 100,
            metrics["gini_coefficient"],
            metrics["top_decile_lift"],
            metrics["n_customers"],
        )

        # Persist to model_performance_history
        try:
            self.db.bulk_upsert(
                "model_performance_history",
                [{
                    "evaluated_at":     datetime.now(timezone.utc).isoformat(),
                    "model_version":    model_version or "unknown",
                    "evaluation_type":  "rolling_validation",
                    "period_start":     str(date.today() - timedelta(days=390)),
                    "period_end":       str(date.today()),
                    "n_customers":      metrics["n_customers"],
                    **{k: float(v) for k, v in metrics.items()
                       if k not in ("n_customers", "mean_actual_ltv")},
                }],
                conflict_columns=None,
            )
        except Exception as exc:
            logger.warning("Failed to persist rolling validation: {}", exc)

        return metrics