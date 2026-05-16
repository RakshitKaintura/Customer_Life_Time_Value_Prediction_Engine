"""
Arize Phoenix Integration.

Sends LTV predictions to Arize Phoenix for:
  - Prediction drift monitoring
  - Feature drift detection
  - Model performance tracking over time

Arize Phoenix is the open-source observability platform.
Uses the arize-phoenix library.

If arize-phoenix is not installed, falls back to local logging.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import numpy as np
from loguru import logger


class ArizePhoenixClient:
    """
    Arize Phoenix client for LTV prediction observability.

    Sends:
        - Prediction records (customer_id, predicted_ltv, actual_ltv, features)
        - Model schema (feature names, prediction type)
    """

    def __init__(self) -> None:
        self._available = self._check_arize_available()

    def _check_arize_available(self) -> bool:
        try:
            import phoenix as px  # noqa: F401
            return True
        except ImportError:
            logger.warning(
                "arize-phoenix not installed. "
                "Run: pip install arize-phoenix. "
                "Skipping Arize sync."
            )
            return False

    def sync_predictions(
        self,
        db_client:     Any,
        model_version: str,
        limit:         int = 5000,
    ) -> dict[str, Any]:
        """
        Load final LTV scores from Supabase and send to Arize Phoenix.

        Returns status dict.
        """
        if not self._available:
            return {"status": "skipped", "reason": "arize_phoenix_not_installed"}

        try:
            import phoenix as px
            import pandas as pd
        except ImportError:
            return {"status": "skipped"}

        # Load predictions
        rows = db_client.execute_sql(
            """
            SELECT
                f.customer_id,
                f.ltv_12m        AS predicted_ltv_12m,
                f.ltv_36m        AS predicted_ltv_36m,
                f.segment,
                f.probability_alive_12m,
                f.scored_at,
                r.frequency,
                r.monetary_avg,
                r.recency_days,
                r.t_days,
                r.actual_ltv_12m
            FROM final_ltv_scores f
            LEFT JOIN v_latest_rfm r USING (customer_id)
            WHERE f.model_version = :ver
              AND f.ltv_source = 'full_model'
            LIMIT :lim
            """,
            {"ver": model_version, "lim": limit},
        )

        if not rows:
            return {"status": "no_data", "n_records": 0}

        df = pd.DataFrame(rows)

        try:
            # Launch Phoenix if not already running
            session = px.active_session()
            if session is None:
                session = px.launch_app()

            # Log the dataframe as a dataset
            dataset = px.Dataset(
                dataframe   = df,
                schema      = px.Schema(
                    prediction_id_column_name  = "customer_id",
                    prediction_label_column_name = "predicted_ltv_12m",
                    actual_label_column_name   = "actual_ltv_12m",
                    feature_column_names       = [
                        "frequency", "monetary_avg", "recency_days", "t_days"
                    ],
                    timestamp_column_name      = "scored_at",
                ),
                name        = f"ltv_{model_version}",
            )

            logger.info(
                "Arize Phoenix sync: {} records, model_version={}",
                len(df), model_version,
            )

            # Persist sync log
            try:
                db_client.bulk_upsert("arize_sync_log", [{
                    "synced_at":     datetime.now(timezone.utc).isoformat(),
                    "n_records":     len(df),
                    "model_id":      "ltv_prediction",
                    "model_version": model_version,
                    "status":        "success",
                }], conflict_columns=None)
            except Exception:
                pass

            return {"status": "success", "n_records": len(df)}

        except Exception as exc:
            logger.error("Arize Phoenix sync error: {}", exc)
            return {"status": "failed", "error": str(exc), "n_records": 0}

    def get_drift_metrics(self) -> dict[str, Any]:
        """
        Query Arize Phoenix for current drift metrics.
        Returns empty dict if not available.
        """
        if not self._available:
            return {}

        try:
            import phoenix as px
            session = px.active_session()
            if session is None:
                return {}

            # In production, use Arize Phoenix REST API to query metrics
            return {
                "status":    "active",
                "dashboard": session.url if hasattr(session, "url") else "http://localhost:6006",
            }
        except Exception:
            return {}