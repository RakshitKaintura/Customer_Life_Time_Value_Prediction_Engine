"""
Dagster sensors — event-driven triggers.

Sensors:
  - drift_alert_sensor:       Triggers retraining when PSI > 0.15
  - retraining_trigger_sensor: Monitors DB for manual retraining requests
"""

from __future__ import annotations

from dagster import (
    RunRequest,
    SensorEvaluationContext,
    SensorResult,
    sensor,
    define_asset_job,
    AssetSelection,
)


retraining_job = define_asset_job(
    name      = "retraining_job",
    selection = AssetSelection.groups("ml_models"),
    description = "Retrain all ML models (triggered by sensor)",
)


@sensor(
    job         = retraining_job,
    minimum_interval_seconds = 3600,   # check every hour
    description = "Trigger retraining when LTV distribution drift exceeds threshold",
)
def drift_alert_sensor(context: SensorEvaluationContext) -> SensorResult:
    """
    Check the ltv_drift_alerts table.
    If any open alert has PSI > 0.15, trigger full retraining.
    """
    try:
        from backend.db.supabase_client import SupabaseClient
        db = SupabaseClient(use_service_role=True)

        rows = db.execute_sql(
            """
            SELECT alert_id, psi_score, alert_type
            FROM ltv_drift_alerts
            WHERE status = 'open'
              AND psi_score > 0.15
              AND detected_at > NOW() - INTERVAL '48 hours'
            ORDER BY psi_score DESC
            LIMIT 1
            """
        )

        if rows:
            alert = rows[0]
            context.log.info(
                f"Drift alert detected: alert_id={alert['alert_id']} "
                f"PSI={alert['psi_score']:.4f}"
            )
            return SensorResult(
                run_requests=[
                    RunRequest(
                        run_key    = f"drift_retrain_{alert['alert_id']}",
                        run_config = {
                            "ops": {
                                "trigger_reason": "drift_alert",
                                "alert_id":       alert["alert_id"],
                            }
                        },
                    )
                ],
                cursor = str(alert["alert_id"]),
            )
    except Exception as exc:
        context.log.warning(f"Drift sensor check failed: {exc}")

    return SensorResult(run_requests=[])


@sensor(
    job         = retraining_job,
    minimum_interval_seconds = 1800,
    description = "Check for manual retraining requests in the database",
)
def retraining_trigger_sensor(context: SensorEvaluationContext) -> SensorResult:
    """
    Watches pipeline_runs table for manual retraining requests
    (e.g. inserted by the dashboard's 'Retrain Now' button).
    """
    try:
        from backend.db.supabase_client import SupabaseClient
        db = SupabaseClient(use_service_role=True)

        rows = db.execute_sql(
            """
            SELECT run_id, metadata
            FROM pipeline_runs
            WHERE pipeline_name = 'manual_retrain_request'
              AND status = 'pending'
              AND started_at > NOW() - INTERVAL '2 hours'
            LIMIT 1
            """
        )

        if rows:
            req = rows[0]
            # Mark as picked up
            db.execute_sql(
                "UPDATE pipeline_runs SET status = 'running' WHERE run_id = :rid",
                {"rid": req["run_id"]},
            )
            return SensorResult(
                run_requests=[
                    RunRequest(run_key=f"manual_{req['run_id']}")
                ]
            )
    except Exception as exc:
        context.log.warning(f"Retraining sensor check failed: {exc}")

    return SensorResult(run_requests=[])