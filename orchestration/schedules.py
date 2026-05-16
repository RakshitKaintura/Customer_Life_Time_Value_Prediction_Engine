"""
Dagster schedule definitions.

Schedules:
  - monthly_retraining:   1st of every month at 02:00 UTC
  - weekly_drift_check:   Every Monday at 06:00 UTC
  - daily_validation:     Every day at 04:00 UTC
  - marketing_sync:       Every day at 08:00 UTC
"""

from dagster import (
    AssetSelection,
    DefaultScheduleStatus,
    ScheduleDefinition,
    define_asset_job,
)

# ── Job definitions ───────────────────────────────────────────

full_pipeline_job = define_asset_job(
    name      = "full_ltv_pipeline",
    selection = AssetSelection.groups(
        "data_engineering", "ml_models", "monitoring"
    ),
    description = "Full LTV pipeline: ingest → feature eng → train → monitor",
)

data_only_job = define_asset_job(
    name      = "data_engineering_job",
    selection = AssetSelection.groups("data_engineering"),
    description = "Data ingestion and feature engineering only",
)

monitoring_job = define_asset_job(
    name      = "monitoring_job",
    selection = AssetSelection.groups("monitoring"),
    description = "Drift detection and rolling validation",
)

# ── Schedule definitions ──────────────────────────────────────

monthly_retraining_schedule = ScheduleDefinition(
    name            = "monthly_retraining",
    cron_schedule   = "0 2 1 * *",   # 1st of month at 02:00 UTC
    job             = full_pipeline_job,
    default_status  = DefaultScheduleStatus.RUNNING,
    description     = "Monthly model retraining — full pipeline",
    execution_timezone = "UTC",
)

weekly_drift_check_schedule = ScheduleDefinition(
    name            = "weekly_drift_check",
    cron_schedule   = "0 6 * * 1",   # Every Monday at 06:00 UTC
    job             = monitoring_job,
    default_status  = DefaultScheduleStatus.RUNNING,
    description     = "Weekly drift detection and rolling validation",
    execution_timezone = "UTC",
)

daily_data_refresh_schedule = ScheduleDefinition(
    name            = "daily_data_refresh",
    cron_schedule   = "0 4 * * *",   # Every day at 04:00 UTC
    job             = data_only_job,
    default_status  = DefaultScheduleStatus.RUNNING,
    description     = "Daily feature engineering refresh",
    execution_timezone = "UTC",
)