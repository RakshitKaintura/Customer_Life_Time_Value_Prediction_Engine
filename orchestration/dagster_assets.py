"""
Main Dagster definitions file.

Registers all assets, resources, schedules, and sensors
into a single Definitions object consumed by dagster dev / dagster-daemon.
"""

from __future__ import annotations

from dagster import (
    Definitions,
    EnvVar,
    load_assets_from_modules,
)

from orchestration.assets import data_assets, model_assets, monitoring_assets
from orchestration.schedules import (
    daily_data_refresh_schedule,
    monthly_retraining_schedule,
    weekly_drift_check_schedule,
)
from orchestration.resources import (
    ModelStorageResource,
    SupabaseResource,
    WandbResource,
)
from orchestration.sensors import drift_alert_sensor, retraining_trigger_sensor

# ── Load all assets ───────────────────────────────────────────

all_assets = load_assets_from_modules([
    data_assets,
    model_assets,
    monitoring_assets,
])

# ── Resources ─────────────────────────────────────────────────

resources = {
    "supabase": SupabaseResource(
        database_url     = "",
        service_role_key = "",
        supabase_url     = "",
    ),
    "wandb": WandbResource(
        api_key  = "",
        project  = "ltv-prediction",
        entity   = "",
    ),
    "model_storage": ModelStorageResource(
        models_dir = "./models",
    ),
}

# ── Definitions ───────────────────────────────────────────────

defs = Definitions(
    assets    = all_assets,
    resources = resources,
    schedules = [
        monthly_retraining_schedule,
        weekly_drift_check_schedule,
        daily_data_refresh_schedule,
    ],
    sensors   = [
        drift_alert_sensor,
        retraining_trigger_sensor,
    ],
)