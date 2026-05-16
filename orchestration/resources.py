"""
Dagster resource definitions.

Resources are shared connections/clients injected into assets and ops.
Defined once here, used throughout the asset graph.
"""

from __future__ import annotations

from dagster import ConfigurableResource, EnvVar
from pydantic import Field


class SupabaseResource(ConfigurableResource):
    """Supabase database connection resource."""
    database_url:        str = Field(default_factory=lambda: "")
    service_role_key:    str = Field(default_factory=lambda: "")
    supabase_url:        str = Field(default_factory=lambda: "")

    def get_client(self):
        from backend.db.supabase_client import SupabaseClient
        return SupabaseClient(use_service_role=True)


class WandbResource(ConfigurableResource):
    """Weights & Biases tracking resource."""
    api_key:  str = Field(default_factory=lambda: "")
    project:  str = Field(default="ltv-prediction")
    entity:   str = Field(default_factory=lambda: "")

    def init_run(self, name: str, tags: list[str] | None = None, config: dict | None = None):
        import wandb
        return wandb.init(
            project = self.project,
            name    = name,
            tags    = tags or [],
            config  = config or {},
            reinit  = True,
        )


class ModelStorageResource(ConfigurableResource):
    """Local model storage resource."""
    models_dir: str = Field(default="./models")

    @property
    def path(self):
        from pathlib import Path
        p = Path(self.models_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p


def get_resources():
    """Return resource dict for Dagster definitions."""
    return {
        "supabase": SupabaseResource(
            database_url     = "",
            service_role_key = "",
            supabase_url     = "",
        ),
        "wandb": WandbResource(
            api_key = "",
            project = "ltv-prediction",
        ),
        "model_storage": ModelStorageResource(
            models_dir = "./models",
        ),
    }