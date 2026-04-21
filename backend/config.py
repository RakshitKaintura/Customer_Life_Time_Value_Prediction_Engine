"""
Centralised settings loaded from .env via pydantic-settings.
Import `settings` throughout the project — never read os.environ directly.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# ── Paths ────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Supabase ─────────────────────────────────────────────
    SUPABASE_URL: str = Field(default="", description="Supabase project URL")
    SUPABASE_ANON_KEY: str = Field(default="", description="Supabase anon key")
    SUPABASE_SERVICE_ROLE_KEY: str = Field(default="", description="Service role key")
    DATABASE_URL: str = Field(default="", description="Sync Postgres URL")
    DATABASE_URL_ASYNC: str = Field(default="", description="Async Postgres URL (asyncpg)")

    # ── Weights & Biases ─────────────────────────────────────
    WANDB_API_KEY: str = Field(default="")
    WANDB_PROJECT: str = Field(default="ltv-prediction")
    WANDB_ENTITY: str = Field(default="")

    # ── Paths ────────────────────────────────────────────────
    DATA_DIR: Path = Field(default=ROOT_DIR / "backend/data/raw")
    PROCESSED_DIR: Path = Field(default=ROOT_DIR / "backend/data/processed")
    MODELS_DIR: Path = Field(default=ROOT_DIR / "models")
    UCI_CSV_PATH: Path = Field(default=ROOT_DIR / "backend/data/raw/OnlineRetail.csv")

    # ── Pipeline Config ──────────────────────────────────────
    OBSERVATION_WINDOW_MONTHS: int = Field(default=6)
    HOLDOUT_WINDOW_MONTHS: int = Field(default=6)
    MAX_SEQUENCE_LENGTH: int = Field(default=50)
    AMOUNT_BUCKET_COUNT: int = Field(default=5)

    # ── API ──────────────────────────────────────────────────
    API_SECRET_KEY: str = Field(default="change-me")
    API_RATE_LIMIT_PER_MINUTE: int = Field(default=100)

    # ── Environment ──────────────────────────────────────────
    ENVIRONMENT: str = Field(default="development")
    LOG_LEVEL: str = Field(default="INFO")

    @field_validator("DATA_DIR", "PROCESSED_DIR", "MODELS_DIR", "UCI_CSV_PATH", mode="before")
    @classmethod
    def resolve_paths(cls, v: str | Path) -> Path:
        path = Path(v)
        # If the path is relative, resolve it against the project root
        if not path.is_absolute():
            path = ROOT_DIR / path
        
        # For directory fields, ensure they exist
        # (We check the name to avoid creating a directory for the CSV file itself)
        # Using a simple check: if the path doesn't have a suffix, it's a directory
        if not path.suffix:
            path.mkdir(parents=True, exist_ok=True)
            
        return path


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


settings = get_settings()