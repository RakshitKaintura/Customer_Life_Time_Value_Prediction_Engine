"""
API-specific configuration and model loading.
Models are loaded once at startup and reused across requests.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

from loguru import logger
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_ENV_FILE = Path(__file__).resolve().parents[2] / ".env"


class APISettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(ROOT_ENV_FILE),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Model versions
    FUSION_MODEL_VERSION:      str = "fusion_v1"
    BGNBD_MODEL_VERSION:       str = "bgnbd_uci_v1"
    TRANSFORMER_MODEL_VERSION: str = "transformer_v1"
    CAUSAL_MODEL_VERSION:      str = "causal_v1"

    # Model paths
    MODELS_DIR:   Path = Path("./models")
    ONNX_PATH:    Path = Path("./models/transformer.onnx")

    # Supabase
    SUPABASE_URL:              str = ""
    SUPABASE_SERVICE_ROLE_KEY: str = ""
    DATABASE_URL:              str = ""

    # API security
    API_SECRET_KEY:            str = "change-me"
    API_RATE_LIMIT_PER_MINUTE: int = 100

    # Integrations
    SEGMENT_WRITE_KEY:    str = ""
    GOOGLE_ADS_DEVELOPER_TOKEN: str = ""
    META_ACCESS_TOKEN:    str = ""
    AIRTABLE_API_TOKEN:   str = ""
    AIRTABLE_BASE_ID:     str = ""
    AIRTABLE_TABLE_ID:    str = ""
    AIRTABLE_EMAIL_FIELD: str = "email"

    BREVO_API_KEY:        str = ""
    BREVO_SENDER_EMAIL:   str = ""
    BREVO_SENDER_NAME:    str = ""
    BREVO_TEMPLATE_CHAMPIONS: int | None = None
    BREVO_TEMPLATE_HIGH:      int | None = None
    BREVO_TEMPLATE_MEDIUM:    int | None = None
    BREVO_TEMPLATE_LOW:       int | None = None
    BREVO_DAILY_LIMIT:        int = 300

    # Environment
    ENVIRONMENT: str = "development"
    LOG_LEVEL:   str = "INFO"

    # Scoring
    MAX_SEQ_LEN: int = 50

    # Deployment toggles
    DISABLE_HEAVY_MODELS: bool = False

    @field_validator("ENVIRONMENT")
    @classmethod
    def normalize_environment(cls, value: str) -> str:
        # Strip inline comments like "development # ..."
        return value.split("#", 1)[0].strip()


@lru_cache(maxsize=1)
def get_api_settings() -> APISettings:
    return APISettings()


api_settings = get_api_settings()