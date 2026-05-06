"""
FastAPI dependency injection.

All heavy objects (models, DB client, scorers) are loaded
once at startup and injected via FastAPI's dependency system.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Annotated, Protocol

from fastapi import Depends, HTTPException, Security, status
from fastapi.security import APIKeyHeader
from loguru import logger

from backend.api.config import api_settings
from backend.db.supabase_client import SupabaseClient

# ─────────────────────────────────────────────────────────────
# API Key Authentication
# ─────────────────────────────────────────────────────────────

API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(
    api_key: str | None = Security(API_KEY_HEADER),
) -> str:
    """Verify the X-API-Key header matches the configured secret."""
    if api_settings.ENVIRONMENT == "development":
        # Allow unauthenticated requests in development
        return api_key or "dev"

    if not api_key or api_key != api_settings.API_SECRET_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
            headers={"WWW-Authenticate": "APIKey"},
        )
    return api_key


# ─────────────────────────────────────────────────────────────
# Database client (singleton)
# ─────────────────────────────────────────────────────────────

_db_client: SupabaseClient | None = None


def get_db_client() -> SupabaseClient:
    global _db_client
    if _db_client is None:
        _db_client = SupabaseClient(use_service_role=True)
    return _db_client


# ─────────────────────────────────────────────────────────────
# Scoring engine (singleton, loaded at startup)
# ─────────────────────────────────────────────────────────────

class ScoringEngineProtocol(Protocol):
    def score(self, customer_id: str, return_components: bool = False) -> dict: ...

    def score_batch(self, customer_ids: list[str]) -> list[dict]: ...


class ColdStartScorerProtocol(Protocol):
    def score(self, vertical: str, company_size: str, channel: str, plan_tier: str) -> dict: ...


_scoring_engine: ScoringEngineProtocol | None = None


def get_scoring_engine() -> ScoringEngineProtocol:
    """Return the global scoring engine (loaded at startup)."""
    if _scoring_engine is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Scoring engine not yet loaded. Try again in a moment.",
        )
    return _scoring_engine


def set_scoring_engine(engine: ScoringEngineProtocol) -> None:
    global _scoring_engine
    _scoring_engine = engine
    logger.info("Scoring engine registered in dependency injection")


# ─────────────────────────────────────────────────────────────
# Cold-start scorer (singleton)
# ─────────────────────────────────────────────────────────────

_cold_start_scorer: ColdStartScorerProtocol | None = None


def get_cold_start_scorer() -> ColdStartScorerProtocol:
    if _cold_start_scorer is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Cold-start scorer not ready.",
        )
    return _cold_start_scorer


def set_cold_start_scorer(scorer: ColdStartScorerProtocol) -> None:
    global _cold_start_scorer
    _cold_start_scorer = scorer
    logger.info("Cold-start scorer registered")


# Type aliases for cleaner endpoint signatures
DBClient    = Annotated[SupabaseClient, Depends(get_db_client)]
AuthKey     = Annotated[str, Depends(verify_api_key)]
ScoringEng  = Annotated[ScoringEngineProtocol, Depends(get_scoring_engine)]
ColdStartSc = Annotated[ColdStartScorerProtocol, Depends(get_cold_start_scorer)]