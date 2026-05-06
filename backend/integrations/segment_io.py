"""
Segment.io Integration.

Sends LTV data back to Segment as:
  1. Identify call — updates user traits with ltv_segment, ltv_36m
  2. Track call    — fires ltv_scored event for analytics
"""

from __future__ import annotations

from typing import Any

import httpx
from loguru import logger

from backend.api.config import api_settings


SEGMENT_TRACK_URL    = "https://api.segment.io/v1/track"
SEGMENT_IDENTIFY_URL = "https://api.segment.io/v1/identify"


class SegmentClient:
    """Async Segment.io API client."""

    def __init__(self, write_key: str | None = None) -> None:
        self.write_key = write_key or api_settings.SEGMENT_WRITE_KEY
        # Segment uses HTTP Basic Auth: write_key as username, empty password
        self.auth = (self.write_key, "") if self.write_key else None

    async def identify(
        self,
        user_id:     str,
        ltv_36m:     float,
        segment:     str,
        ltv_source:  str = "full_model",
        max_cac:     float | None = None,
    ) -> dict:
        """Update Segment user traits with LTV prediction."""
        if not self.write_key:
            logger.warning("Segment write key not configured — skipping identify")
            return {"status": "skipped"}

        payload = {
            "userId": user_id,
            "traits": {
                "ltv_segment":  segment,
                "ltv_36m":      round(ltv_36m, 2),
                "ltv_source":   ltv_source,
                **({"recommended_max_cac": round(max_cac, 2)} if max_cac else {}),
            },
        }

        async with httpx.AsyncClient(timeout=10.0) as client:
            try:
                response = await client.post(
                    SEGMENT_IDENTIFY_URL,
                    json=payload,
                    auth=self.auth,
                )
                response.raise_for_status()
                logger.info("Segment identify sent for user {}", user_id)
                return {"status": "ok"}
            except Exception as exc:
                logger.error("Segment identify failed: {}", exc)
                return {"status": "failed", "error": str(exc)}

    async def track_ltv_scored(
        self,
        user_id:  str,
        ltv_36m:  float,
        segment:  str,
        properties: dict[str, Any] | None = None,
    ) -> dict:
        """Fire ltv_scored track event."""
        if not self.write_key:
            return {"status": "skipped"}

        payload = {
            "userId": user_id,
            "event":  "ltv_scored",
            "properties": {
                "ltv_36m":  round(ltv_36m, 2),
                "segment":  segment,
                **(properties or {}),
            },
        }

        async with httpx.AsyncClient(timeout=10.0) as client:
            try:
                response = await client.post(
                    SEGMENT_TRACK_URL,
                    json=payload,
                    auth=self.auth,
                )
                response.raise_for_status()
                logger.info("Segment ltv_scored event sent for user {}", user_id)
                return {"status": "ok"}
            except Exception as exc:
                logger.error("Segment track failed: {}", exc)
                return {"status": "failed", "error": str(exc)}