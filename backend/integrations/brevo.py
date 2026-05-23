"""
Brevo (Sendinblue) Integration.

Sends transactional emails using template IDs per LTV segment.
"""

from __future__ import annotations

from typing import Any

import httpx
from loguru import logger

from backend.api.config import api_settings


class BrevoClient:
    """Async Brevo API client for transactional emails."""

    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key or api_settings.BREVO_API_KEY
        self.sender_email = api_settings.BREVO_SENDER_EMAIL
        self.sender_name = api_settings.BREVO_SENDER_NAME or "LTV Team"
        self.templates = {
            "champions": api_settings.BREVO_TEMPLATE_CHAMPIONS,
            "high_value": api_settings.BREVO_TEMPLATE_HIGH,
            "medium_value": api_settings.BREVO_TEMPLATE_MEDIUM,
            "low_value": api_settings.BREVO_TEMPLATE_LOW,
        }

    @property
    def _headers(self) -> dict[str, str]:
        return {
            "api-key": self.api_key,
            "Content-Type": "application/json",
        }

    async def send_segment_email(
        self,
        to_email: str,
        to_name: str | None,
        segment: str,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not self.api_key or not self.sender_email:
            logger.warning("Brevo not configured — skipping send")
            return {"status": "skipped", "reason": "missing_config"}

        template_id = self.templates.get(segment)
        if not template_id:
            logger.warning("Brevo template missing for segment {}", segment)
            return {"status": "skipped", "reason": "missing_template"}

        payload = {
            "sender": {"email": self.sender_email, "name": self.sender_name},
            "to": [{"email": to_email, "name": to_name or to_email}],
            "templateId": int(template_id),
            "params": params or {},
        }

        async with httpx.AsyncClient(timeout=20.0) as client:
            response = await client.post(
                "https://api.brevo.com/v3/smtp/email",
                headers=self._headers,
                json=payload,
            )
            try:
                response.raise_for_status()
            except httpx.HTTPStatusError:
                logger.error("Brevo send failed: {}", response.text)
                raise

        return {"status": "sent", "segment": segment}
