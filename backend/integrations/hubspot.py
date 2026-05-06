"""
HubSpot CRM Integration.

Updates contact properties with LTV predictions:
  - ltv_score          (numeric)
  - ltv_segment        (string: champions | high_value | medium_value | low_value)
  - recommended_max_cac (numeric)
  - ltv_source         (string: full_model | firmographic_prior)

Also triggers high-value lead workflows:
  - If segment == 'champions' or 'high_value' → assign to senior AE queue
"""

from __future__ import annotations

import asyncio
from typing import Any

import httpx
from loguru import logger

from backend.api.config import api_settings


HUBSPOT_BASE_URL = "https://api.hubapi.com"


class HubSpotClient:
    """Async HubSpot API client."""

    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key or api_settings.HUBSPOT_API_KEY
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type":  "application/json",
        }

    async def update_contact_ltv(
        self,
        contact_id: str,
        ltv_36m:    float,
        segment:    str,
        max_cac:    float,
        ltv_source: str = "full_model",
    ) -> dict[str, Any]:
        """
        Update a HubSpot contact with LTV prediction properties.

        HubSpot requires custom properties to be pre-created in the portal.
        Property names used:
            ltv_score_36m, ltv_segment, recommended_max_cac, ltv_source
        """
        if not self.api_key:
            logger.warning("HubSpot API key not configured — skipping update")
            return {"status": "skipped", "reason": "no_api_key"}

        payload = {
            "properties": {
                "ltv_score_36m":      str(round(ltv_36m, 2)),
                "ltv_segment":        segment,
                "recommended_max_cac": str(round(max_cac, 2)),
                "ltv_source":         ltv_source,
            }
        }

        url = f"{HUBSPOT_BASE_URL}/crm/v3/objects/contacts/{contact_id}"

        async with httpx.AsyncClient(timeout=10.0) as client:
            try:
                response = await client.patch(url, json=payload, headers=self.headers)
                response.raise_for_status()
                logger.info(
                    "HubSpot contact {} updated — segment={} ltv_36m={}",
                    contact_id, segment, ltv_36m,
                )
                return response.json()
            except httpx.HTTPStatusError as exc:
                logger.error(
                    "HubSpot update failed for contact {}: {} {}",
                    contact_id, exc.response.status_code, exc.response.text,
                )
                raise
            except Exception as exc:
                logger.error("HubSpot update error: {}", exc)
                raise

    async def trigger_high_value_workflow(
        self,
        contact_id: str,
        workflow_id: str = "high_value_lead_routing",
    ) -> dict:
        """
        Trigger a HubSpot workflow for high-value leads.
        Typically used to assign champions/high_value to senior AEs.
        """
        if not self.api_key:
            return {"status": "skipped"}

        url = f"{HUBSPOT_BASE_URL}/automation/v3/workflows/{workflow_id}/enrollments/contacts/{contact_id}"
        async with httpx.AsyncClient(timeout=10.0) as client:
            try:
                response = await client.post(url, headers=self.headers)
                logger.info("Enrolled contact {} in workflow {}", contact_id, workflow_id)
                return {"status": "enrolled", "workflow_id": workflow_id}
            except Exception as exc:
                logger.warning("Workflow enroll failed: {}", exc)
                return {"status": "failed", "error": str(exc)}

    async def get_contact(self, contact_id: str) -> dict:
        """Fetch a HubSpot contact by ID."""
        url = f"{HUBSPOT_BASE_URL}/crm/v3/objects/contacts/{contact_id}"
        params = {"properties": "email,firstname,lastname,company,industry,plan_tier"}
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(url, headers=self.headers, params=params)
            response.raise_for_status()
            return response.json()