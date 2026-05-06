"""
Meta Ads Integration.

Uploads high-LTV lookalike seed audiences powered by pgvector embeddings.
Suppresses low-LTV segments from acquisition campaigns.

Requires: facebook-business SDK (optional)
    pip install facebook-business
"""

from __future__ import annotations

from typing import Any

from loguru import logger

from backend.api.config import api_settings


class MetaAdsClient:
    """
    Meta Marketing API client for audience management.
    """

    def __init__(
        self,
        access_token:  str | None = None,
        ad_account_id: str | None = None,
    ) -> None:
        self.access_token  = access_token  or api_settings.META_ACCESS_TOKEN
        self.ad_account_id = ad_account_id
        self.base_url = "https://graph.facebook.com/v18.0"

    async def upload_custom_audience(
        self,
        customer_emails: list[str],
        audience_name:   str,
        description:     str = "LTV-based segment",
    ) -> dict:
        """
        Upload a custom audience list to Meta Ads.
        Hashes emails with SHA-256 before upload (Meta requirement).
        """
        import hashlib

        if not self.access_token:
            logger.warning("Meta Ads access token not configured — skipping upload")
            return {"status": "skipped", "audience_name": audience_name}

        # Hash emails per Meta's requirements
        hashed_emails = [
            hashlib.sha256(email.lower().strip().encode()).hexdigest()
            for email in customer_emails
            if email
        ]

        logger.info(
            "Meta custom audience upload: {} hashed emails → '{}'",
            len(hashed_emails), audience_name,
        )

        try:
            import httpx
            async with httpx.AsyncClient(timeout=30.0) as client:
                # 1. Create audience
                create_resp = await client.post(
                    f"{self.base_url}/act_{self.ad_account_id}/customaudiences",
                    params={"access_token": self.access_token},
                    json={
                        "name":        audience_name,
                        "subtype":     "CUSTOM",
                        "description": description,
                        "customer_file_source": "USER_PROVIDED_ONLY",
                    },
                )
                create_resp.raise_for_status()
                audience_id = create_resp.json().get("id")

                # 2. Upload hashed emails
                if audience_id and hashed_emails:
                    upload_resp = await client.post(
                        f"{self.base_url}/{audience_id}/users",
                        params={"access_token": self.access_token},
                        json={
                            "payload": {
                                "schema": "EMAIL_SHA256",
                                "data":   hashed_emails[:10_000],  # Meta limit per call
                            }
                        },
                    )
                    upload_resp.raise_for_status()

                return {
                    "status":      "uploaded",
                    "audience_id": audience_id,
                    "n_emails":    len(hashed_emails),
                }
        except Exception as exc:
            logger.error("Meta Ads upload failed: {}", exc)
            return {"status": "failed", "error": str(exc)}

    async def create_lookalike_audience(
        self,
        source_audience_id: str,
        country:            str = "US",
        ratio:              float = 0.01,  # 1% lookalike
    ) -> dict:
        """
        Create a Lookalike Audience from a custom audience.
        This builds on the high-LTV seed audience created above.
        """
        if not self.access_token:
            return {"status": "skipped"}

        try:
            import httpx
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(
                    f"{self.base_url}/act_{self.ad_account_id}/customaudiences",
                    params={"access_token": self.access_token},
                    json={
                        "name":    f"LTV_Lookalike_{int(ratio*100)}pct_{country}",
                        "subtype": "LOOKALIKE",
                        "origin_audience_id": source_audience_id,
                        "lookalike_spec": {
                            "type":    "similarity",
                            "ratio":   ratio,
                            "country": country,
                        },
                    },
                )
                response.raise_for_status()
                data = response.json()
                logger.info("Meta lookalike audience created: {}", data.get("id"))
                return {"status": "created", "audience_id": data.get("id")}
        except Exception as exc:
            logger.error("Meta lookalike creation failed: {}", exc)
            return {"status": "failed", "error": str(exc)}