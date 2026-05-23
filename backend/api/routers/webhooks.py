"""
Webhook handlers:
    POST /webhook/airtable — Airtable contact → score + upsert
    POST /webhook/segment  — Segment.io identify → score + return traits
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, BackgroundTasks
from loguru import logger

from backend.api.dependencies import ColdStartSc
from backend.api.schemas import AirtableWebhookPayload, SegmentWebhookPayload, WebhookResponse

router = APIRouter(prefix="/webhook", tags=["Webhooks"])


# ─────────────────────────────────────────────────────────────
# Airtable webhook (simple JSON)
# ─────────────────────────────────────────────────────────────

@router.post(
    "/airtable",
    response_model=WebhookResponse,
    summary="Airtable contact → score and upsert",
)
async def airtable_webhook(
    payload: AirtableWebhookPayload,
    background: BackgroundTasks,
    cold: ColdStartSc,
) -> WebhookResponse:
    """
    Called by Airtable automation or manual webhook.
    Scores the contact using firmographic prior and upserts Airtable.
    """
    logger.info("Airtable webhook: contact_id={}", payload.contact_id)

    vertical = payload.vertical or "other"
    company_size = payload.company_size or "smb"
    channel = payload.channel or "organic"
    plan_tier = payload.plan_tier or "free"

    score = cold.score(
        vertical=vertical,
        company_size=company_size,
        channel=channel,
        plan_tier=plan_tier,
    )

    background.add_task(
        _update_airtable_contact,
        contact_id=payload.contact_id,
        score=score,
    )

    return WebhookResponse(
        status="scored",
        customer_id=payload.contact_id,
        ltv_36m=score.get("ltv_36m"),
        segment=score.get("segment"),
        message=f"Cold-start score applied. Segment: {score.get('segment')}",
    )


async def _update_airtable_contact(
    contact_id: str,
    score: dict[str, Any],
) -> None:
    """Update Airtable contact with LTV properties (background task)."""
    try:
        from backend.integrations.airtable import AirtableClient

        client = AirtableClient()
        await client.upsert_contacts([
            {
                "contact_id": str(contact_id),
                "ltv_score_36m": score.get("ltv_36m", 0),
                "ltv_segment": score.get("segment", "low_value"),
                "recommended_max_cac": score.get("recommended_max_cac", 0),
                "ltv_source": score.get("ltv_source", "firmographic_prior"),
            }
        ])
        logger.info("Airtable contact {} updated with LTV score", contact_id)
    except Exception as exc:
        logger.error("Failed to update Airtable contact {}: {}", contact_id, exc)


# ─────────────────────────────────────────────────────────────
# Segment.io webhook
# ─────────────────────────────────────────────────────────────

@router.post(
    "/segment",
    response_model=WebhookResponse,
    summary="Segment.io identify event → score and return LTV trait",
)
async def segment_webhook(
    payload:    SegmentWebhookPayload,
    background: BackgroundTasks,
    cold:       ColdStartSc,
) -> WebhookResponse:
    """
    Called when Segment.io fires an Identify event for a new user.
    Returns LTV as a user trait.
    """
    logger.info("Segment.io webhook: user_id={}", payload.user_id)

    traits = payload.traits or {}
    vertical     = traits.get("vertical", traits.get("industry", "other"))
    company_size = traits.get("company_size", "smb")
    channel      = (
        payload.context.get("campaign", {}).get("source", "organic")
        if payload.context else "organic"
    )
    plan_tier = traits.get("plan", traits.get("plan_tier", "free"))

    score = cold.score(
        vertical     = str(vertical),
        company_size = str(company_size),
        channel      = str(channel),
        plan_tier    = str(plan_tier),
    )

    # Background: send ltv_scored event back to Segment
    background.add_task(
        _track_segment_ltv_event,
        user_id = payload.user_id,
        score   = score,
    )

    return WebhookResponse(
        status      = "scored",
        customer_id = payload.user_id,
        ltv_36m     = score.get("ltv_36m"),
        segment     = score.get("segment"),
        message     = "LTV trait assigned",
    )


async def _track_segment_ltv_event(
    user_id: str,
    score: dict[str, Any],
) -> None:
    """Send ltv_scored track event to Segment.io (background task)."""
    try:
        from backend.integrations.segment_io import SegmentClient
        client = SegmentClient()
        await client.track_ltv_scored(
            user_id = user_id,
            ltv_36m = score.get("ltv_36m", 0),
            segment = score.get("segment", "low_value"),
        )
        logger.info("Segment ltv_scored event sent for user {}", user_id)
    except Exception as exc:
        logger.error("Failed to send Segment event for {}: {}", user_id, exc)