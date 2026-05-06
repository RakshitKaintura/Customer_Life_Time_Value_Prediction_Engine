"""
Webhook handlers:
  POST /webhook/hubspot  — HubSpot new contact → score + update CRM
  POST /webhook/segment  — Segment.io identify → score + return traits
"""

from __future__ import annotations

import asyncio
from typing import Any

from fastapi import APIRouter, BackgroundTasks, HTTPException, Request, status
from loguru import logger

from backend.api.dependencies import ColdStartSc, DBClient
from backend.api.schemas import (
    HubSpotWebhookPayload,
    SegmentWebhookPayload,
    WebhookResponse,
)

router = APIRouter(prefix="/webhook", tags=["Webhooks"])


# ─────────────────────────────────────────────────────────────
# HubSpot webhook
# ─────────────────────────────────────────────────────────────

@router.post(
    "/hubspot",
    response_model=WebhookResponse,
    summary="HubSpot new contact → score and update CRM",
)
async def hubspot_webhook(
    payload:    HubSpotWebhookPayload,
    background: BackgroundTasks,
    cold:       ColdStartSc,
    db:         DBClient,
) -> WebhookResponse:
    """
    Called when HubSpot creates a new contact.
    Scores the contact using firmographic prior and schedules
    a background task to update HubSpot contact properties.
    """
    logger.info("HubSpot webhook: contact_id={}", payload.contact_id)

    # Extract firmographic fields from payload properties
    vertical     = payload.vertical or payload.properties.get("industry", "other")
    company_size = payload.company_size or payload.properties.get("company_size", "smb")
    channel      = payload.channel or payload.properties.get("hs_analytics_source", "organic")
    plan_tier    = payload.plan_tier or payload.properties.get("plan_tier", "free")

    # Score immediately (cold-start, < 20ms)
    score = cold.score(
        vertical     = vertical,
        company_size = company_size,
        channel      = channel,
        plan_tier    = plan_tier,
    )

    # Background: update HubSpot contact properties
    background.add_task(
        _update_hubspot_contact,
        contact_id = payload.contact_id,
        score      = score,
    )

    return WebhookResponse(
        status      = "scored",
        customer_id = payload.contact_id,
        ltv_36m     = score.get("ltv_36m"),
        segment     = score.get("segment"),
        message     = f"Cold-start score applied. Segment: {score.get('segment')}",
    )


async def _update_hubspot_contact(
    contact_id: str,
    score: dict[str, Any],
) -> None:
    """Update HubSpot contact with LTV properties (background task)."""
    try:
        from backend.integrations.hubspot import HubSpotClient
        client = HubSpotClient()
        await client.update_contact_ltv(
            contact_id = contact_id,
            ltv_36m    = score.get("ltv_36m", 0),
            segment    = score.get("segment", "low_value"),
            max_cac    = score.get("recommended_max_cac", 0),
            ltv_source = score.get("ltv_source", "firmographic_prior"),
        )
        logger.info("HubSpot contact {} updated with LTV score", contact_id)
    except Exception as exc:
        logger.error("Failed to update HubSpot contact {}: {}", contact_id, exc)


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