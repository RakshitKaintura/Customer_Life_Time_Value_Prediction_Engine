"""
Marketing Sync Worker.

Scheduled background worker that runs on Render:
  - Syncs LTV scores to Airtable contact records
  - Sends segment-based emails via Brevo
  - Schedules: every 24 hours

Run:
    python -m backend.workers.marketing_sync
"""

from __future__ import annotations

import asyncio
import os
import time
from datetime import datetime, timezone

from loguru import logger

from backend.api.config import api_settings
from backend.db.supabase_client import SupabaseClient
from backend.integrations.airtable import AirtableClient
from backend.integrations.brevo import BrevoClient


async def sync_airtable(db: SupabaseClient) -> dict:
    """Sync LTV scores to Airtable contact records."""
    client = AirtableClient()

    days_raw = os.getenv("AIRTABLE_SYNC_DAYS", "0").strip()
    days = int(days_raw) if days_raw.isdigit() else 0
    where_clause = ""
    params = {}
    if days > 0:
        where_clause = "WHERE f.scored_at > NOW() - (:days || ' days')::interval"
        params = {"days": days}

    customers = db.execute_sql(
        f"""
        SELECT f.customer_id, f.ltv_36m, f.segment,
               f.recommended_max_cac, f.ltv_source
        FROM final_ltv_scores f
        {where_clause}
        LIMIT 1000
        """,
        params,
    )

    if not customers:
        total = db.execute_sql("SELECT COUNT(*) AS n FROM final_ltv_scores")
        logger.warning("Airtable sync found 0 records; total in final_ltv_scores: {}", total)

    records = [
        {
            "contact_id": str(cust["customer_id"]),
            "ltv_score_36m": float(cust["ltv_36m"] or 0),
            "ltv_segment": cust["segment"] or "low_value",
            "recommended_max_cac": float(cust["recommended_max_cac"] or 0),
            "ltv_source": cust["ltv_source"] or "full_model",
        }
        for cust in customers
    ]

    result = await client.upsert_contacts(records)
    logger.info("Airtable sync: {}", result)
    return result


async def sync_brevo() -> dict:
    """Send segment-based emails using Brevo from Airtable contacts."""
    airtable = AirtableClient()
    brevo = BrevoClient()
    email_field = api_settings.AIRTABLE_EMAIL_FIELD

    fields = [
        "contact_id",
        email_field,
        "ltv_segment",
        "ltv_score_36m",
        "recommended_max_cac",
        "ltv_source",
    ]

    records = await airtable.list_contacts(fields=fields, max_records=api_settings.BREVO_DAILY_LIMIT)
    if not records:
        return {"status": "skipped", "reason": "no_contacts"}

    sent = 0
    skipped = 0
    for record in records:
        fields_map = record.get("fields", {})
        email = fields_map.get(email_field)
        segment = fields_map.get("ltv_segment")
        if not email or not segment:
            skipped += 1
            continue

        params = {
            "contact_id": fields_map.get("contact_id"),
            "ltv_score_36m": fields_map.get("ltv_score_36m"),
            "recommended_max_cac": fields_map.get("recommended_max_cac"),
            "ltv_source": fields_map.get("ltv_source"),
            "segment": segment,
        }

        result = await brevo.send_segment_email(
            to_email=str(email),
            to_name=None,
            segment=str(segment),
            params=params,
        )

        if result.get("status") == "sent":
            sent += 1
        else:
            skipped += 1

    return {"status": "complete", "sent": sent, "skipped": skipped}


async def run_sync_cycle() -> None:
    """Run one full marketing sync cycle."""
    db = SupabaseClient(use_service_role=True)

    logger.info("=== Marketing sync started at {} ===", datetime.now(timezone.utc).isoformat())

    # Run all syncs in parallel (they are independent)
    results = await asyncio.gather(
        sync_airtable(db),
        sync_brevo(),
        return_exceptions=True,
    )

    names = ["airtable", "brevo"]
    for name, result in zip(names, results):
        if isinstance(result, Exception):
            logger.error("{} sync failed: {}", name, result)
        else:
            logger.info("{} sync complete: {}", name, result)

    # Log sync run
    db.bulk_upsert("pipeline_runs", [{
        "run_id":        f"mkt_sync_{int(time.time())}",
        "pipeline_name": "marketing_sync",
        "status":        "success",
        "started_at":    datetime.now(timezone.utc).isoformat(),
        "records_processed": None,
        "metadata": {"results": {n: str(r) for n, r in zip(names, results)}},
    }], conflict_columns=["run_id"])

    logger.info("=== Marketing sync complete ===")


async def main() -> None:
    """Run sync every 24 hours."""
    run_once = os.getenv("MARKETING_SYNC_ONCE", "").strip().lower() in {"1", "true", "yes"}
    if run_once:
        await run_sync_cycle()
        return

    while True:
        try:
            await run_sync_cycle()
        except Exception as exc:
            logger.error("Sync cycle failed: {}", exc)

        # Wait 24 hours
        logger.info("Next sync in 24 hours")
        await asyncio.sleep(24 * 60 * 60)


if __name__ == "__main__":
    asyncio.run(main())