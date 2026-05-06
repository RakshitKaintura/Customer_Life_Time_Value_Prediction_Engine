"""
Marketing Sync Worker.

Scheduled background worker that runs on Render:
  - Uploads customer LTV segments to Google Ads Customer Match
  - Uploads high-LTV seed audience to Meta Ads
  - Syncs LTV scores to HubSpot contact properties
  - Schedules: every 24 hours

Run:
    python -m backend.workers.marketing_sync
"""

from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone

from loguru import logger

from backend.db.supabase_client import SupabaseClient
from backend.integrations.google_ads import GoogleAdsClient
from backend.integrations.hubspot import HubSpotClient
from backend.integrations.meta_ads import MetaAdsClient


async def sync_google_ads(db: SupabaseClient) -> dict:
    """Upload LTV segments to Google Ads Customer Match."""
    client = GoogleAdsClient()
    results = {}

    segments = ["champions", "high_value", "medium_value", "low_value"]

    for segment in segments:
        customers = db.execute_sql(
            """
            SELECT c.customer_id, f.ltv_36m
            FROM final_ltv_scores f
            JOIN customers c USING (customer_id)
            WHERE f.segment = :seg
            ORDER BY f.ltv_36m DESC
            LIMIT 50000
            """,
            {"seg": segment},
        )

        if not customers:
            continue

        # In production, would fetch emails from CRM
        emails = [f"customer_{c['customer_id']}@example.com" for c in customers]

        result = client.upload_customer_match_list(
            customer_emails = emails,
            list_name       = f"LTV_{segment.upper()}",
            segment         = segment,
        )
        results[segment] = result
        logger.info("Google Ads sync: {} → {} customers", segment, len(customers))

    return results


async def sync_meta_ads(db: SupabaseClient) -> dict:
    """Upload high-LTV seed audience to Meta Ads."""
    client = MetaAdsClient()

    # High-LTV seed: champions + high_value
    customers = db.execute_sql(
        """
        SELECT c.customer_id, f.ltv_36m
        FROM final_ltv_scores f
        JOIN customers c USING (customer_id)
        WHERE f.segment IN ('champions', 'high_value')
        ORDER BY f.ltv_36m DESC
        LIMIT 10000
        """,
    )

    if not customers:
        return {"status": "skipped", "reason": "no_high_value_customers"}

    emails = [f"customer_{c['customer_id']}@example.com" for c in customers]

    result = await client.upload_custom_audience(
        customer_emails = emails,
        audience_name   = "LTV_High_Value_Seed",
        description     = "Champions + High Value LTV segments for lookalike targeting",
    )
    logger.info("Meta Ads sync: {} high-value customers uploaded", len(emails))
    return result


async def sync_hubspot(db: SupabaseClient) -> dict:
    """Sync LTV scores to HubSpot contact properties."""
    client  = HubSpotClient()
    updated = 0
    errors  = 0

    customers = db.execute_sql(
        """
        SELECT f.customer_id, f.ltv_36m, f.segment,
               f.recommended_max_cac, f.ltv_source
        FROM final_ltv_scores f
        WHERE f.scored_at > NOW() - INTERVAL '25 hours'
        LIMIT 1000
        """
    )

    for cust in customers:
        try:
            await client.update_contact_ltv(
                contact_id = cust["customer_id"],
                ltv_36m    = float(cust["ltv_36m"] or 0),
                segment    = cust["segment"] or "low_value",
                max_cac    = float(cust["recommended_max_cac"] or 0),
                ltv_source = cust["ltv_source"] or "full_model",
            )
            updated += 1

            # Trigger workflow for high-value customers
            if cust["segment"] in ("champions", "high_value"):
                await client.trigger_high_value_workflow(cust["customer_id"])

        except Exception as exc:
            logger.warning("HubSpot sync failed for {}: {}", cust["customer_id"], exc)
            errors += 1

    logger.info("HubSpot sync: {} updated, {} errors", updated, errors)
    return {"updated": updated, "errors": errors}


async def run_sync_cycle() -> None:
    """Run one full marketing sync cycle."""
    db = SupabaseClient(use_service_role=True)

    logger.info("=== Marketing sync started at {} ===", datetime.now(timezone.utc).isoformat())

    # Run all syncs in parallel (they are independent)
    results = await asyncio.gather(
        sync_google_ads(db),
        sync_meta_ads(db),
        sync_hubspot(db),
        return_exceptions=True,
    )

    names = ["google_ads", "meta_ads", "hubspot"]
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