"""Reset Airtable table and seed random rows from final_ltv_scores."""

from __future__ import annotations

import asyncio

from loguru import logger

from backend.db.supabase_client import SupabaseClient
from backend.integrations.airtable import AirtableClient


async def main() -> None:
    airtable = AirtableClient()

    # 1) Delete all Airtable rows
    record_ids = await airtable.list_record_ids(max_records=100000)
    logger.info("Found {} Airtable records", len(record_ids))
    await airtable.delete_records(record_ids)

    # 2) Pull random 300 rows from final_ltv_scores
    db = SupabaseClient(use_service_role=True)
    rows = db.execute_sql(
        """
        SELECT f.customer_id, f.ltv_36m, f.segment,
               f.recommended_max_cac, f.ltv_source
        FROM final_ltv_scores f
        ORDER BY RANDOM()
        LIMIT 300
        """
    )

    # 3) Seed Airtable
    records = [
        {
            "contact_id": str(row["customer_id"]),
            "ltv_score_36m": float(row["ltv_36m"] or 0),
            "ltv_segment": row["segment"] or "low_value",
            "recommended_max_cac": float(row["recommended_max_cac"] or 0),
            "ltv_source": row["ltv_source"] or "full_model",
            "email": f"customer_{row['customer_id']}@example.com",
        }
        for row in rows
    ]

    result = await airtable.upsert_contacts(records)
    logger.info("Seed complete: {}", result)


if __name__ == "__main__":
    asyncio.run(main())
