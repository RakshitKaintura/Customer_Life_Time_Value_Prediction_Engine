"""Fill missing Airtable `email` fields with placeholder addresses.

This script fetches contacts from Airtable, finds those missing the configured
`AIRTABLE_EMAIL_FIELD`, and upserts an `email` value of the form
`customer_{contact_id}@example.com` for those records only.

Run: `python scripts/fill_airtable_missing_emails.py`
"""

from __future__ import annotations

import asyncio
from loguru import logger

from backend.api.config import api_settings
from backend.integrations.airtable import AirtableClient


async def main() -> None:
    airtable = AirtableClient()
    email_field = api_settings.AIRTABLE_EMAIL_FIELD

    # Fetch up to BREVO_DAILY_LIMIT records (or 1000)
    max_records = api_settings.BREVO_DAILY_LIMIT or 1000
    fields = ["contact_id", email_field]
    recs = await airtable.list_contacts(fields=fields, max_records=max_records)

    missing = []
    for r in recs:
        fields_map = r.get("fields", {})
        contact_id = fields_map.get("contact_id")
        email = fields_map.get(email_field)
        if contact_id and not email:
            placeholder = f"customer_{contact_id}@example.com"
            missing.append({
                "contact_id": str(contact_id),
                email_field: placeholder,
            })

    logger.info("Found {} records missing email", len(missing))
    if not missing:
        logger.info("No missing emails to fill. Exiting.")
        return

    # Upsert in smaller batches to improve observability
    batch_size = 50
    total = len(missing)
    for i in range(0, total, batch_size):
        batch = missing[i : i + batch_size]
        logger.info("Upserting batch {}/{} ({} records)", (i // batch_size) + 1, (total + batch_size - 1) // batch_size, len(batch))
        result = await airtable.upsert_contacts(batch)
        logger.info("Batch upsert result: {}", result)

    logger.info("Completed upserting {} records", total)


if __name__ == "__main__":
    asyncio.run(main())
