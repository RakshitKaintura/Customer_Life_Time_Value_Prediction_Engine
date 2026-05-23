import asyncio
import json
from backend.db.supabase_client import SupabaseClient
from backend.integrations.airtable import AirtableClient


async def main():
    db = SupabaseClient(use_service_role=True)
    pr = db.execute_sql("SELECT * FROM pipeline_runs ORDER BY started_at DESC LIMIT 1")
    print('PIPELINE_RUN:')
    print(json.dumps(pr, indent=2, default=str))

    airtable = AirtableClient()
    fields = ["contact_id", "email", "ltv_segment"]
    recs = await airtable.list_contacts(fields=fields, max_records=300)
    total = len(recs)
    missing_email = [r for r in recs if not r.get('fields', {}).get('email')]
    missing_segment = [r for r in recs if not r.get('fields', {}).get('ltv_segment')]
    print('\nAIRTABLE SAMPLE REPORT:')
    print('total_fetched:', total)
    print('missing_email_count:', len(missing_email))
    print('missing_segment_count:', len(missing_segment))
    print('\nSample missing_email records (up to 10):')
    print(json.dumps([{'id': r.get('id'), 'fields': r.get('fields')} for r in missing_email[:10]], indent=2))


if __name__ == '__main__':
    asyncio.run(main())
