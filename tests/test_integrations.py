"""Unit tests for marketing integration clients (mocked)."""

from __future__ import annotations

import pytest

from backend.integrations.airtable import AirtableClient
from backend.integrations.brevo import BrevoClient
from backend.integrations.google_ads import GoogleAdsClient, SEGMENT_TARGET_ROAS
from backend.integrations.segment_io import SegmentClient


def test_google_ads_no_token_skips() -> None:
    client = GoogleAdsClient(developer_token="")
    result = client.upload_customer_match_list(["a@b.com"], "test_list", "champions")
    assert result["status"] == "skipped"


def test_google_ads_target_roas_champions() -> None:
    assert SEGMENT_TARGET_ROAS["champions"] == 5.0


def test_google_ads_target_roas_low_value() -> None:
    assert SEGMENT_TARGET_ROAS["low_value"] == 2.0


def test_google_ads_build_csv() -> None:
    client = GoogleAdsClient()
    customers = [{"email": "a@b.com", "phone": "555-1234"}]
    csv_str = client.build_customer_match_csv(customers)
    assert "Email" in csv_str
    assert "a@b.com" in csv_str


def test_google_ads_roas_setting() -> None:
    client = GoogleAdsClient()
    result = client.set_target_roas_for_segment("camp_123", "champions", 10000.0)
    assert result["target_roas"] == 5.0
    assert result["status"] == "configured"


@pytest.mark.asyncio
async def test_airtable_no_key_skips() -> None:
    client = AirtableClient(api_token="", base_id="", table_id="")
    result = await client.upsert_contacts([{"contact_id": "C001"}])
    assert result["status"] == "skipped"


@pytest.mark.asyncio
async def test_segment_no_key_skips() -> None:
    client = SegmentClient(write_key="")
    result = await client.identify("user_123", 3000.0, "medium_value")
    assert result["status"] == "skipped"


@pytest.mark.asyncio
async def test_segment_track_no_key_skips() -> None:
    client = SegmentClient(write_key="")
    result = await client.track_ltv_scored("user_123", 3000.0, "medium_value")
    assert result["status"] == "skipped"


@pytest.mark.asyncio
async def test_brevo_no_key_skips() -> None:
    client = BrevoClient(api_key="")
    result = await client.send_segment_email("a@b.com", None, "champions")
    assert result["status"] == "skipped"