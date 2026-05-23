"""
Airtable CRM Integration.

Upserts contact properties with LTV predictions into an Airtable table.
Requires: personal access token with data.records read/write.
"""

from __future__ import annotations

from typing import Any

import httpx
from loguru import logger

from backend.api.config import api_settings


class AirtableClient:
    """Async Airtable API client for contact upserts."""

    def __init__(
        self,
        api_token: str | None = None,
        base_id: str | None = None,
        table_id: str | None = None,
    ) -> None:
        self.api_token = api_token or api_settings.AIRTABLE_API_TOKEN
        self.base_id = base_id or api_settings.AIRTABLE_BASE_ID
        self.table_id = table_id or api_settings.AIRTABLE_TABLE_ID

    @property
    def _url(self) -> str:
        return f"https://api.airtable.com/v0/{self.base_id}/{self.table_id}"

    @property
    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json",
        }

    async def upsert_contacts(self, records: list[dict[str, Any]]) -> dict[str, Any]:
        """
        Upsert contact records by contact_id.

        Requires an Airtable field named 'contact_id' (primary field is fine).
        """
        if not self.api_token or not self.base_id or not self.table_id:
            logger.warning("Airtable not configured — skipping upsert")
            return {"status": "skipped", "reason": "missing_config"}

        if not records:
            return {"status": "skipped", "reason": "no_records"}

        created = 0
        updated = 0

        async with httpx.AsyncClient(timeout=20.0) as client:
            for item in records:
                contact_id = str(item.get("contact_id", "")).strip()
                if not contact_id:
                    continue

                record_id = await _find_record_id(client, self._url, self._headers, contact_id)
                if record_id:
                    updated += await _update_record(client, self._url, self._headers, record_id, item)
                else:
                    created += await _create_record(client, self._url, self._headers, item)

        logger.info("Airtable upsert complete: {} created, {} updated", created, updated)
        return {"status": "upserted", "created": created, "updated": updated}

    async def list_contacts(
        self,
        fields: list[str],
        page_size: int = 100,
        max_records: int = 1000,
    ) -> list[dict[str, Any]]:
        if not self.api_token or not self.base_id or not self.table_id:
            logger.warning("Airtable not configured — skipping list")
            return []

        results: list[dict[str, Any]] = []
        offset: str | None = None
        async with httpx.AsyncClient(timeout=20.0) as client:
            while len(results) < max_records:
                params: list[tuple[str, str]] = [("pageSize", str(page_size))]
                for field in fields:
                    params.append(("fields[]", field))
                if offset:
                    params.append(("offset", offset))

                response = await client.get(self._url, headers=self._headers, params=params)
                try:
                    response.raise_for_status()
                except httpx.HTTPStatusError:
                    logger.error("Airtable list failed: {}", response.text)
                    raise

                data = response.json()
                records = data.get("records", [])
                results.extend(records)
                offset = data.get("offset")
                if not offset:
                    break

        return results[:max_records]

    async def list_record_ids(self, page_size: int = 100, max_records: int = 10000) -> list[str]:
        if not self.api_token or not self.base_id or not self.table_id:
            logger.warning("Airtable not configured — skipping list")
            return []

        record_ids: list[str] = []
        offset: str | None = None
        async with httpx.AsyncClient(timeout=20.0) as client:
            while len(record_ids) < max_records:
                params: list[tuple[str, str]] = [("pageSize", str(page_size))]
                if offset:
                    params.append(("offset", offset))

                response = await client.get(self._url, headers=self._headers, params=params)
                try:
                    response.raise_for_status()
                except httpx.HTTPStatusError:
                    logger.error("Airtable list failed: {}", response.text)
                    raise

                data = response.json()
                records = data.get("records", [])
                record_ids.extend([r.get("id") for r in records if r.get("id")])
                offset = data.get("offset")
                if not offset:
                    break

        return record_ids[:max_records]

    async def delete_records(self, record_ids: list[str]) -> dict[str, Any]:
        if not self.api_token or not self.base_id or not self.table_id:
            logger.warning("Airtable not configured — skipping delete")
            return {"status": "skipped", "reason": "missing_config"}

        if not record_ids:
            return {"status": "skipped", "reason": "no_records"}

        deleted = 0
        async with httpx.AsyncClient(timeout=20.0) as client:
            for chunk in _chunk_list(record_ids, size=10):
                params: list[tuple[str, str]] = []
                for rid in chunk:
                    params.append(("records[]", rid))

                response = await client.delete(self._url, headers=self._headers, params=params)
                try:
                    response.raise_for_status()
                except httpx.HTTPStatusError:
                    logger.error("Airtable delete failed: {}", response.text)
                    raise

                data = response.json()
                deleted += len([r for r in data.get("records", []) if r.get("deleted")])

        logger.info("Airtable delete complete: {} records", deleted)
        return {"status": "deleted", "n_records": deleted}


async def _find_record_id(
    client: httpx.AsyncClient,
    base_url: str,
    headers: dict[str, str],
    contact_id: str,
) -> str | None:
    formula_value = contact_id.replace("'", "\\'")
    params = {"filterByFormula": f"{{contact_id}}='{formula_value}'", "maxRecords": 1}
    response = await client.get(base_url, headers=headers, params=params)
    try:
        response.raise_for_status()
    except httpx.HTTPStatusError:
        logger.error("Airtable lookup failed: {}", response.text)
        raise

    records = response.json().get("records", [])
    return records[0].get("id") if records else None


async def _create_record(
    client: httpx.AsyncClient,
    base_url: str,
    headers: dict[str, str],
    fields: dict[str, Any],
) -> int:
    payload = {"records": [{"fields": fields}], "typecast": True}
    response = await client.post(base_url, json=payload, headers=headers)
    try:
        response.raise_for_status()
    except httpx.HTTPStatusError:
        logger.error("Airtable create failed: {}", response.text)
        raise
    return len(response.json().get("records", []))


async def _update_record(
    client: httpx.AsyncClient,
    base_url: str,
    headers: dict[str, str],
    record_id: str,
    fields: dict[str, Any],
) -> int:
    payload = {"fields": fields, "typecast": True}
    response = await client.patch(f"{base_url}/{record_id}", json=payload, headers=headers)
    try:
        response.raise_for_status()
    except httpx.HTTPStatusError:
        logger.error("Airtable update failed: {}", response.text)
        raise
    return 1


def _chunk_list(items: list[str], size: int = 10) -> list[list[str]]:
    return [items[i : i + size] for i in range(0, len(items), size)]
