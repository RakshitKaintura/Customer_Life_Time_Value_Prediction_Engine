"""
Integration tests for FastAPI endpoints.
Uses httpx.AsyncClient with the ASGI transport (no server needed).
"""

from __future__ import annotations

import pytest
import pytest_asyncio
from fastapi.testclient import TestClient
from httpx import AsyncClient, ASGITransport

from backend.api.main import app
from backend.api.dependencies import set_cold_start_scorer, set_scoring_engine


# ─────────────────────────────────────────────────────────────
# Mock dependencies
# ─────────────────────────────────────────────────────────────

class MockScoringEngine:
    def score(self, customer_id: str, **kwargs) -> dict:
        return {
            "customer_id":          customer_id,
            "ltv_source":           "full_model",
            "ltv_12m":              1200.0,
            "ltv_24m":              2800.0,
            "ltv_36m":              4500.0,
            "ltv_percentile":       72,
            "segment":              "medium_value",
            "probability_alive_12m": 0.82,
            "recommended_max_cac":  1350.0,
            "confidence_interval_36m": [3000.0, 6000.0],
            "top_ltv_drivers":      ["High purchase frequency", "Good avg order value"],
            "causal_levers":        ["Complete onboarding to increase LTV by £300"],
            "lookalike_customer_ids": ["C001", "C002", "C003"],
            "scoring_latency_ms":   45,
        }

    def score_batch(self, customer_ids: list[str]) -> list[dict]:
        return [self.score(cid) for cid in customer_ids]


class MockColdStartScorer:
    def score(self, vertical, company_size, channel, plan_tier) -> dict:
        return {
            "ltv_source":          "firmographic_prior",
            "ltv_12m":             2100.0,
            "ltv_36m":             5500.0,
            "ci_lower_36m":        3000.0,
            "ci_upper_36m":        9000.0,
            "segment":             "medium_value",
            "recommended_max_cac": 2200.0,
            "match_quality":       "exact",
            "firmographic_inputs": {
                "vertical":    vertical,
                "company_size": company_size,
                "channel":     channel,
                "plan_tier":   plan_tier,
            },
        }


# Set mock dependencies at import time
set_scoring_engine(MockScoringEngine())
set_cold_start_scorer(MockColdStartScorer())


# ─────────────────────────────────────────────────────────────
# Sync client fixture (TestClient)
# ─────────────────────────────────────────────────────────────

@pytest.fixture
def client() -> TestClient:
    return TestClient(app, raise_server_exceptions=True)


# ─────────────────────────────────────────────────────────────
# Health endpoint
# ─────────────────────────────────────────────────────────────

def test_health_returns_200(client: TestClient) -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert "status" in data
    assert "environment" in data


def test_root_returns_service_name(client: TestClient) -> None:
    resp = client.get("/")
    assert resp.status_code == 200
    assert "LTV" in resp.json()["service"]


# ─────────────────────────────────────────────────────────────
# Score endpoint
# ─────────────────────────────────────────────────────────────

def test_score_existing_customer(client: TestClient) -> None:
    resp = client.post("/score", json={"customer_id": "C12345"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["customer_id"] == "C12345"
    assert data["ltv_36m"] > 0
    assert data["segment"] in {"champions", "high_value", "medium_value", "low_value"}


def test_score_cold_start(client: TestClient) -> None:
    resp = client.post("/score", json={
        "vertical":            "healthcare",
        "company_size":        "enterprise",
        "acquisition_channel": "paid_search",
        "plan_tier":           "enterprise_trial",
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["ltv_source"] == "firmographic_prior"
    assert data["ltv_36m"] > 0


def test_score_missing_both_fields(client: TestClient) -> None:
    resp = client.post("/score", json={})
    assert resp.status_code == 422


def test_cold_start_explicit_endpoint(client: TestClient) -> None:
    resp = client.post("/cold-start", json={
        "vertical":            "fintech",
        "company_size":        "smb",
        "acquisition_channel": "organic",
        "plan_tier":           "free",
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["ltv_source"] == "firmographic_prior"


def test_cold_start_invalid_company_size(client: TestClient) -> None:
    resp = client.post("/cold-start", json={
        "vertical":            "healthcare",
        "company_size":        "giant_corp",
        "acquisition_channel": "organic",
        "plan_tier":           "free",
    })
    assert resp.status_code == 422


# ─────────────────────────────────────────────────────────────
# Batch score
# ─────────────────────────────────────────────────────────────

def test_batch_score(client: TestClient) -> None:
    resp = client.post("/batch-score", json={
        "customer_ids": ["C001", "C002", "C003"]
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 3
    assert data["success"] == 3
    assert data["errors"] == 0
    assert len(data["results"]) == 3


def test_batch_score_empty_list(client: TestClient) -> None:
    resp = client.post("/batch-score", json={"customer_ids": []})
    assert resp.status_code == 422


# ─────────────────────────────────────────────────────────────
# Segment endpoint
# ─────────────────────────────────────────────────────────────

def test_segment_invalid_name(client: TestClient) -> None:
    resp = client.get("/segment/super_rich")
    assert resp.status_code == 400


# ─────────────────────────────────────────────────────────────
# Response structure tests
# ─────────────────────────────────────────────────────────────

def test_score_response_has_confidence_interval(client: TestClient) -> None:
    resp = client.post("/score", json={"customer_id": "C99"})
    data = resp.json()
    assert "confidence_interval_36m" in data


def test_score_response_has_causal_levers(client: TestClient) -> None:
    resp = client.post("/score", json={"customer_id": "C99"})
    data = resp.json()
    assert "causal_levers" in data
    assert isinstance(data["causal_levers"], list)


def test_score_response_has_lookalikes(client: TestClient) -> None:
    resp = client.post("/score", json={"customer_id": "C99"})
    data = resp.json()
    assert "lookalike_customer_ids" in data
    assert isinstance(data["lookalike_customer_ids"], list)


def test_openapi_schema_available(client: TestClient) -> None:
    resp = client.get("/openapi.json")
    assert resp.status_code == 200
    schema = resp.json()
    assert "paths" in schema
    assert "/score" in schema["paths"]