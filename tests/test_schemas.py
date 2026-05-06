"""Unit tests for Pydantic v2 API schemas."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from backend.api.schemas import (
    BatchScoreRequest,
    ColdStartRequest,
    ScoreRequest,
    LTVScoreResponse,
    ColdStartScoreResponse,
)


def test_score_request_customer_id_ok() -> None:
    req = ScoreRequest(customer_id="C12345")
    assert req.customer_id == "C12345"


def test_score_request_cold_start_ok() -> None:
    req = ScoreRequest(
        vertical="healthcare",
        company_size="enterprise",
        acquisition_channel="paid_search",
        plan_tier="enterprise_trial",
    )
    assert req.vertical == "healthcare"


def test_score_request_neither_raises() -> None:
    with pytest.raises(ValidationError):
        ScoreRequest()


def test_batch_request_empty_raises() -> None:
    with pytest.raises(ValidationError):
        BatchScoreRequest(customer_ids=[])


def test_batch_request_too_many_raises() -> None:
    with pytest.raises(ValidationError):
        BatchScoreRequest(customer_ids=[f"C{i}" for i in range(1001)])


def test_cold_start_invalid_size_raises() -> None:
    with pytest.raises(ValidationError):
        ColdStartRequest(
            vertical="healthcare",
            company_size="megacorp",
            acquisition_channel="organic",
            plan_tier="free",
        )


def test_cold_start_valid() -> None:
    req = ColdStartRequest(
        vertical="fintech",
        company_size="smb",
        acquisition_channel="organic",
        plan_tier="free",
    )
    assert req.company_size == "smb"


def test_ltv_score_response_valid() -> None:
    resp = LTVScoreResponse(
        customer_id="C001",
        ltv_12m=1200.0,
        ltv_24m=2500.0,
        ltv_36m=4000.0,
        segment="medium_value",
        recommended_max_cac=1200.0,
    )
    assert resp.ltv_36m == 4000.0
    assert resp.ltv_source == "full_model"


def test_cold_start_response_valid() -> None:
    resp = ColdStartScoreResponse(
        ltv_12m=500.0,
        ltv_36m=1500.0,
        ci_lower_36m=800.0,
        ci_upper_36m=2500.0,
        segment="medium_value",
        recommended_max_cac=600.0,
        match_quality="exact",
        firmographic_inputs={
            "vertical": "retail",
            "company_size": "smb",
            "channel": "organic",
            "plan_tier": "free",
        },
    )
    assert resp.ltv_source == "firmographic_prior"