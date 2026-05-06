"""
/score, /batch-score, /customer, /segment endpoints.
"""

from __future__ import annotations

import time
import uuid
from typing import Any

from fastapi import APIRouter, HTTPException, Query, status
from loguru import logger

from backend.api.dependencies import AuthKey, ColdStartSc, DBClient, ScoringEng
from backend.api.schemas import (
    BatchScoreRequest,
    BatchScoreResponse,
    ColdStartRequest,
    ColdStartScoreResponse,
    CustomerResponse,
    LookalikeResponse,
    LTVScoreResponse,
    ScoreRequest,
    SegmentListResponse,
)

router = APIRouter(tags=["Scoring"])


# ─────────────────────────────────────────────────────────────
# POST /score
# ─────────────────────────────────────────────────────────────

@router.post(
    "/score",
    response_model=LTVScoreResponse | ColdStartScoreResponse,
    summary="Score a customer for LTV",
    description=(
        "Score an existing customer using the full ensemble model, "
        "or a new customer using firmographic prior (cold-start)."
    ),
)
async def score_customer(
    request: ScoreRequest,
    _auth:   AuthKey,
    engine:  ScoringEng,
    cold:    ColdStartSc,
) -> dict[str, Any]:
    """
    Score endpoint:
    - If customer_id provided → full ensemble score
    - If firmographic fields provided → cold-start score
    - If both provided → full score (falls back to cold-start if no RFM data)
    """
    t0 = time.perf_counter()

    # Cold-start path
    if not request.customer_id and request.vertical:
        result = cold.score(
            vertical     = request.vertical or "other",
            company_size = request.company_size or "smb",
            channel      = request.acquisition_channel or "organic",
            plan_tier    = request.plan_tier or "free",
        )
        result["scoring_latency_ms"] = int((time.perf_counter() - t0) * 1000)
        return result

    # Full model path
    if not request.customer_id:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="customer_id is required when firmographic fields are not provided",
        )
    try:
        result = engine.score(customer_id=request.customer_id)
        return result
    except Exception as exc:
        logger.error("Scoring failed for {}: {}", request.customer_id, exc)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Scoring failed: {exc}",
        )


# ─────────────────────────────────────────────────────────────
# POST /batch-score
# ─────────────────────────────────────────────────────────────

@router.post(
    "/batch-score",
    response_model=BatchScoreResponse,
    summary="Score a batch of customers",
)
async def batch_score(
    request: BatchScoreRequest,
    _auth:   AuthKey,
    engine:  ScoringEng,
) -> BatchScoreResponse:
    """Score up to 1000 customers in a single request."""
    t0 = time.perf_counter()

    results  = engine.score_batch(request.customer_ids)
    errors   = sum(1 for r in results if "error" in r)
    success  = len(results) - errors
    elapsed  = int((time.perf_counter() - t0) * 1000)

    return BatchScoreResponse(
        results    = results,
        total      = len(results),
        success    = success,
        errors     = errors,
        latency_ms = elapsed,
    )


# ─────────────────────────────────────────────────────────────
# GET /customer/{id}
# ─────────────────────────────────────────────────────────────

@router.get(
    "/customer/{customer_id}",
    response_model=CustomerResponse,
    summary="Get LTV prediction for an existing customer",
)
async def get_customer(
    customer_id: str,
    _auth: AuthKey,
    db:    DBClient,
) -> CustomerResponse:
    """Fetch the latest LTV prediction and profile for a customer."""
    rows = db.execute_sql(
        """
        SELECT
            c.customer_id, c.country, c.acquisition_channel,
            c.first_purchase_date, c.total_orders, c.total_revenue,
            f.ltv_12m, f.ltv_24m, f.ltv_36m, f.segment,
            f.ltv_percentile, f.probability_alive_12m,
            f.recommended_max_cac, f.ltv_source, f.scored_at
        FROM customers c
        LEFT JOIN LATERAL (
            SELECT * FROM final_ltv_scores
            WHERE customer_id = :cid
            ORDER BY scored_at DESC LIMIT 1
        ) f ON TRUE
        WHERE c.customer_id = :cid
        """,
        {"cid": customer_id},
    )
    if not rows:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Customer '{customer_id}' not found",
        )
    return CustomerResponse(**rows[0])


# ─────────────────────────────────────────────────────────────
# GET /customer/{id}/lookalikes
# ─────────────────────────────────────────────────────────────

@router.get(
    "/customer/{customer_id}/lookalikes",
    response_model=LookalikeResponse,
    summary="Find lookalike customers via pgvector ANN search",
)
async def get_lookalikes(
    customer_id:   str,
    _auth:         AuthKey,
    db:            DBClient,
    top_n:         int = Query(default=10, ge=1, le=50),
    model_version: str = Query(default="transformer_v1"),
) -> LookalikeResponse:
    """
    Return the top-N customers most similar to the given customer
    based on their purchase sequence embeddings (cosine similarity).
    """
    try:
        rows = db.execute_sql(
            """
            SELECT candidate_customer_id, similarity, ltv_36m, segment
            FROM find_lookalikes(:cid, :model_ver, :n)
            ORDER BY similarity DESC
            """,
            {"cid": customer_id, "model_ver": model_version, "n": top_n},
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No embeddings found for customer '{customer_id}': {exc}",
        )

    return LookalikeResponse(
        query_customer_id = customer_id,
        lookalikes        = rows,
        model_version     = model_version,
        n_results         = len(rows),
    )


# ─────────────────────────────────────────────────────────────
# GET /segment/{segment}
# ─────────────────────────────────────────────────────────────

@router.get(
    "/segment/{segment}",
    response_model=SegmentListResponse,
    summary="List customers in an LTV segment",
)
async def get_segment_customers(
    segment:   str,
    _auth:     AuthKey,
    db:        DBClient,
    page:      int = Query(default=1, ge=1),
    page_size: int = Query(default=50, ge=1, le=200),
) -> SegmentListResponse:
    """List all customers in a segment: champions | high_value | medium_value | low_value."""
    valid_segments = {"champions", "high_value", "medium_value", "low_value"}
    if segment not in valid_segments:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid segment. Must be one of: {valid_segments}",
        )

    offset = (page - 1) * page_size

    rows = db.execute_sql(
        """
        SELECT f.customer_id, f.ltv_36m, f.ltv_12m, f.segment,
               f.ltv_percentile, f.recommended_max_cac,
               f.probability_alive_12m, f.scored_at,
               c.country, c.acquisition_channel
        FROM final_ltv_scores f
        JOIN customers c USING (customer_id)
        WHERE f.segment = :seg
        ORDER BY f.ltv_36m DESC
        LIMIT :lim OFFSET :off
        """,
        {"seg": segment, "lim": page_size, "off": offset},
    )

    count_rows = db.execute_sql(
        "SELECT COUNT(*) AS n FROM final_ltv_scores WHERE segment = :seg",
        {"seg": segment},
    )
    total = int(count_rows[0]["n"]) if count_rows else 0

    return SegmentListResponse(
        segment   = segment,
        customers = rows,
        total     = total,
        page      = page,
        page_size = page_size,
    )


# ─────────────────────────────────────────────────────────────
# POST /cold-start (explicit cold-start endpoint)
# ─────────────────────────────────────────────────────────────

@router.post(
    "/cold-start",
    response_model=ColdStartScoreResponse,
    summary="Score a zero-transaction customer using firmographic prior",
)
async def cold_start_score(
    request: ColdStartRequest,
    _auth:   AuthKey,
    cold:    ColdStartSc,
) -> ColdStartScoreResponse:
    """Explicit cold-start endpoint for fresh signups with no transaction history."""
    t0 = time.perf_counter()
    result = cold.score(
        vertical     = request.vertical,
        company_size = request.company_size,
        channel      = request.acquisition_channel,
        plan_tier    = request.plan_tier,
    )
    if request.customer_id:
        result["customer_id"] = request.customer_id
    result["scoring_latency_ms"] = int((time.perf_counter() - t0) * 1000)
    return ColdStartScoreResponse(**result)