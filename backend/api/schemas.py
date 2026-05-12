"""
Pydantic v2 request and response schemas for all API endpoints.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator


# ─────────────────────────────────────────────────────────────
# Score request schemas
# ─────────────────────────────────────────────────────────────

class ScoreRequest(BaseModel):
    """Request body for POST /score — score a new or existing customer."""

    customer_id: str | None = Field(None, description="Existing customer ID")

    # Firmographic fields (used for cold-start if no transactions)
    vertical:            str | None = Field(None, description="Industry vertical")
    company_size:        str | None = Field(None, description="smb | mid_market | enterprise")
    acquisition_channel: str | None = Field(None, description="organic | paid_search | ...")
    plan_tier:           str | None = Field(None, description="free | starter | professional | enterprise_trial")

    # Direct RFM fields (override DB lookup for testing)
    frequency:    int | None   = Field(None, ge=0)
    recency_days: float | None = Field(None, ge=0)
    t_days:       int | None   = Field(None, ge=1)
    monetary_avg: float | None = Field(None, gt=0)

    @model_validator(mode="after")
    def require_customer_or_firmographic(self) -> "ScoreRequest":
        if not self.customer_id and not self.vertical:
            raise ValueError(
                "Either customer_id or firmographic fields (vertical, company_size, "
                "acquisition_channel, plan_tier) must be provided"
            )
        return self


class BatchScoreRequest(BaseModel):
    """Request body for POST /batch-score."""
    customer_ids: list[str] = Field(..., min_length=1, max_length=1000)


class ColdStartRequest(BaseModel):
    """Request body for scoring a zero-transaction customer."""
    vertical:            str = Field(..., description="Industry vertical")
    company_size:        str = Field(..., description="smb | mid_market | enterprise")
    acquisition_channel: str = Field(..., description="Channel that acquired this customer")
    plan_tier:           str = Field(..., description="Signup plan tier")
    customer_id:         str | None = None

    @field_validator("company_size")
    @classmethod
    def validate_company_size(cls, v: str) -> str:
        valid = {"smb", "mid_market", "enterprise"}
        if v.lower() not in valid:
            raise ValueError(f"company_size must be one of {valid}")
        return v.lower()


# ─────────────────────────────────────────────────────────────
# Score response schemas
# ─────────────────────────────────────────────────────────────

class LTVScoreResponse(BaseModel):
    """Full model LTV score response."""

    customer_id:              str
    ltv_source:               str = "full_model"

    ltv_12m:                  float
    ltv_24m:                  float
    ltv_36m:                  float

    ltv_percentile:           int | None   = None
    segment:                  str
    probability_alive_12m:    float | None = None
    recommended_max_cac:      float

    confidence_interval_36m:  tuple[float, float] | None = None

    top_ltv_drivers:          list[str] = Field(default_factory=list)
    causal_levers:            list[str] = Field(default_factory=list)
    lookalike_customer_ids:   list[str] = Field(default_factory=list)

    scoring_latency_ms:       int | None = None


class ColdStartScoreResponse(BaseModel):
    """Firmographic prior LTV score response (zero-transaction customer)."""

    customer_id:              str | None   = None
    ltv_source:               str = "firmographic_prior"

    ltv_12m:                  float
    ltv_36m:                  float

    ci_lower_36m:             float
    ci_upper_36m:             float

    segment:                  str
    recommended_max_cac:      float
    match_quality:            str

    firmographic_inputs:      dict[str, str]
    scoring_latency_ms:       int | None   = None


class BatchScoreResponse(BaseModel):
    """Response for POST /batch-score."""
    results:    list[dict[str, Any]]
    total:      int
    success:    int
    errors:     int
    latency_ms: int


# ─────────────────────────────────────────────────────────────
# Customer schemas
# ─────────────────────────────────────────────────────────────

class CustomerResponse(BaseModel):
    """Response for GET /customer/{id}."""

    customer_id:           str
    country:               str | None = None
    acquisition_channel:   str | None = None
    first_purchase_date:   datetime | None = None
    total_orders:          int | None = None
    total_revenue:         float | None = None

    # LTV prediction
    ltv_12m:               float | None = None
    ltv_24m:               float | None = None
    ltv_36m:               float | None = None
    segment:               str | None   = None
    ltv_percentile:        int | None   = None
    probability_alive_12m: float | None = None
    recommended_max_cac:   float | None = None
    ltv_source:            str | None   = None
    scored_at:             datetime | None = None


class LookalikeResponse(BaseModel):
    """Response for GET /customer/{id}/lookalikes."""
    query_customer_id:  str
    lookalikes:         list[dict[str, Any]]
    model_version:      str
    n_results:          int


class SegmentListResponse(BaseModel):
    """Response for GET /segment/{segment}."""
    segment:    str
    customers:  list[dict[str, Any]]
    total:      int
    page:       int
    page_size:  int


# ─────────────────────────────────────────────────────────────
# Model performance schemas
# ─────────────────────────────────────────────────────────────

class ModelPerformanceResponse(BaseModel):
    """Response for GET /model-performance."""

    fusion_model_version:     str | None = None
    bgnbd_model_version:      str | None = None
    transformer_model_version: str | None = None

    # Fusion metrics
    fusion_mae_ltv_12m:       float | None = None
    fusion_gini:              float | None = None
    fusion_top_decile_lift:   float | None = None
    fusion_calibration_error: float | None = None

    # BG/NBD metrics
    bgnbd_r2_frequency:       float | None = None
    bgnbd_mae_ltv_12m:        float | None = None

    # Transformer metrics
    transformer_mae_ltv_12m:  float | None = None
    transformer_gini:         float | None = None

    # Segment distribution
    segment_distribution:     dict[str, int] = Field(default_factory=lambda: {})

    # Coverage
    n_customers_scored:       int | None = None
    last_scored_at:           datetime | None = None


# ─────────────────────────────────────────────────────────────
# Webhook schemas
# ─────────────────────────────────────────────────────────────

class HubSpotWebhookPayload(BaseModel):
    """HubSpot new contact webhook payload."""
    contact_id:     str
    email:          str | None = None
    company:        str | None = None
    vertical:       str | None = None
    company_size:   str | None = None
    plan_tier:      str | None = None
    channel:        str | None = None
    properties:     dict[str, Any] = Field(default_factory=lambda: {})


class SegmentWebhookPayload(BaseModel):
    """Segment.io identify event payload."""
    user_id:    str
    anonymous_id: str | None = None
    traits:     dict[str, Any] = Field(default_factory=lambda: {})
    context:    dict[str, Any] = Field(default_factory=lambda: {})


class WebhookResponse(BaseModel):
    """Generic webhook response."""
    status:      str = "ok"
    customer_id: str | None = None
    ltv_36m:     float | None = None
    segment:     str | None = None
    message:     str | None = None


# ─────────────────────────────────────────────────────────────
# Health check
# ─────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status:           str = "ok"
    environment:      str
    models_loaded:    bool
    db_connected:     bool
    scoring_engine:   bool
    version:          str = "1.0.0"