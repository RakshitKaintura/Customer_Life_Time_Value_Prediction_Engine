"""
LTV Prediction Engine — FastAPI Application.

Endpoints:
    POST   /score                   Score a customer (full model or cold-start)
    POST   /cold-start              Explicit cold-start scoring
    GET    /customer/{id}           Fetch LTV prediction for existing customer
    GET    /customer/{id}/lookalikes Top-N similar customers via pgvector
    GET    /segment/{segment}       List customers in a LTV segment
    POST   /batch-score             Score a batch of customers
    GET    /health                  Service health check
    GET    /model-performance       Current model MAE and calibration metrics
    POST   /webhook/hubspot         HubSpot new contact → score + update CRM
    POST   /webhook/segment         Segment.io identify → score + return LTV
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.gzip import GZipMiddleware
from loguru import logger

from backend.api.config import api_settings
from backend.api.middleware import (
    RateLimitMiddleware,
    RequestLoggingMiddleware,
    configure_cors,
)
from backend.api.routers import health, scoring, webhooks
from backend.api.startup import lifespan

# ─────────────────────────────────────────────────────────────
# Application
# ─────────────────────────────────────────────────────────────

app = FastAPI(
    title       = "LTV Prediction Engine",
    description = (
        "Customer Lifetime Value scoring API combining BG/NBD probabilistic models, "
        "Transformer sequence models, Causal ML feature attribution, and XGBoost stacking "
        "into a single real-time scoring endpoint."
    ),
    version     = "1.0.0",
    lifespan    = lifespan,
    docs_url    = "/docs",
    redoc_url   = "/redoc",
    openapi_url = "/openapi.json",
)

# ─────────────────────────────────────────────────────────────
# Middleware (order matters — outermost runs first)
# ─────────────────────────────────────────────────────────────

configure_cors(app)
app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(RequestLoggingMiddleware)
app.add_middleware(
    RateLimitMiddleware,
    requests_per_minute=api_settings.API_RATE_LIMIT_PER_MINUTE,
)

# ─────────────────────────────────────────────────────────────
# Routers
# ─────────────────────────────────────────────────────────────

app.include_router(scoring.router)
app.include_router(health.router)
app.include_router(webhooks.router)

# ─────────────────────────────────────────────────────────────
# Root
# ─────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
async def root() -> dict:
    return {
        "service":     "LTV Prediction Engine",
        "version":     "1.0.0",
        "environment": api_settings.ENVIRONMENT,
        "docs":        "/docs",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend.api.main:app",
        host    = "0.0.0.0",
        port    = 8000,
        reload  = api_settings.ENVIRONMENT == "development",
        workers = 1,
        log_level = api_settings.LOG_LEVEL.lower(),
    )