"""
FastAPI middleware:
  - Request/response logging
  - Latency tracking
  - Rate limiting (simple in-memory)
  - CORS
"""

from __future__ import annotations

import time
import uuid
from collections import defaultdict, deque
from typing import Callable

from fastapi import Request, Response
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from backend.api.config import api_settings


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Log every request with method, path, status, and latency."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        request_id = str(uuid.uuid4())[:8]
        start = time.perf_counter()

        # Attach request ID
        request.state.request_id = request_id

        response = await call_next(request)

        elapsed_ms = int((time.perf_counter() - start) * 1000)
        logger.info(
            "{} {} {} {} {}ms",
            request_id,
            request.method,
            request.url.path,
            response.status_code,
            elapsed_ms,
        )

        response.headers["X-Request-ID"]    = request_id
        response.headers["X-Response-Time"] = f"{elapsed_ms}ms"
        return response


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Simple sliding-window rate limiter.
    Keyed by API key (or IP if no key).
    Default: 100 requests / minute.
    """

    def __init__(self, app: ASGIApp, requests_per_minute: int = 100) -> None:
        super().__init__(app)
        self.rpm   = requests_per_minute
        self.window = 60   # seconds
        self._counters: dict[str, deque] = defaultdict(deque)

    def _get_key(self, request: Request) -> str:
        api_key = request.headers.get("X-API-Key", "")
        if api_key:
            return f"key:{api_key}"
        forwarded = request.headers.get("X-Forwarded-For", "")
        return f"ip:{forwarded or request.client.host if request.client else 'unknown'}"

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Skip rate limiting for health checks
        if request.url.path in ("/health", "/docs", "/openapi.json", "/redoc"):
            return await call_next(request)

        key = self._get_key(request)
        now = time.time()
        window_start = now - self.window

        # Remove old entries
        q = self._counters[key]
        while q and q[0] < window_start:
            q.popleft()

        if len(q) >= self.rpm:
            logger.warning("Rate limit exceeded for {}", key)
            from fastapi.responses import JSONResponse
            return JSONResponse(
                status_code=429,
                content={"detail": "Rate limit exceeded. Try again in a minute."},
                headers={"Retry-After": "60"},
            )

        q.append(now)
        return await call_next(request)


def configure_cors(app: ASGIApp) -> None:
    """Add CORS middleware to allow dashboard and integrations."""
    origins = [
        "http://localhost:3000",
        "http://localhost:3001",
        "https://*.vercel.app",
    ]
    if api_settings.ENVIRONMENT == "production":
        origins = ["https://*.vercel.app"]

    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )