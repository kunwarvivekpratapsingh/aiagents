"""Structured JSON access logging middleware."""
from __future__ import annotations

import json
import logging
import time

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

logger = logging.getLogger("agentic_rag.access")

_SKIP_PATHS = frozenset({"/health", "/health/ready"})


class AccessLogMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if request.url.path in _SKIP_PATHS:
            return await call_next(request)

        start = time.perf_counter()
        response = await call_next(request)
        duration_ms = round((time.perf_counter() - start) * 1000, 2)
        req_id = getattr(request.state, "request_id", "-")

        logger.info(
            json.dumps({
                "event": "http_request",
                "req_id": req_id,
                "method": request.method,
                "path": request.url.path,
                "status": response.status_code,
                "duration_ms": duration_ms,
                "ip": request.client.host if request.client else "unknown",
            })
        )
        return response
