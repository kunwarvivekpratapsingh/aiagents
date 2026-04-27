"""Sliding-window rate limiter middleware.

Limits requests per (client_ip, route_prefix) using an in-memory deque.
No external dependencies — suitable for single-instance deployments.
For multi-instance, replace with a Redis-backed counter.
"""
from __future__ import annotations

import threading
import time
from collections import defaultdict, deque

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

from ...config import config

# (max_requests, window_seconds)  per path prefix
_ROUTE_LIMITS: dict[str, tuple[int, int]] = {
    "/chat/stream": (30, 60),
    "/chat":        (60, 60),
    "/documents/upload": (10, 60),
    "/documents":   (120, 60),
}


class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, limits: dict[str, tuple[int, int]] | None = None) -> None:
        super().__init__(app)
        self._limits = limits or _ROUTE_LIMITS
        self._windows: dict[tuple[str, str], deque] = defaultdict(deque)
        self._lock = threading.Lock()

    async def dispatch(self, request: Request, call_next):
        if not config.rate_limiting_enabled:
            return await call_next(request)

        client_ip = (request.client.host if request.client else "unknown")
        path = request.url.path

        for prefix, (max_req, window_secs) in self._limits.items():
            if path.startswith(prefix):
                key = (client_ip, prefix)
                now = time.monotonic()
                with self._lock:
                    dq = self._windows[key]
                    cutoff = now - window_secs
                    while dq and dq[0] < cutoff:
                        dq.popleft()
                    if len(dq) >= max_req:
                        return JSONResponse(
                            {"detail": f"Rate limit exceeded ({max_req} req/{window_secs}s). Try again later."},
                            status_code=429,
                            headers={"Retry-After": str(window_secs)},
                        )
                    dq.append(now)
                break

        return await call_next(request)
