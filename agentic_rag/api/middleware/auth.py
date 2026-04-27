"""API key authentication middleware.

Checks the X-API-Key header (or Authorization: Bearer <key>).
If config.api_key is empty the middleware is disabled — auth is off in dev.
Health endpoints are always public.
"""
from __future__ import annotations

import secrets

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

from ...config import config

_PUBLIC_PATHS = frozenset({
    "/health",
    "/health/ready",
    "/docs",
    "/redoc",
    "/openapi.json",
})


class AuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if not config.api_key or request.url.path in _PUBLIC_PATHS:
            return await call_next(request)

        key = (
            request.headers.get("X-API-Key")
            or _extract_bearer(request)
        )
        if not key or not secrets.compare_digest(key.encode(), config.api_key.encode()):
            return JSONResponse(
                {"detail": "Invalid or missing API key. Pass it as 'X-API-Key' header."},
                status_code=401,
            )
        return await call_next(request)


def _extract_bearer(request: Request) -> str | None:
    auth = request.headers.get("Authorization", "")
    if auth.lower().startswith("bearer "):
        return auth[7:]
    return None
