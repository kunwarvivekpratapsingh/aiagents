"""Health and readiness endpoints."""
from __future__ import annotations

from fastapi import APIRouter

from ..dependencies import PipelineDep, VectorStoreDep
from ..models import HealthResponse, ReadyResponse
from ...config import config

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health(vs: VectorStoreDep) -> HealthResponse:
    return HealthResponse(
        status="ok",
        model=config.model,
        vector_store_docs=vs.count(),
    )


@router.get("/health/ready", response_model=ReadyResponse)
async def ready(vs: VectorStoreDep, pipeline: PipelineDep) -> ReadyResponse:
    vs_ok = False
    api_ok = bool(config.anthropic_api_key)
    try:
        vs.count()
        vs_ok = True
    except Exception:
        pass

    checks = {"vector_store": vs_ok, "anthropic_api_key": api_ok}
    return ReadyResponse(ready=all(checks.values()), checks=checks)
