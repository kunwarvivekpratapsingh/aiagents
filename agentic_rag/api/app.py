"""FastAPI application factory."""
from __future__ import annotations

import logging
import sys

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from ..config import config
from ..logging_config import configure_logging
from ..memory.conversation import SessionStore
from ..pipeline.rag_pipeline import AgenticRAGPipeline
from ..sources.vector_store import VectorStore, create_vector_store
from .middleware import (
    AccessLogMiddleware,
    AuthMiddleware,
    RateLimitMiddleware,
    RequestIDMiddleware,
)
from .routes import backup, chat, documents, health

logger = logging.getLogger(__name__)


# ── Lifespan ──────────────────────────────────────────────────────────────────

from contextlib import asynccontextmanager


@asynccontextmanager
async def _lifespan(app: FastAPI):
    configure_logging(level=config.log_level, json_logs=config.json_logs)

    if not config.anthropic_api_key:
        logger.error(
            "ANTHROPIC_API_KEY is not set. "
            "Set it in the environment or .env file before starting the server."
        )
        sys.exit(1)

    if not config.api_key:
        logger.warning(
            "API_KEY is not set — authentication is DISABLED. "
            "Set API_KEY to protect this server in production."
        )

    logger.info("Starting Agentic RAG API (model=%s)", config.model)
    vs = create_vector_store()
    app.state.vector_store  = vs
    app.state.pipeline      = AgenticRAGPipeline(vs)
    app.state.session_store = SessionStore()
    logger.info("Vector store ready — %d chunks indexed", vs.count())
    yield
    logger.info("Shutting down Agentic RAG API")


# ── Factory ───────────────────────────────────────────────────────────────────

def create_app() -> FastAPI:
    app = FastAPI(
        title="Agentic RAG API",
        description="Production-grade Agentic RAG with streaming, memory, and LLM reranking.",
        version="1.0.0",
        lifespan=_lifespan,
    )

    # ── Middleware (applied inner-first) ──────────────────────────────────
    # 1. Request ID — must be first so all subsequent middleware can read it
    app.add_middleware(RequestIDMiddleware)
    # 2. Access log — logs after the response is built
    app.add_middleware(AccessLogMiddleware)
    # 3. Rate limiting — before auth so even invalid keys count against the limit
    app.add_middleware(RateLimitMiddleware)
    # 4. Authentication
    app.add_middleware(AuthMiddleware)
    # 5. CORS — use explicit origin list; never combine wildcard with credentials
    origins = config.cors_origins
    if origins == ["*"]:
        # Wildcard is fine as long as credentials are not also allowed
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=False,
            allow_methods=["GET", "POST", "DELETE"],
            allow_headers=["X-API-Key", "X-Request-ID", "Content-Type", "Authorization"],
        )
    else:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_credentials=True,
            allow_methods=["GET", "POST", "DELETE"],
            allow_headers=["X-API-Key", "X-Request-ID", "Content-Type", "Authorization"],
        )

    # ── Routers ───────────────────────────────────────────────────────────
    app.include_router(health.router)
    app.include_router(chat.router)
    app.include_router(documents.router)
    app.include_router(backup.router)

    # ── Global exception handler — never leak tracebacks to clients ───────
    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception) -> JSONResponse:
        req_id = getattr(request.state, "request_id", "-")
        logger.exception("Unhandled exception req_id=%s %s %s", req_id, request.method, request.url)
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error", "req_id": req_id},
        )

    return app
