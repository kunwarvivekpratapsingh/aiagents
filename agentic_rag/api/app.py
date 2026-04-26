"""FastAPI application factory."""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from ..config import config
from ..memory.conversation import SessionStore
from ..pipeline.rag_pipeline import AgenticRAGPipeline
from ..sources.vector_store import VectorStore
from .routes import chat, documents, health

logger = logging.getLogger(__name__)


@asynccontextmanager
async def _lifespan(app: FastAPI):
    """Initialize shared singletons on startup; clean up on shutdown."""
    logger.info("Starting Agentic RAG API (model=%s)", config.model)
    vs = VectorStore()
    app.state.vector_store  = vs
    app.state.pipeline      = AgenticRAGPipeline(vs)
    app.state.session_store = SessionStore()
    logger.info("Vector store ready — %d chunks indexed", vs.count())
    yield
    logger.info("Shutting down Agentic RAG API")


def create_app() -> FastAPI:
    app = FastAPI(
        title="Agentic RAG API",
        description="Production-grade Agentic RAG with streaming, memory, and LLM reranking.",
        version="1.0.0",
        lifespan=_lifespan,
    )

    # CORS — tighten origins for production
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Routers
    app.include_router(health.router)
    app.include_router(chat.router)
    app.include_router(documents.router)

    # Global exception handler — never leak tracebacks to clients
    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unhandled exception on %s %s", request.method, request.url)
        return JSONResponse(status_code=500, content={"detail": "Internal server error"})

    return app
