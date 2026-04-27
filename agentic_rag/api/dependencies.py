"""FastAPI dependency injection — shared singletons for the request lifecycle."""
from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Request

from ..memory.conversation import SessionStore
from ..pipeline.rag_pipeline import AgenticRAGPipeline
from ..sources.vector_store import VectorStore


def get_vector_store(request: Request) -> VectorStore:
    return request.app.state.vector_store


def get_pipeline(request: Request) -> AgenticRAGPipeline:
    return request.app.state.pipeline


def get_session_store(request: Request) -> SessionStore:
    return request.app.state.session_store


VectorStoreDep   = Annotated[VectorStore,         Depends(get_vector_store)]
PipelineDep      = Annotated[AgenticRAGPipeline,  Depends(get_pipeline)]
SessionStoreDep  = Annotated[SessionStore,        Depends(get_session_store)]
