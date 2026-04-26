"""Chat endpoints — sync and SSE streaming."""
from __future__ import annotations

import json
import uuid

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from ..dependencies import PipelineDep, SessionStoreDep
from ..models import ChatRequest, ChatResponse

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("", response_model=ChatResponse)
async def chat(body: ChatRequest, pipeline: PipelineDep, sessions: SessionStoreDep) -> ChatResponse:
    """Synchronous RAG query."""
    memory = None
    session_id = body.session_id
    if session_id:
        memory = sessions.get_or_create(session_id)

    state = pipeline.run(body.query, memory=memory)

    if state.error:
        raise HTTPException(status_code=500, detail=state.error)

    return ChatResponse(
        response=state.response,
        source=state.source_used,
        score=state.relevance_score,
        iterations=state.iteration,
        complete=state.is_complete,
        session_id=session_id,
    )


@router.post("/stream")
async def chat_stream(
    body: ChatRequest,
    pipeline: PipelineDep,
    sessions: SessionStoreDep,
) -> StreamingResponse:
    """Token-streaming RAG query via Server-Sent Events."""
    memory = None
    session_id = body.session_id or str(uuid.uuid4())
    memory = sessions.get_or_create(session_id)

    def _event_stream():
        # Send session_id first so client knows which session this is
        yield _sse({"type": "session", "session_id": session_id})
        for event in pipeline.stream(body.query, memory=memory):
            yield _sse(event)

    return StreamingResponse(
        _event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/sessions/{session_id}")
async def get_session(session_id: str, sessions: SessionStoreDep) -> dict:
    """Return conversation history for a session."""
    memory = sessions.get(session_id)
    if memory is None:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")
    return memory.to_dict()


@router.delete("/sessions/{session_id}")
async def delete_session(session_id: str, sessions: SessionStoreDep) -> dict:
    """Delete a session and its conversation history."""
    deleted = sessions.delete(session_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")
    return {"deleted": True, "session_id": session_id}


@router.get("/sessions")
async def list_sessions(sessions: SessionStoreDep) -> dict:
    """List all active session IDs."""
    return {"sessions": sessions.list_sessions()}


# ── SSE helper ────────────────────────────────────────────────────────────────

def _sse(data: dict) -> str:
    return f"data: {json.dumps(data)}\n\n"
