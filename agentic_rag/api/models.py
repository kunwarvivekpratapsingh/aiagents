"""Pydantic request/response models for the REST API."""
from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, Field, field_validator


# ── Injection / prompt-attack detection ──────────────────────────────────────

_INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?previous\s+instructions",
    r"disregard\s+(all\s+)?instructions",
    r"you\s+are\s+now\s+(?:a|an)\s+",
    r"act\s+as\s+(?:a|an)\s+",
    r"jailbreak",
    r"DAN\s+mode",
    r"system\s+prompt",
    r"<\s*script",
    r"\{\{.*\}\}",          # template injection
    r"\$\{.*\}",            # JS template literal injection
]

_INJECTION_RE = re.compile(
    "|".join(_INJECTION_PATTERNS), re.IGNORECASE | re.DOTALL
)


def _check_injection(value: str) -> str:
    if _INJECTION_RE.search(value):
        raise ValueError("Query contains potentially unsafe content")
    return value


# ── Request models ────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=4096)
    session_id: str | None = Field(default=None, max_length=128)

    @field_validator("query")
    @classmethod
    def no_injection(cls, v: str) -> str:
        return _check_injection(v.strip())

    @field_validator("session_id")
    @classmethod
    def sanitize_session(cls, v: str | None) -> str | None:
        if v is None:
            return v
        v = v.strip()
        if not re.match(r"^[a-zA-Z0-9_\-]{1,128}$", v):
            raise ValueError("session_id must be alphanumeric/dash/underscore, 1–128 chars")
        return v


# ── Response models ───────────────────────────────────────────────────────────

class ChatResponse(BaseModel):
    response: str
    source: str
    score: int
    iterations: int
    complete: bool
    session_id: str | None = None


class StreamEvent(BaseModel):
    type: str
    data: dict[str, Any]


class DocumentInfo(BaseModel):
    source: str
    chunk_count: int


class DocumentListResponse(BaseModel):
    documents: list[DocumentInfo]
    total_chunks: int


class UploadResponse(BaseModel):
    filename: str
    chunks_added: int
    message: str


class DeleteResponse(BaseModel):
    deleted: bool
    message: str


class HealthResponse(BaseModel):
    status: str
    model: str
    vector_store_docs: int


class ReadyResponse(BaseModel):
    ready: bool
    checks: dict[str, bool]
