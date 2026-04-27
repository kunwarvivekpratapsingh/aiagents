from __future__ import annotations

import os
from dataclasses import dataclass, field


def _bool(key: str, default: bool = True) -> bool:
    return os.getenv(key, str(default)).lower() not in ("0", "false", "no", "off")


def _list(key: str, default: str = "") -> list[str]:
    raw = os.getenv(key, default)
    return [v.strip() for v in raw.split(",") if v.strip()]


@dataclass
class Config:
    # ── Anthropic ──────────────────────────────────────────────────────────
    anthropic_api_key: str = field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY", ""))
    model: str = field(default_factory=lambda: os.getenv("MODEL", "claude-sonnet-4-6"))

    # ── ChromaDB ───────────────────────────────────────────────────────────
    chroma_persist_dir: str = field(default_factory=lambda: os.getenv("CHROMA_DIR", "./chroma_db"))
    collection_name: str = field(default_factory=lambda: os.getenv("COLLECTION_NAME", "agentic_rag_docs"))

    # ── Pipeline ───────────────────────────────────────────────────────────
    max_iterations: int = field(default_factory=lambda: int(os.getenv("MAX_ITERATIONS", "3")))
    top_k_retrieval: int = field(default_factory=lambda: int(os.getenv("TOP_K", "5")))
    max_tokens: int = field(default_factory=lambda: int(os.getenv("MAX_TOKENS", "2048")))
    chunk_size: int = field(default_factory=lambda: int(os.getenv("CHUNK_SIZE", "400")))
    chunk_overlap: int = field(default_factory=lambda: int(os.getenv("CHUNK_OVERLAP", "60")))

    # ── API security ───────────────────────────────────────────────────────
    # Set API_KEY to a long random string to enable authentication.
    # Leave empty to disable (useful for local dev).
    api_key: str = field(default_factory=lambda: os.getenv("API_KEY", ""))

    # Comma-separated allowed CORS origins.
    # Use "*" ONLY for internal/dev deployments.
    cors_origins: list[str] = field(
        default_factory=lambda: _list("CORS_ORIGINS", "http://localhost:3000,http://localhost:8000")
    )

    # ── Rate limiting ──────────────────────────────────────────────────────
    rate_limiting_enabled: bool = field(default_factory=lambda: _bool("RATE_LIMITING", True))

    # ── Session memory ─────────────────────────────────────────────────────
    session_ttl_seconds: int = field(
        default_factory=lambda: int(os.getenv("SESSION_TTL_SECONDS", "3600"))
    )
    max_sessions: int = field(default_factory=lambda: int(os.getenv("MAX_SESSIONS", "1000")))

    # ── Logging ────────────────────────────────────────────────────────────
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))
    json_logs: bool = field(default_factory=lambda: _bool("JSON_LOGS", True))


config = Config()
