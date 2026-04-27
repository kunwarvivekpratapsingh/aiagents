"""
Conversation memory — tracks turns within a session and formats them
for injection into the query rewriter and LLM generator.
"""
from __future__ import annotations

import logging
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import ClassVar

logger = logging.getLogger(__name__)


@dataclass
class Turn:
    query: str
    response: str
    source_used: str
    timestamp: float = field(default_factory=time.time)


class ConversationMemory:
    """In-process session memory.  Thread-safe for concurrent API requests."""

    _MAX_TURNS: ClassVar[int] = 10
    _CONTEXT_TURNS: ClassVar[int] = 4   # turns injected into prompts
    _SUMMARY_LEN: ClassVar[int] = 300   # chars of response kept in context

    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        self._turns: list[Turn] = []
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def add(self, query: str, response: str, source_used: str = "") -> None:
        with self._lock:
            self._turns.append(Turn(query=query, response=response, source_used=source_used))
            if len(self._turns) > self._MAX_TURNS:
                self._turns = self._turns[-self._MAX_TURNS :]

    def clear(self) -> None:
        with self._lock:
            self._turns.clear()

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    @property
    def turns(self) -> list[Turn]:
        with self._lock:
            return list(self._turns)

    def is_empty(self) -> bool:
        return len(self._turns) == 0

    def format_for_rewriter(self) -> str:
        """Short context block for query rewriting (resolves coreferences)."""
        recent = self._turns[-self._CONTEXT_TURNS :]
        if not recent:
            return ""
        lines = ["Recent conversation (for context only):"]
        for t in recent:
            lines.append(f"  User: {t.query}")
            lines.append(f"  Assistant: {t.response[:100]}…")
        return "\n".join(lines)

    def as_messages(self) -> list[dict]:
        """Format as alternating user/assistant messages for multi-turn prompting."""
        recent = self._turns[-self._CONTEXT_TURNS :]
        messages: list[dict] = []
        for t in recent:
            messages.append({"role": "user", "content": t.query})
            messages.append({"role": "assistant", "content": t.response[: self._SUMMARY_LEN]})
        return messages

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "turns": [
                {
                    "query": t.query,
                    "response": t.response,
                    "source_used": t.source_used,
                    "timestamp": t.timestamp,
                }
                for t in self._turns
            ],
        }


class SessionStore:
    """
    Process-global registry of ConversationMemory objects.

    Eviction policy (applied on every get_or_create call):
      1. Expired sessions (idle > ttl_seconds) are removed first.
      2. If still over max_sessions, the least-recently-used session is evicted (LRU).
    """

    def __init__(
        self,
        max_sessions: int | None = None,
        ttl_seconds: int | None = None,
    ) -> None:
        from ..config import config as _cfg  # lazy import avoids circular dep
        self._max_sessions = max_sessions if max_sessions is not None else _cfg.max_sessions
        self._ttl_seconds  = ttl_seconds  if ttl_seconds  is not None else _cfg.session_ttl_seconds
        # OrderedDict preserves insertion order for LRU eviction
        self._store: OrderedDict[str, tuple[ConversationMemory, float]] = OrderedDict()
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def get_or_create(self, session_id: str) -> ConversationMemory:
        now = time.time()
        with self._lock:
            self._evict_expired(now)

            if session_id in self._store:
                mem, _ = self._store[session_id]
                # Refresh access time and move to end (most-recently-used)
                self._store.move_to_end(session_id)
                self._store[session_id] = (mem, now)
                return mem

            # Evict LRU entries until under capacity
            while len(self._store) >= self._max_sessions:
                evicted_id, _ = self._store.popitem(last=False)
                logger.debug("SessionStore: evicted LRU session %s", evicted_id)

            mem = ConversationMemory(session_id)
            self._store[session_id] = (mem, now)
            return mem

    def delete(self, session_id: str) -> bool:
        with self._lock:
            if session_id in self._store:
                del self._store[session_id]
                return True
            return False

    def clear_all(self) -> int:
        with self._lock:
            count = len(self._store)
            self._store.clear()
            return count

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def get(self, session_id: str) -> ConversationMemory | None:
        with self._lock:
            entry = self._store.get(session_id)
            if entry is None:
                return None
            mem, ts = entry
            if time.time() - ts > self._ttl_seconds:
                del self._store[session_id]
                return None
            return mem

    def list_sessions(self) -> list[str]:
        with self._lock:
            now = time.time()
            return [sid for sid, (_, ts) in self._store.items()
                    if now - ts <= self._ttl_seconds]

    def session_count(self) -> int:
        return len(self._store)

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _evict_expired(self, now: float) -> None:
        expired = [sid for sid, (_, ts) in self._store.items()
                   if now - ts > self._ttl_seconds]
        for sid in expired:
            del self._store[sid]
        if expired:
            logger.debug("SessionStore: expired %d session(s)", len(expired))
