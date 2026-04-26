"""
Conversation memory — tracks turns within a session and formats them
for injection into the query rewriter and LLM generator.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import ClassVar


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
            # Truncate to avoid bloating context window
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
    """Process-global registry of ConversationMemory objects keyed by session_id."""

    def __init__(self) -> None:
        self._store: dict[str, ConversationMemory] = {}
        self._lock = threading.Lock()

    def get_or_create(self, session_id: str) -> ConversationMemory:
        with self._lock:
            if session_id not in self._store:
                self._store[session_id] = ConversationMemory(session_id)
            return self._store[session_id]

    def get(self, session_id: str) -> ConversationMemory | None:
        return self._store.get(session_id)

    def delete(self, session_id: str) -> bool:
        with self._lock:
            if session_id in self._store:
                del self._store[session_id]
                return True
            return False

    def list_sessions(self) -> list[str]:
        return list(self._store.keys())
