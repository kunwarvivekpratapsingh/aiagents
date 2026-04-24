"""Step 5: Chooses the best retrieval source for the rewritten query."""
from __future__ import annotations

import json
import logging
import time

import anthropic

from ..config import config

logger = logging.getLogger(__name__)

_VALID_SOURCES = frozenset({"vector_db", "web_search", "tools"})

_SYSTEM = """\
You are a source-routing agent. Select the best information source for the query.

Available sources:
- "vector_db"   : Pre-indexed knowledge base covering AI, ML, NLP, neural networks,
                  Transformers, LLMs, RAG, RL, knowledge graphs, vector databases,
                  prompt engineering, embeddings, deep learning, and AI safety.
                  Use for any conceptual, technical, or factual AI/ML question.
- "web_search"  : Live internet search. Use for current events, very recent news,
                  real-time prices, specific URLs, or topics outside the knowledge base.
- "tools"       : Computational tools. Use for arithmetic, date/time queries,
                  unit conversions, or any calculation.

Respond with ONLY valid JSON (no markdown fences):
{"source": "vector_db|web_search|tools", "reason": "one sentence"}\
"""


class SourceSelector:
    def __init__(self, client: anthropic.Anthropic) -> None:
        self._client = client

    def select(self, query: str, max_retries: int = 3) -> tuple[str, str]:
        """Returns (source_name, reason). Defaults to 'vector_db' on failure."""
        for attempt in range(max_retries):
            try:
                resp = self._client.messages.create(
                    model=config.model,
                    max_tokens=128,
                    system=[
                        {"type": "text", "text": _SYSTEM, "cache_control": {"type": "ephemeral"}}
                    ],
                    messages=[{"role": "user", "content": f"Query: {query}"}],
                )
                raw = resp.content[0].text.strip()
                data = json.loads(raw)
                source = data.get("source", "vector_db")
                if source not in _VALID_SOURCES:
                    logger.warning("SourceSelector returned unknown source %r; using vector_db", source)
                    source = "vector_db"
                reason = data.get("reason", "")
                logger.debug("SourceSelector: %s (%s)", source, reason)
                return source, reason
            except json.JSONDecodeError as exc:
                logger.warning("SourceSelector JSON parse error (attempt %d): %s", attempt + 1, exc)
                return "vector_db", "JSON parse error — defaulting to vector_db"
            except anthropic.RateLimitError:
                wait = 2 ** attempt
                logger.warning("SourceSelector rate-limited; retrying in %ds", wait)
                time.sleep(wait)
            except anthropic.APIError as exc:
                logger.error("SourceSelector API error (attempt %d): %s", attempt + 1, exc)
                if attempt == max_retries - 1:
                    break
                time.sleep(2 ** attempt)
        return "vector_db", "API error — defaulting to vector_db"
