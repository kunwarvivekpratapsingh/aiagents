"""Step 4: Decides whether external retrieval is needed or the LLM can answer directly."""
from __future__ import annotations

import json
import logging
import time

import anthropic

from ..config import config

logger = logging.getLogger(__name__)

_SYSTEM = """\
You are a knowledge-gap assessment agent. Decide if the query requires external
information retrieval or can be answered directly from broad training knowledge.

Respond with ONLY valid JSON (no markdown fences):
{"needs_retrieval": true|false, "reason": "one sentence"}

Return needs_retrieval TRUE when the query:
- Asks for specific facts, statistics, citations, or recent events
- Requires recent, proprietary, or domain-specific knowledge
- Mentions specific named entities, products, papers, or events

Return needs_retrieval FALSE when the query:
- Asks for explanations of well-known general concepts
- Is pure logic, math, or definitional
- Can be answered confidently from broad training data\
"""


class DetailChecker:
    def __init__(self, client: anthropic.Anthropic) -> None:
        self._client = client

    def needs_retrieval(self, query: str, max_retries: int = 3) -> tuple[bool, str]:
        """Returns (needs_retrieval, reason). Defaults to True on failure."""
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
                result = bool(data.get("needs_retrieval", True))
                reason = data.get("reason", "")
                logger.debug("DetailChecker: needs_retrieval=%s (%s)", result, reason)
                return result, reason
            except json.JSONDecodeError as exc:
                logger.warning("DetailChecker JSON parse error (attempt %d): %s", attempt + 1, exc)
                return True, "JSON parse error — defaulting to retrieval"
            except anthropic.RateLimitError:
                wait = 2 ** attempt
                logger.warning("DetailChecker rate-limited; retrying in %ds", wait)
                time.sleep(wait)
            except anthropic.APIError as exc:
                logger.error("DetailChecker API error (attempt %d): %s", attempt + 1, exc)
                if attempt == max_retries - 1:
                    break
                time.sleep(2 ** attempt)
        return True, "API error — defaulting to retrieval"
