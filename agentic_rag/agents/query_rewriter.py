"""Step 2: Rewrites the user query for better retrieval precision."""
from __future__ import annotations

import logging
import time

import anthropic

from ..config import config

logger = logging.getLogger(__name__)

_SYSTEM = """\
You are a query optimization expert. Rewrite the given user query to make it:
1. More specific and information-rich for semantic document retrieval
2. Expand abbreviations and acronyms
3. Include relevant technical context that is implied but unstated
4. Remove filler phrases that don't aid retrieval

Return ONLY the rewritten query — no explanation, no preamble, no quotes.\
"""


class QueryRewriter:
    def __init__(self, client: anthropic.Anthropic) -> None:
        self._client = client

    def rewrite(self, query: str, max_retries: int = 3) -> str:
        """Rewrite *query* for retrieval. Returns original on repeated failure."""
        for attempt in range(max_retries):
            try:
                resp = self._client.messages.create(
                    model=config.model,
                    max_tokens=256,
                    system=[
                        {"type": "text", "text": _SYSTEM, "cache_control": {"type": "ephemeral"}}
                    ],
                    messages=[
                        {"role": "user", "content": f"Rewrite this query:\n\n{query}"}
                    ],
                )
                rewritten = resp.content[0].text.strip()
                logger.debug("QueryRewriter: %r → %r", query, rewritten)
                return rewritten if rewritten else query
            except anthropic.RateLimitError:
                wait = 2 ** attempt
                logger.warning("QueryRewriter rate-limited; retrying in %ds", wait)
                time.sleep(wait)
            except anthropic.APIError as exc:
                logger.error("QueryRewriter API error (attempt %d): %s", attempt + 1, exc)
                if attempt == max_retries - 1:
                    break
                time.sleep(2 ** attempt)
        logger.warning("QueryRewriter giving up; returning original query")
        return query
