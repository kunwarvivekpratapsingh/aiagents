"""
LLM-based reranker — uses a single Claude call to re-order retrieved
documents by relevance, significantly improving retrieval precision
without requiring any additional model downloads.
"""
from __future__ import annotations

import json
import logging
import re
import time
from typing import Any

import anthropic

from ..config import config

logger = logging.getLogger(__name__)

_SYSTEM = """\
You are a document relevance ranker. Given a query and a list of document excerpts,
rank them strictly by how well they answer the query.

Return ONLY a JSON array of 1-based document indices ordered from most to least relevant.
Example for 5 docs: [3, 1, 5, 2, 4]
No explanation. No markdown. Just the JSON array.\
"""


class LLMReranker:
    """Re-ranks a list of retrieved chunks using a single LLM inference call."""

    def __init__(self, client: anthropic.Anthropic) -> None:
        self._client = client

    def rerank(
        self,
        query: str,
        documents: list[dict[str, Any]],
        max_retries: int = 2,
    ) -> list[dict[str, Any]]:
        """
        Return *documents* sorted by relevance to *query*.
        Falls back to original order on any failure.
        """
        if len(documents) <= 2:
            return documents

        numbered = "\n\n".join(
            f"[Doc {i + 1}]: {d['content'][:350]}"
            for i, d in enumerate(documents)
        )
        user_msg = f"Query: {query}\n\nDocuments:\n{numbered}"

        for attempt in range(max_retries):
            try:
                resp = self._client.messages.create(
                    model=config.model,
                    max_tokens=128,
                    system=[
                        {"type": "text", "text": _SYSTEM, "cache_control": {"type": "ephemeral"}}
                    ],
                    messages=[{"role": "user", "content": user_msg}],
                )
                raw = resp.content[0].text.strip()
                m = re.search(r"\[[\d,\s]+\]", raw)
                if not m:
                    logger.warning("Reranker: no JSON array in response — using original order")
                    return documents

                ranking: list[int] = json.loads(m.group())
                reranked: list[dict] = []
                seen: set[int] = set()
                for idx in ranking:
                    if 1 <= idx <= len(documents) and idx not in seen:
                        reranked.append(documents[idx - 1])
                        seen.add(idx)
                # Append any docs the LLM omitted (shouldn't happen, but safe)
                for i, doc in enumerate(documents):
                    if i + 1 not in seen:
                        reranked.append(doc)

                logger.debug("Reranker: order %s", ranking)
                return reranked

            except (json.JSONDecodeError, ValueError) as exc:
                logger.warning("Reranker parse error (attempt %d): %s", attempt + 1, exc)
                return documents
            except anthropic.RateLimitError:
                wait = 2 ** attempt
                logger.warning("Reranker rate-limited; retrying in %ds", wait)
                time.sleep(wait)
            except anthropic.APIError as exc:
                logger.warning("Reranker API error (attempt %d): %s", attempt + 1, exc)
                if attempt == max_retries - 1:
                    break
                time.sleep(2 ** attempt)

        return documents
