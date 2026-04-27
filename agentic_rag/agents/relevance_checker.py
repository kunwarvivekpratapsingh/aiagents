"""Step 10: Evaluates whether the generated response sufficiently answers the original query."""
from __future__ import annotations

import json
import logging
import time

import anthropic

from ..config import config

logger = logging.getLogger(__name__)

_SYSTEM = """\
You are a strict response quality evaluator. Assess whether the response adequately
answers the original query.

Scoring rubric (0–10):
  9–10 : Directly and completely answers the query with accurate information
  6–8  : Mostly answers; minor gaps or slightly tangential
  3–5  : Partially relevant; significant gaps or potential inaccuracies
  0–2  : Off-topic, factually wrong, or refuses to answer without good reason

Respond with ONLY valid JSON (no markdown fences):
{"is_relevant": true|false, "score": <int 0-10>, "feedback": "one sentence on what is missing"}

Set is_relevant to true when score >= 6.\
"""


class RelevanceChecker:
    def __init__(self, client: anthropic.Anthropic) -> None:
        self._client = client

    def check(
        self, original_query: str, response: str, max_retries: int = 3
    ) -> tuple[bool, int, str]:
        """Returns (is_relevant, score, feedback). Defaults to (True, 5, '') on failure."""
        for attempt in range(max_retries):
            try:
                payload = self._client.messages.create(
                    model=config.model,
                    max_tokens=200,
                    system=[
                        {"type": "text", "text": _SYSTEM, "cache_control": {"type": "ephemeral"}}
                    ],
                    messages=[
                        {
                            "role": "user",
                            "content": f"Query: {original_query}\n\nResponse:\n{response}",
                        }
                    ],
                )
                raw = payload.content[0].text.strip()
                data = json.loads(raw)
                score = int(data.get("score", 5))
                is_relevant = bool(data.get("is_relevant", score >= 6))
                feedback = data.get("feedback", "")
                logger.debug(
                    "RelevanceChecker: relevant=%s score=%d feedback=%s",
                    is_relevant, score, feedback,
                )
                return is_relevant, score, feedback
            except json.JSONDecodeError as exc:
                logger.warning("RelevanceChecker JSON parse error (attempt %d): %s", attempt + 1, exc)
                return True, 5, "JSON parse error — accepting response"
            except ValueError:
                return True, 5, "score parse error — accepting response"
            except anthropic.RateLimitError:
                wait = 2 ** attempt
                logger.warning("RelevanceChecker rate-limited; retrying in %ds", wait)
                time.sleep(wait)
            except anthropic.APIError as exc:
                logger.error("RelevanceChecker API error (attempt %d): %s", attempt + 1, exc)
                if attempt == max_retries - 1:
                    break
                time.sleep(2 ** attempt)
        return True, 5, "API error — accepting response"
