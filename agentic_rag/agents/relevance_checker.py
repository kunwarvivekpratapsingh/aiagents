"""Step 10: Evaluates whether the generated response sufficiently answers the original query."""
import json
import anthropic
from ..config import config

_SYSTEM = """You are a response quality evaluator. Assess whether the response adequately answers the query.

Scoring rubric (0-10):
  9-10  : Directly answers the query with accurate, complete information
  6-8   : Mostly answers the query; minor gaps or tangential info
  3-5   : Partially relevant; significant gaps or inaccuracies
  0-2   : Off-topic or factually wrong

Respond with ONLY this JSON (no markdown fences):
{"is_relevant": true|false, "score": <int 0-10>, "feedback": "one sentence on what is missing or wrong"}

Set is_relevant to true when score >= 6."""


class RelevanceChecker:
    def __init__(self, client: anthropic.Anthropic):
        self.client = client

    def check(self, original_query: str, response: str) -> tuple[bool, int, str]:
        """Returns (is_relevant, score, feedback)."""
        payload = self.client.messages.create(
            model=config.model,
            max_tokens=200,
            system=[{"type": "text", "text": _SYSTEM, "cache_control": {"type": "ephemeral"}}],
            messages=[
                {
                    "role": "user",
                    "content": f"Query: {original_query}\n\nResponse:\n{response}",
                }
            ],
        )
        try:
            data = json.loads(payload.content[0].text.strip())
            score = int(data.get("score", 5))
            return data.get("is_relevant", score >= 6), score, data.get("feedback", "")
        except (json.JSONDecodeError, KeyError, ValueError):
            return True, 5, "could not parse evaluator response"
