"""Step 4: Decides whether external retrieval is needed or LLM can answer directly."""
import json
import anthropic
from ..config import config

_SYSTEM = """You are a knowledge-gap assessment agent. Decide if the query requires external information retrieval.

Respond with ONLY this JSON (no markdown fences):
{"needs_retrieval": true|false, "reason": "one sentence"}

Return needs_retrieval: TRUE when the query:
- Asks for specific facts, statistics, or citations
- Requires recent or domain-specific knowledge
- Mentions specific named entities, products, or events

Return needs_retrieval: FALSE when the query:
- Asks for general explanations of well-known concepts
- Is a pure logic / math problem
- Can be answered confidently from broad training data"""


class DetailChecker:
    def __init__(self, client: anthropic.Anthropic):
        self.client = client

    def needs_retrieval(self, query: str) -> tuple[bool, str]:
        """Returns (needs_retrieval, reason)."""
        response = self.client.messages.create(
            model=config.model,
            max_tokens=128,
            system=[{"type": "text", "text": _SYSTEM, "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": f"Query: {query}"}],
        )
        try:
            data = json.loads(response.content[0].text.strip())
            return bool(data.get("needs_retrieval", True)), data.get("reason", "")
        except (json.JSONDecodeError, KeyError):
            return True, "defaulting to retrieval"
