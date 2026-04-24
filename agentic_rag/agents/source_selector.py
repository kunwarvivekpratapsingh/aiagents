"""Step 5: Chooses the best retrieval source for the rewritten query."""
import json
import anthropic
from ..config import config

_SYSTEM = """You are a source-routing agent. Given a query, select the best information source.

Available sources:
- "vector_db"   : Pre-indexed Wikipedia knowledge base covering AI, ML, NLP, neural networks,
                  transformers, LLMs, RAG, reinforcement learning, knowledge graphs, etc.
- "web_search"  : Live internet search for current events, recent news, real-time data, prices,
                  specific URLs, or anything not covered by the knowledge base.
- "tools"       : Computational tools — use for arithmetic, date/time queries, unit conversions.

Respond with ONLY this JSON (no markdown fences):
{"source": "vector_db|web_search|tools", "reason": "one sentence"}"""


class SourceSelector:
    def __init__(self, client: anthropic.Anthropic):
        self.client = client

    def select(self, query: str) -> tuple[str, str]:
        """Returns (source_name, reason)."""
        response = self.client.messages.create(
            model=config.model,
            max_tokens=128,
            system=[{"type": "text", "text": _SYSTEM, "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": f"Query: {query}"}],
        )
        try:
            data = json.loads(response.content[0].text.strip())
            source = data.get("source", "vector_db")
            if source not in ("vector_db", "web_search", "tools"):
                source = "vector_db"
            return source, data.get("reason", "")
        except (json.JSONDecodeError, KeyError):
            return "vector_db", "defaulting to vector_db"
