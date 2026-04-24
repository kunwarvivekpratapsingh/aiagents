"""Step 2: Rewrites the user query for better retrieval precision."""
import anthropic
from ..config import config

_SYSTEM = """You are a query optimization expert. Rewrite the given user query to make it:
1. More specific and information-rich
2. Expand abbreviations and acronyms
3. Include relevant technical context that is implied but unstated
4. Optimized for semantic search and document retrieval

Return ONLY the rewritten query — no explanation, no preamble, no quotes."""


class QueryRewriter:
    def __init__(self, client: anthropic.Anthropic):
        self.client = client

    def rewrite(self, query: str) -> str:
        response = self.client.messages.create(
            model=config.model,
            max_tokens=256,
            system=[{"type": "text", "text": _SYSTEM, "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": f"Rewrite this query:\n\n{query}"}],
        )
        rewritten = response.content[0].text.strip()
        return rewritten if rewritten else query
