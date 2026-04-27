"""DuckDuckGo-powered live web search (no API key required)."""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class WebSearch:
    def search(self, query: str, max_results: int = 5) -> list[dict[str, Any]]:
        try:
            from duckduckgo_search import DDGS

            results: list[dict[str, Any]] = []
            with DDGS() as ddgs:
                for r in ddgs.text(query, max_results=max_results):
                    results.append(
                        {
                            "content": f"{r.get('title', '')}\n{r.get('body', '')}",
                            "metadata": {
                                "source": r.get("href", ""),
                                "title": r.get("title", "Web result"),
                            },
                            "relevance_score": 1.0,
                        }
                    )
            return results
        except Exception as exc:
            logger.warning("Web search failed: %s", exc)
            return [
                {
                    "content": f"Web search unavailable: {exc}",
                    "metadata": {"source": "error", "title": "Error"},
                    "relevance_score": 0.0,
                }
            ]
