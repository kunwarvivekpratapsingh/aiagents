"""
Agentic RAG pipeline — orchestrates the 11-step loop from the architecture diagram:

  1  Query
  2  Rewrite Query
  3  Updated Query
  4  Need More Details?  ──NO──▶  8 (LLM, no context)
       │YES
  5  Which Source?
  6  Sources  (Vector DB / Web / Tools)
  7  Retrieved Context + Updated Query
  8  LLM
  9  Response
  10 Is the answer relevant?  ──YES──▶  11 Final Response
       │NO  (loop back to 2, up to max_iterations)
  11 Final Response
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable

import anthropic

from ..agents.detail_checker import DetailChecker
from ..agents.query_rewriter import QueryRewriter
from ..agents.relevance_checker import RelevanceChecker
from ..agents.source_selector import SourceSelector
from ..config import config
from ..sources.tools import Tools
from ..sources.vector_store import VectorStore
from ..sources.web_search import WebSearch

logger = logging.getLogger(__name__)

# Callback event names emitted during pipeline execution
EventName = str


@dataclass
class PipelineState:
    """Mutable state threaded through each iteration."""

    original_query: str
    current_query: str = ""
    source_used: str = ""
    retrieved_context: str = ""
    response: str = ""
    iteration: int = 0
    relevance_score: int = 0
    relevance_feedback: str = ""
    is_complete: bool = False

    def __post_init__(self) -> None:
        self.current_query = self.original_query


# Event name constants (used by main.py for display)
EV_REWRITING = "rewriting"
EV_CHECKING_DETAILS = "checking_details"
EV_SELECTING_SOURCE = "selecting_source"
EV_RETRIEVING = "retrieving"
EV_GENERATING = "generating"
EV_CHECKING_RELEVANCE = "checking_relevance"
EV_COMPLETE = "complete"
EV_RETRY = "retry"


_GENERATOR_SYSTEM = """\
You are a knowledgeable AI assistant. Answer the user's question accurately and concisely.

When context passages are provided:
  • Prioritise information from the context.
  • Cite sources inline as [Source: <title>] when referencing specific passages.
  • If the context is incomplete, supplement with your training knowledge and say so.

When no context is provided, answer from your training knowledge directly.\
"""


class AgenticRAGPipeline:
    def __init__(self, vector_store: VectorStore) -> None:
        self._client = anthropic.Anthropic(api_key=config.anthropic_api_key)
        self._vs = vector_store
        self._web = WebSearch()
        self._tools = Tools()

        self._rewriter = QueryRewriter(self._client)
        self._detail_checker = DetailChecker(self._client)
        self._source_selector = SourceSelector(self._client)
        self._relevance_checker = RelevanceChecker(self._client)

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(
        self,
        query: str,
        on_event: Callable[[EventName, PipelineState], None] | None = None,
    ) -> PipelineState:
        """
        Execute the agentic RAG loop and return the final PipelineState.

        *on_event* is called after each significant step so callers can
        render progress without coupling the pipeline to any UI library.
        """
        state = PipelineState(original_query=query)

        def emit(event: EventName) -> None:
            if on_event:
                on_event(event, state)

        while state.iteration < config.max_iterations and not state.is_complete:
            state.iteration += 1
            logger.info("[iter %d] start — query: %s", state.iteration, state.current_query)

            # ── Step 2: Rewrite ──────────────────────────────────────────
            emit(EV_REWRITING)
            state.current_query = self._rewriter.rewrite(state.current_query)
            logger.info("[iter %d] rewritten: %s", state.iteration, state.current_query)

            # ── Step 4: Need more details? ───────────────────────────────
            emit(EV_CHECKING_DETAILS)
            needs_retrieval, detail_reason = self._detail_checker.needs_retrieval(state.current_query)
            logger.info("[iter %d] needs_retrieval=%s (%s)", state.iteration, needs_retrieval, detail_reason)

            if needs_retrieval:
                # ── Step 5: Which source? ────────────────────────────────
                emit(EV_SELECTING_SOURCE)
                source, source_reason = self._source_selector.select(state.current_query)
                state.source_used = source
                logger.info("[iter %d] source=%s (%s)", state.iteration, source, source_reason)

                # ── Step 6–7: Retrieve context ───────────────────────────
                emit(EV_RETRIEVING)
                state.retrieved_context = self._retrieve(state.current_query, source)
            else:
                state.source_used = "llm_knowledge"
                state.retrieved_context = ""
                logger.info("[iter %d] skipping retrieval — answering from LLM knowledge", state.iteration)

            # ── Step 8–9: Generate response ──────────────────────────────
            emit(EV_GENERATING)
            state.response = self._generate(state.current_query, state.retrieved_context)
            logger.info("[iter %d] response length=%d chars", state.iteration, len(state.response))

            # ── Step 10: Relevance check ─────────────────────────────────
            emit(EV_CHECKING_RELEVANCE)
            relevant, score, feedback = self._relevance_checker.check(
                state.original_query, state.response
            )
            state.relevance_score = score
            state.relevance_feedback = feedback
            logger.info("[iter %d] relevant=%s score=%d feedback: %s", state.iteration, relevant, score, feedback)

            if relevant:
                # ── Step 11: Final response ──────────────────────────────
                state.is_complete = True
                emit(EV_COMPLETE)
            else:
                # Loop back to step 2 with enriched query
                emit(EV_RETRY)
                state.current_query = (
                    f"{state.original_query}\n"
                    f"[Previous attempt (score {score}/10) was insufficient: {feedback}. "
                    "Please provide a more complete answer.]"
                )

        return state

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _retrieve(self, query: str, source: str) -> str:
        if source == "vector_db":
            results = self._vs.search(query)
            if not results:
                return "No relevant documents found in the knowledge base."
            parts = []
            for r in results:
                title = r["metadata"].get("title", "Unknown")
                score = r["relevance_score"]
                parts.append(f"[Source: {title} | relevance: {score}]\n{r['content']}")
            return "\n\n---\n\n".join(parts)

        if source == "web_search":
            results = self._web.search(query)
            if not results:
                return "No web search results found."
            parts = []
            for r in results:
                title = r["metadata"].get("title", "Web result")
                url = r["metadata"].get("source", "")
                parts.append(f"[Source: {title} | {url}]\n{r['content']}")
            return "\n\n---\n\n".join(parts)

        if source == "tools":
            return self._tools.execute(query)

        return ""

    def _generate(self, query: str, context: str) -> str:
        if context:
            user_content = (
                f"Context from knowledge base:\n\n{context}\n\n"
                "---\n\n"
                f"Question: {query}"
            )
        else:
            user_content = f"Question: {query}"

        response = self._client.messages.create(
            model=config.model,
            max_tokens=config.max_tokens,
            system=[
                {
                    "type": "text",
                    "text": _GENERATOR_SYSTEM,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=[{"role": "user", "content": user_content}],
        )
        return response.content[0].text.strip()
