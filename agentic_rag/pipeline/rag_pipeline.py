"""
Agentic RAG pipeline — orchestrates the 11-step loop from the architecture diagram.

  1  Query
  2  Rewrite Query             (QueryRewriter)
  3  Updated Query
  4  Need More Details?        (DetailChecker)  ──NO──▶ 8
       │YES
  5  Which Source?             (SourceSelector)
  6  Sources  (VectorDB / WebSearch / Tools)
  7  Retrieved Context + Updated Query
  8  LLM                       (Claude)
  9  Response
  10 Is the answer relevant?   (RelevanceChecker)  ──YES──▶ 11
       │NO  (back to 2, up to max_iterations)
  11 Final Response
"""
from __future__ import annotations

import logging
import time
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

EventName = str

# Event constants for UI callbacks
EV_REWRITING = "rewriting"
EV_CHECKING_DETAILS = "checking_details"
EV_SELECTING_SOURCE = "selecting_source"
EV_RETRIEVING = "retrieving"
EV_GENERATING = "generating"
EV_CHECKING_RELEVANCE = "checking_relevance"
EV_COMPLETE = "complete"
EV_RETRY = "retry"
EV_MAX_ITER = "max_iterations_reached"

_GENERATOR_SYSTEM = """\
You are a knowledgeable AI assistant. Answer the user's question accurately and concisely.

When context passages are provided:
  • Base your answer primarily on the provided context.
  • Cite sources inline as [Source: <title>] when referencing specific passages.
  • If the context is incomplete or contradicts your training, say so explicitly.
  • Do NOT fabricate information not present in the context or your verified knowledge.

When no context is provided:
  • Answer from your training knowledge.
  • Acknowledge uncertainty where appropriate.\
"""


@dataclass
class PipelineState:
    original_query: str
    current_query: str = ""
    source_used: str = ""
    retrieved_context: str = ""
    response: str = ""
    iteration: int = 0
    relevance_score: int = 0
    relevance_feedback: str = ""
    is_complete: bool = False
    error: str = ""
    # Audit trail — one entry per iteration
    trace: list[dict] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.current_query = self.original_query


class AgenticRAGPipeline:
    def __init__(self, vector_store: VectorStore) -> None:
        if not config.anthropic_api_key:
            raise ValueError("ANTHROPIC_API_KEY is not set in environment or .env file")
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
        """Execute the agentic RAG loop. Returns the final PipelineState."""
        state = PipelineState(original_query=query)

        def emit(event: EventName) -> None:
            if on_event:
                try:
                    on_event(event, state)
                except Exception as exc:  # never let UI errors crash the pipeline
                    logger.warning("on_event callback raised: %s", exc)

        while state.iteration < config.max_iterations and not state.is_complete:
            state.iteration += 1
            step_trace: dict = {"iteration": state.iteration}
            logger.info("[iter %d] query: %s", state.iteration, state.current_query[:120])

            # ── Step 2: Rewrite ──────────────────────────────────────────
            emit(EV_REWRITING)
            state.current_query = self._rewriter.rewrite(state.current_query)
            step_trace["rewritten_query"] = state.current_query

            # ── Step 4: Need retrieval? ──────────────────────────────────
            emit(EV_CHECKING_DETAILS)
            needs_retrieval, detail_reason = self._detail_checker.needs_retrieval(
                state.current_query
            )
            step_trace["needs_retrieval"] = needs_retrieval

            if needs_retrieval:
                # ── Step 5: Select source ────────────────────────────────
                emit(EV_SELECTING_SOURCE)
                source, source_reason = self._source_selector.select(state.current_query)
                state.source_used = source
                step_trace["source"] = source

                # ── Step 6–7: Retrieve ───────────────────────────────────
                emit(EV_RETRIEVING)
                state.retrieved_context = self._retrieve(state.current_query, source)
                step_trace["context_length"] = len(state.retrieved_context)
            else:
                state.source_used = "llm_knowledge"
                state.retrieved_context = ""
                step_trace["source"] = "llm_knowledge"

            # ── Step 8–9: Generate ───────────────────────────────────────
            emit(EV_GENERATING)
            state.response = self._generate(state.current_query, state.retrieved_context)
            step_trace["response_length"] = len(state.response)

            # ── Step 10: Relevance check ─────────────────────────────────
            emit(EV_CHECKING_RELEVANCE)
            relevant, score, feedback = self._relevance_checker.check(
                state.original_query, state.response
            )
            state.relevance_score = score
            state.relevance_feedback = feedback
            step_trace.update({"relevant": relevant, "score": score, "feedback": feedback})
            state.trace.append(step_trace)

            if relevant:
                # ── Step 11: Done ────────────────────────────────────────
                state.is_complete = True
                emit(EV_COMPLETE)
            else:
                emit(EV_RETRY)
                # Enrich the query with feedback for the next iteration
                state.current_query = (
                    f"{state.original_query}\n\n"
                    f"[Note: Previous answer scored {score}/10. Issue: {feedback}. "
                    "Please provide a more complete and accurate response.]"
                )

        if not state.is_complete:
            emit(EV_MAX_ITER)
            logger.warning(
                "Max iterations (%d) reached without a high-quality answer",
                config.max_iterations,
            )

        return state

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _retrieve(self, query: str, source: str) -> str:
        if source == "vector_db":
            return self._retrieve_vector(query)
        if source == "web_search":
            return self._retrieve_web(query)
        if source == "tools":
            return self._tools.execute(query)
        logger.warning("Unknown source %r; falling back to vector_db", source)
        return self._retrieve_vector(query)

    def _retrieve_vector(self, query: str) -> str:
        try:
            results = self._vs.search(query)
        except Exception as exc:
            logger.error("VectorStore search failed: %s", exc)
            return "Vector search temporarily unavailable."

        if not results:
            return "No relevant documents found in the knowledge base."

        parts: list[str] = []
        for r in results:
            title = r["metadata"].get("title", "Unknown")
            score = r["relevance_score"]
            parts.append(f"[Source: {title} | relevance: {score:.3f}]\n{r['content']}")
        return "\n\n---\n\n".join(parts)

    def _retrieve_web(self, query: str) -> str:
        results = self._web.search(query)
        if not results:
            return "No web search results found."
        parts: list[str] = []
        for r in results:
            title = r["metadata"].get("title", "Web result")
            url = r["metadata"].get("source", "")
            parts.append(f"[Source: {title} | {url}]\n{r['content']}")
        return "\n\n---\n\n".join(parts)

    def _generate(self, query: str, context: str, max_retries: int = 3) -> str:
        user_content = (
            f"Context from knowledge base:\n\n{context}\n\n---\n\nQuestion: {query}"
            if context
            else f"Question: {query}"
        )

        for attempt in range(max_retries):
            try:
                resp = self._client.messages.create(
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
                return resp.content[0].text.strip()
            except anthropic.RateLimitError:
                wait = 2 ** attempt
                logger.warning("Generator rate-limited; retrying in %ds", wait)
                time.sleep(wait)
            except anthropic.APIError as exc:
                logger.error("Generator API error (attempt %d): %s", attempt + 1, exc)
                if attempt == max_retries - 1:
                    return f"Error generating response: {exc}"
                time.sleep(2 ** attempt)
        return "Failed to generate a response after multiple attempts."
