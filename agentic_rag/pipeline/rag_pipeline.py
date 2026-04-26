"""
Agentic RAG pipeline — 11-step loop with streaming, conversation memory,
and LLM reranking.

Modes:
  run(query, memory)    → PipelineState          (synchronous, for tests / CLI)
  stream(query, memory) → Generator[dict, None]  (streaming, for API / real-time UI)

Event types emitted by stream():
  {"type": "step",       "step": str, "label": str}
  {"type": "token",      "delta": str}
  {"type": "relevance",  "score": int, "relevant": bool, "feedback": str}
  {"type": "retry",      "iteration": int, "feedback": str}
  {"type": "done",       "response": str, "source": str, "score": int,
                          "iterations": int, "complete": bool}
  {"type": "error",      "message": str}
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Callable, Generator

import anthropic

from ..agents.detail_checker import DetailChecker
from ..agents.query_rewriter import QueryRewriter
from ..agents.relevance_checker import RelevanceChecker
from ..agents.source_selector import SourceSelector
from ..config import config
from ..sources.reranker import LLMReranker
from ..sources.tools import Tools
from ..sources.vector_store import VectorStore
from ..sources.web_search import WebSearch

logger = logging.getLogger(__name__)

# ── Event name constants ──────────────────────────────────────────────────────
EV_REWRITING          = "rewriting"
EV_CHECKING_DETAILS   = "checking_details"
EV_SELECTING_SOURCE   = "selecting_source"
EV_RETRIEVING         = "retrieving"
EV_GENERATING         = "generating"
EV_CHECKING_RELEVANCE = "checking_relevance"
EV_COMPLETE           = "complete"
EV_RETRY              = "retry"
EV_MAX_ITER           = "max_iterations_reached"

_GENERATOR_SYSTEM = """\
You are a knowledgeable AI assistant. Answer the user's question accurately and concisely.

When context passages are provided:
  • Base your answer primarily on the provided context.
  • Cite sources inline as [Source: <title>] when referencing specific passages.
  • If the context is incomplete, supplement with your training knowledge and say so.
  • Do NOT fabricate information not present in context or verified knowledge.

When no context is provided, answer from training knowledge directly.\
"""


@dataclass
class PipelineState:
    original_query: str
    current_query: str          = ""
    source_used: str            = ""
    retrieved_context: str      = ""
    response: str               = ""
    iteration: int              = 0
    relevance_score: int        = 0
    relevance_feedback: str     = ""
    is_complete: bool           = False
    error: str                  = ""
    trace: list[dict]           = field(default_factory=list)

    def __post_init__(self) -> None:
        self.current_query = self.original_query


class AgenticRAGPipeline:
    def __init__(self, vector_store: VectorStore) -> None:
        if not config.anthropic_api_key:
            raise ValueError("ANTHROPIC_API_KEY is not set")
        self._client     = anthropic.Anthropic(api_key=config.anthropic_api_key)
        self._vs         = vector_store
        self._web        = WebSearch()
        self._tools      = Tools()
        self._rewriter   = QueryRewriter(self._client)
        self._detail     = DetailChecker(self._client)
        self._selector   = SourceSelector(self._client)
        self._relevance  = RelevanceChecker(self._client)
        self._reranker   = LLMReranker(self._client)

    # ── Synchronous run (used by CLI + tests) ─────────────────────────────────

    def run(
        self,
        query: str,
        on_event: Callable[[str, PipelineState], None] | None = None,
        memory=None,  # ConversationMemory | None
    ) -> PipelineState:
        """Execute the full loop and return the final PipelineState."""
        state = PipelineState(original_query=query)

        def emit(ev: str) -> None:
            if on_event:
                try:
                    on_event(ev, state)
                except Exception:
                    pass

        ctx = memory.format_for_rewriter() if memory else ""

        while state.iteration < config.max_iterations and not state.is_complete:
            state.iteration += 1
            step: dict = {"iteration": state.iteration}

            emit(EV_REWRITING)
            state.current_query = self._rewriter.rewrite(state.current_query, ctx)
            step["rewritten_query"] = state.current_query

            emit(EV_CHECKING_DETAILS)
            needs_retrieval, _ = self._detail.needs_retrieval(state.current_query)
            step["needs_retrieval"] = needs_retrieval

            if needs_retrieval:
                emit(EV_SELECTING_SOURCE)
                source, _ = self._selector.select(state.current_query)
                state.source_used = source
                step["source"] = source

                emit(EV_RETRIEVING)
                state.retrieved_context = self._retrieve(state.current_query, source)
                step["context_length"] = len(state.retrieved_context)
            else:
                state.source_used = "llm_knowledge"
                state.retrieved_context = ""
                step["source"] = "llm_knowledge"

            emit(EV_GENERATING)
            state.response = self._generate(
                state.current_query, state.retrieved_context, memory
            )
            step["response_length"] = len(state.response)

            emit(EV_CHECKING_RELEVANCE)
            relevant, score, feedback = self._relevance.check(
                state.original_query, state.response
            )
            state.relevance_score   = score
            state.relevance_feedback = feedback
            step.update({"relevant": relevant, "score": score, "feedback": feedback})
            state.trace.append(step)

            if relevant:
                state.is_complete = True
                emit(EV_COMPLETE)
                if memory:
                    memory.add(state.original_query, state.response, state.source_used)
            else:
                emit(EV_RETRY)
                state.current_query = (
                    f"{state.original_query}\n\n"
                    f"[Previous answer scored {score}/10. Issue: {feedback}. "
                    "Provide a more complete answer.]"
                )

        if not state.is_complete:
            emit(EV_MAX_ITER)
            if memory:
                memory.add(state.original_query, state.response, state.source_used)

        return state

    # ── Streaming run (used by REST API) ──────────────────────────────────────

    def stream(
        self,
        query: str,
        memory=None,  # ConversationMemory | None
    ) -> Generator[dict, None, None]:
        """
        Yield SSE-style event dicts for every meaningful pipeline action,
        including token-by-token generation.
        """
        state = PipelineState(original_query=query)
        ctx = memory.format_for_rewriter() if memory else ""

        try:
            while state.iteration < config.max_iterations and not state.is_complete:
                state.iteration += 1

                # Step 2 — Rewrite
                yield {"type": "step", "step": EV_REWRITING, "label": "Rewriting query"}
                state.current_query = self._rewriter.rewrite(state.current_query, ctx)
                yield {"type": "rewritten_query", "query": state.current_query}

                # Step 4 — Detail check
                yield {"type": "step", "step": EV_CHECKING_DETAILS, "label": "Checking retrieval need"}
                needs_retrieval, _ = self._detail.needs_retrieval(state.current_query)

                if needs_retrieval:
                    # Step 5 — Source
                    yield {"type": "step", "step": EV_SELECTING_SOURCE, "label": "Selecting source"}
                    source, _ = self._selector.select(state.current_query)
                    state.source_used = source
                    yield {"type": "source_selected", "source": source}

                    # Step 6-7 — Retrieve
                    yield {"type": "step", "step": EV_RETRIEVING, "label": f"Retrieving from {source}"}
                    state.retrieved_context = self._retrieve(state.current_query, source)
                else:
                    state.source_used = "llm_knowledge"
                    state.retrieved_context = ""

                # Step 8 — Generate (streaming tokens)
                yield {"type": "step", "step": EV_GENERATING, "label": "Generating response"}
                tokens: list[str] = []
                for token in self._stream_generate(
                    state.current_query, state.retrieved_context, memory
                ):
                    tokens.append(token)
                    yield {"type": "token", "delta": token}
                state.response = "".join(tokens)

                # Step 10 — Relevance
                yield {"type": "step", "step": EV_CHECKING_RELEVANCE, "label": "Evaluating quality"}
                relevant, score, feedback = self._relevance.check(
                    state.original_query, state.response
                )
                state.relevance_score    = score
                state.relevance_feedback = feedback
                yield {"type": "relevance", "score": score, "relevant": relevant, "feedback": feedback}

                if relevant:
                    state.is_complete = True
                    if memory:
                        memory.add(state.original_query, state.response, state.source_used)
                else:
                    yield {"type": "retry", "iteration": state.iteration, "feedback": feedback}
                    state.current_query = (
                        f"{state.original_query}\n\n"
                        f"[Previous answer scored {score}/10. Issue: {feedback}. "
                        "Provide a more complete answer.]"
                    )

            if not state.is_complete and memory:
                memory.add(state.original_query, state.response, state.source_used)

        except Exception as exc:
            logger.exception("Pipeline stream error")
            yield {"type": "error", "message": str(exc)}
            return

        yield {
            "type": "done",
            "response": state.response,
            "source": state.source_used,
            "score": state.relevance_score,
            "iterations": state.iteration,
            "complete": state.is_complete,
        }

    # ── Private helpers ───────────────────────────────────────────────────────

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

        # Rerank before building context
        results = self._reranker.rerank(query, results)

        parts = [
            f"[Source: {r['metadata'].get('title', 'Unknown')} | relevance: {r['relevance_score']:.3f}]\n{r['content']}"
            for r in results
        ]
        return "\n\n---\n\n".join(parts)

    def _retrieve_web(self, query: str) -> str:
        results = self._web.search(query)
        if not results:
            return "No web search results found."
        parts = [
            f"[Source: {r['metadata'].get('title', 'Web')} | {r['metadata'].get('source', '')}]\n{r['content']}"
            for r in results
        ]
        return "\n\n---\n\n".join(parts)

    def _build_messages(
        self, query: str, context: str, memory
    ) -> list[dict]:
        messages: list[dict] = []
        if memory and not memory.is_empty():
            messages.extend(memory.as_messages())
        user_content = (
            f"Context from knowledge base:\n\n{context}\n\n---\n\nQuestion: {query}"
            if context
            else f"Question: {query}"
        )
        messages.append({"role": "user", "content": user_content})
        return messages

    def _generate(
        self, query: str, context: str, memory=None, max_retries: int = 3
    ) -> str:
        messages = self._build_messages(query, context, memory)
        for attempt in range(max_retries):
            try:
                resp = self._client.messages.create(
                    model=config.model,
                    max_tokens=config.max_tokens,
                    system=[{"type": "text", "text": _GENERATOR_SYSTEM,
                             "cache_control": {"type": "ephemeral"}}],
                    messages=messages,
                )
                return resp.content[0].text.strip()
            except anthropic.RateLimitError:
                time.sleep(2 ** attempt)
            except anthropic.APIError as exc:
                logger.error("Generator error (attempt %d): %s", attempt + 1, exc)
                if attempt == max_retries - 1:
                    return f"Error generating response: {exc}"
                time.sleep(2 ** attempt)
        return "Failed to generate a response after multiple attempts."

    def _stream_generate(
        self, query: str, context: str, memory=None, max_retries: int = 3
    ) -> Generator[str, None, None]:
        messages = self._build_messages(query, context, memory)
        for attempt in range(max_retries):
            try:
                with self._client.messages.stream(
                    model=config.model,
                    max_tokens=config.max_tokens,
                    system=[{"type": "text", "text": _GENERATOR_SYSTEM,
                             "cache_control": {"type": "ephemeral"}}],
                    messages=messages,
                ) as stream:
                    for text in stream.text_stream:
                        yield text
                return
            except anthropic.RateLimitError:
                wait = 2 ** attempt
                logger.warning("Stream generator rate-limited; retrying in %ds", wait)
                time.sleep(wait)
            except anthropic.APIError as exc:
                logger.error("Stream generator error (attempt %d): %s", attempt + 1, exc)
                if attempt == max_retries - 1:
                    yield f"\n[Error: {exc}]"
                    return
                time.sleep(2 ** attempt)
