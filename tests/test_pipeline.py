"""
End-to-end test suite for the Agentic RAG pipeline.

All Anthropic API calls are mocked so tests run without an API key.
VectorStore and Tools are exercised with real code.
"""
from __future__ import annotations

import os
import shutil
import tempfile
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers — fake API responses
# ---------------------------------------------------------------------------

def _msg(text: str) -> MagicMock:
    """Create a fake anthropic.Message with a single TextBlock."""
    block = SimpleNamespace(text=text)
    return SimpleNamespace(content=[block])


def _make_client(
    rewrite_text: str = "rewritten query about RAG",
    needs_retrieval_json: str = '{"needs_retrieval": true, "reason": "needs facts"}',
    source_json: str = '{"source": "vector_db", "reason": "AI topic"}',
    reranker_json: str = "[1, 2, 3, 4, 5]",
    response_text: str = "RAG combines retrieval with generation to ground LLM answers in external documents.",
    relevance_json: str = '{"is_relevant": true, "score": 9, "feedback": "complete answer"}',
):
    """Build a mock anthropic.Anthropic client with configurable return values."""
    client = MagicMock()
    client.messages.create.side_effect = [
        _msg(rewrite_text),         # QueryRewriter
        _msg(needs_retrieval_json), # DetailChecker
        _msg(source_json),          # SourceSelector
        _msg(reranker_json),        # LLMReranker
        _msg(response_text),        # Generator
        _msg(relevance_json),       # RelevanceChecker
    ]
    return client


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def tmp_vs():
    """A VectorStore backed by a temp directory, pre-loaded with corpus."""
    tmp = tempfile.mkdtemp(prefix="rag_test_")
    orig = os.environ.get("ANTHROPIC_API_KEY", "")
    os.environ["ANTHROPIC_API_KEY"] = "test-key"

    from agentic_rag.config import Config
    cfg = Config(chroma_persist_dir=tmp)

    with patch("agentic_rag.sources.vector_store.config", cfg), \
         patch("agentic_rag.data.ingest.config", cfg):
        from agentic_rag.sources.vector_store import VectorStore
        from agentic_rag.data.ingest import ingest

        vs = VectorStore()
        # Local corpus only — no network calls
        n = ingest(vs, use_wikipedia=False)
        assert n > 0, "Corpus ingestion produced no chunks"
        yield vs

    shutil.rmtree(tmp, ignore_errors=True)
    if orig:
        os.environ["ANTHROPIC_API_KEY"] = orig
    else:
        os.environ.pop("ANTHROPIC_API_KEY", None)


# ---------------------------------------------------------------------------
# Unit tests — individual components
# ---------------------------------------------------------------------------

class TestVectorStore:
    def test_count_after_ingest(self, tmp_vs):
        assert tmp_vs.count() > 0

    def test_search_returns_results(self, tmp_vs):
        results = tmp_vs.search("retrieval augmented generation", n_results=3)
        assert len(results) >= 1
        assert all("content" in r for r in results)
        assert all("relevance_score" in r for r in results)

    def test_search_rag_topic_is_top(self, tmp_vs):
        results = tmp_vs.search("how does RAG work")
        top_title = results[0]["metadata"]["title"].lower()
        assert "retrieval" in top_title or "rag" in top_title

    def test_search_transformer_topic(self, tmp_vs):
        results = tmp_vs.search("self-attention mechanism transformers")
        titles = [r["metadata"]["title"].lower() for r in results]
        assert any("transformer" in t or "attention" in t for t in titles)


class TestTools:
    def setup_method(self):
        from agentic_rag.sources.tools import Tools
        self.tools = Tools()

    def test_datetime(self):
        result = self.tools.execute("what is today's date")
        assert "Current date" in result

    def test_arithmetic(self):
        result = self.tools.execute("calculate 6 * 7")
        assert "42" in result

    def test_celsius_to_fahrenheit(self):
        result = self.tools.execute("convert 0 celsius to fahrenheit")
        assert "32" in result

    def test_km_to_miles(self):
        result = self.tools.execute("convert 10 km to miles")
        assert "6.2" in result


class TestQueryRewriter:
    def test_returns_rewritten_string(self):
        from agentic_rag.agents.query_rewriter import QueryRewriter
        client = MagicMock()
        client.messages.create.return_value = _msg("detailed query about vector databases")
        rw = QueryRewriter(client)
        result = rw.rewrite("what's a vector db")
        assert result == "detailed query about vector databases"

    def test_falls_back_to_original_on_empty_response(self):
        from agentic_rag.agents.query_rewriter import QueryRewriter
        client = MagicMock()
        client.messages.create.return_value = _msg("")
        rw = QueryRewriter(client)
        original = "my original query"
        assert rw.rewrite(original) == original


class TestDetailChecker:
    def test_needs_retrieval_true(self):
        from agentic_rag.agents.detail_checker import DetailChecker
        client = MagicMock()
        client.messages.create.return_value = _msg('{"needs_retrieval": true, "reason": "needs facts"}')
        result, reason = DetailChecker(client).needs_retrieval("What year was RAG invented?")
        assert result is True
        assert "facts" in reason

    def test_needs_retrieval_false(self):
        from agentic_rag.agents.detail_checker import DetailChecker
        client = MagicMock()
        client.messages.create.return_value = _msg('{"needs_retrieval": false, "reason": "general knowledge"}')
        result, _ = DetailChecker(client).needs_retrieval("What is 2+2?")
        assert result is False

    def test_defaults_to_true_on_bad_json(self):
        from agentic_rag.agents.detail_checker import DetailChecker
        client = MagicMock()
        client.messages.create.return_value = _msg("not json")
        result, _ = DetailChecker(client).needs_retrieval("anything")
        assert result is True


class TestSourceSelector:
    def test_selects_vector_db(self):
        from agentic_rag.agents.source_selector import SourceSelector
        client = MagicMock()
        client.messages.create.return_value = _msg('{"source": "vector_db", "reason": "AI topic"}')
        source, _ = SourceSelector(client).select("how does attention work")
        assert source == "vector_db"

    def test_selects_tools(self):
        from agentic_rag.agents.source_selector import SourceSelector
        client = MagicMock()
        client.messages.create.return_value = _msg('{"source": "tools", "reason": "calculation"}')
        source, _ = SourceSelector(client).select("calculate 5 * 5")
        assert source == "tools"

    def test_unknown_source_falls_back(self):
        from agentic_rag.agents.source_selector import SourceSelector
        client = MagicMock()
        client.messages.create.return_value = _msg('{"source": "magic_db", "reason": "x"}')
        source, _ = SourceSelector(client).select("anything")
        assert source == "vector_db"


class TestRelevanceChecker:
    def test_relevant_response(self):
        from agentic_rag.agents.relevance_checker import RelevanceChecker
        client = MagicMock()
        client.messages.create.return_value = _msg('{"is_relevant": true, "score": 9, "feedback": "great"}')
        relevant, score, feedback = RelevanceChecker(client).check("q", "good answer")
        assert relevant is True
        assert score == 9

    def test_irrelevant_response(self):
        from agentic_rag.agents.relevance_checker import RelevanceChecker
        client = MagicMock()
        client.messages.create.return_value = _msg('{"is_relevant": false, "score": 3, "feedback": "missing details"}')
        relevant, score, feedback = RelevanceChecker(client).check("q", "bad answer")
        assert relevant is False
        assert score == 3
        assert "missing" in feedback


# ---------------------------------------------------------------------------
# Integration tests — full pipeline
# ---------------------------------------------------------------------------

class TestAgenticPipeline:
    def _make_pipeline(self, tmp_vs, **kwargs):
        """Build a pipeline with a fully mocked Anthropic client."""
        from agentic_rag.pipeline.rag_pipeline import AgenticRAGPipeline
        from agentic_rag.config import config as _cfg
        with patch.object(_cfg.__class__, "anthropic_api_key", new="test-key", create=True):
            _cfg.anthropic_api_key = "test-key"
            pipeline = AgenticRAGPipeline(tmp_vs)
        pipeline._client = _make_client(**kwargs)
        # Re-inject mocked client into all agents
        pipeline._rewriter._client = pipeline._client
        pipeline._detail._client = pipeline._client
        pipeline._selector._client = pipeline._client
        pipeline._relevance._client = pipeline._client
        pipeline._reranker._client = pipeline._client
        return pipeline

    def test_happy_path_completes_in_one_iteration(self, tmp_vs):
        pipeline = self._make_pipeline(tmp_vs)
        events: list[str] = []
        state = pipeline.run("What is RAG?", on_event=lambda ev, _: events.append(ev))
        assert state.is_complete
        assert state.iteration == 1
        assert "complete" in events
        assert state.relevance_score == 9

    def test_no_retrieval_path(self, tmp_vs):
        """When detail_checker says no retrieval, source_selector is skipped."""
        pipeline = self._make_pipeline(
            tmp_vs,
            needs_retrieval_json='{"needs_retrieval": false, "reason": "general"}',
            source_json='{"source": "vector_db", "reason": "unused"}',
        )
        # Provide only 4 responses (no source_selector call)
        pipeline._client.messages.create.side_effect = [
            _msg("rewritten query"),
            _msg('{"needs_retrieval": false, "reason": "general"}'),
            _msg("The answer from LLM knowledge."),
            _msg('{"is_relevant": true, "score": 8, "feedback": "good"}'),
        ]
        state = pipeline.run("What is 2+2?")
        assert state.source_used == "llm_knowledge"
        assert state.is_complete

    def test_retry_on_low_relevance(self, tmp_vs):
        """Pipeline retries when relevance score < 6 on first iteration."""
        pipeline = self._make_pipeline(tmp_vs)
        pipeline._client.messages.create.side_effect = [
            # Iteration 1
            _msg("rewritten query v1"),
            _msg('{"needs_retrieval": true, "reason": "needs facts"}'),
            _msg('{"source": "vector_db", "reason": "AI topic"}'),
            _msg("[1, 2, 3, 4, 5]"),                               # reranker
            _msg("Incomplete answer."),
            _msg('{"is_relevant": false, "score": 3, "feedback": "missing key details"}'),
            # Iteration 2
            _msg("rewritten query v2 with feedback"),
            _msg('{"needs_retrieval": true, "reason": "needs facts"}'),
            _msg('{"source": "vector_db", "reason": "AI topic"}'),
            _msg("[1, 2, 3, 4, 5]"),                               # reranker
            _msg("Thorough and complete answer about RAG."),
            _msg('{"is_relevant": true, "score": 8, "feedback": "much better"}'),
        ]
        events: list[str] = []
        state = pipeline.run("What is RAG?", on_event=lambda ev, _: events.append(ev))
        assert state.iteration == 2
        assert state.is_complete
        assert "retry" in events
        assert "complete" in events

    def test_tools_source_used_for_calculation(self, tmp_vs):
        """When source_selector picks 'tools', tools.execute() is called."""
        pipeline = self._make_pipeline(tmp_vs, source_json='{"source": "tools", "reason": "math"}')
        # Patch tools to capture call
        pipeline._tools = MagicMock()
        pipeline._tools.execute.return_value = "6 * 7 = 42"
        # Supply all needed API responses
        pipeline._client.messages.create.side_effect = [
            _msg("calculate 6 * 7"),
            _msg('{"needs_retrieval": true, "reason": "calculation"}'),
            _msg('{"source": "tools", "reason": "math"}'),
            _msg("The answer is 42."),
            _msg('{"is_relevant": true, "score": 10, "feedback": "perfect"}'),
        ]
        state = pipeline.run("What is 6 times 7?")
        assert state.source_used == "tools"
        pipeline._tools.execute.assert_called_once()

    def test_state_trace_populated(self, tmp_vs):
        """Each iteration appends an entry to state.trace."""
        pipeline = self._make_pipeline(tmp_vs)
        state = pipeline.run("Explain transformers")
        assert len(state.trace) == state.iteration
        for entry in state.trace:
            assert "iteration" in entry
            assert "score" in entry


# ---------------------------------------------------------------------------
# Ingestion tests
# ---------------------------------------------------------------------------

class TestIngestion:
    def test_local_corpus_files_exist(self):
        from pathlib import Path
        corpus_dir = Path(__file__).parent.parent / "agentic_rag" / "data" / "corpus"
        txt_files = list(corpus_dir.glob("*.txt"))
        assert len(txt_files) >= 15, f"Expected ≥15 corpus files, found {len(txt_files)}"

    def test_ingest_returns_positive_count(self, tmp_vs):
        assert tmp_vs.count() > 0

    def test_each_topic_searchable(self, tmp_vs):
        queries = [
            ("RAG", "retrieval augmented generation"),
            ("LLM", "large language model training"),
            ("Transformer", "transformer architecture self attention"),
            ("Vector DB", "vector database hnsw approximate nearest neighbour"),
            ("Safety", "ai safety alignment hallucination"),
        ]
        for label, query in queries:
            results = tmp_vs.search(query, n_results=1)
            assert results, f"No results for '{label}' query: {query}"
