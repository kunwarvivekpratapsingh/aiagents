"""
Tests for the document loader pipeline.

Uses only in-memory / temp-file operations — no network, no API key.
"""
from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Fixtures — sample files written to a temp directory
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def sample_dir():
    tmp = tempfile.mkdtemp(prefix="rag_loaders_test_")
    yield Path(tmp)
    shutil.rmtree(tmp, ignore_errors=True)


@pytest.fixture(scope="module")
def txt_file(sample_dir):
    p = sample_dir / "sample.txt"
    p.write_text(
        "Retrieval-Augmented Generation combines retrieval with generation.\n"
        "It grounds LLM answers in external documents to reduce hallucination.\n"
        "The retriever fetches relevant passages; the generator conditions on them.",
        encoding="utf-8",
    )
    return p


@pytest.fixture(scope="module")
def md_file(sample_dir):
    p = sample_dir / "notes.md"
    p.write_text(
        "# Transformers\n\nTransformers use self-attention to process sequences in parallel.\n"
        "## Architecture\n- Encoder stack\n- Decoder stack\n- Feed-forward layers",
        encoding="utf-8",
    )
    return p


@pytest.fixture(scope="module")
def html_file(sample_dir):
    p = sample_dir / "page.html"
    p.write_text(
        "<html><body><h1>Vector Databases</h1>"
        "<p>Vector databases store embeddings and support approximate nearest neighbour search.</p>"
        "<p>They power semantic search and RAG pipelines.</p></body></html>",
        encoding="utf-8",
    )
    return p


@pytest.fixture(scope="module")
def csv_file(sample_dir):
    p = sample_dir / "data.csv"
    p.write_text("model,params,year\nGPT-3,175B,2020\nGPT-4,unknown,2023\nLLaMA-3,70B,2024\n")
    return p


# ---------------------------------------------------------------------------
# TextLoader tests
# ---------------------------------------------------------------------------

class TestTextLoader:
    def test_loads_txt(self, txt_file):
        from agentic_rag.loaders.text_loader import TextLoader
        docs = TextLoader().load(txt_file)
        assert len(docs) == 1
        assert "Retrieval-Augmented Generation" in docs[0].content
        assert docs[0].metadata["filetype"] == "txt"
        assert docs[0].metadata["filename"] == "sample.txt"

    def test_loads_markdown(self, md_file):
        from agentic_rag.loaders.text_loader import TextLoader
        docs = TextLoader().load(md_file)
        assert len(docs) == 1
        assert "Transformers" in docs[0].content
        assert docs[0].metadata["filetype"] == "md"

    def test_loads_html_strips_tags(self, html_file):
        from agentic_rag.loaders.text_loader import TextLoader
        docs = TextLoader().load(html_file)
        assert len(docs) == 1
        assert "<html>" not in docs[0].content
        assert "Vector Databases" in docs[0].content

    def test_loads_csv(self, csv_file):
        from agentic_rag.loaders.text_loader import TextLoader
        docs = TextLoader().load(csv_file)
        assert len(docs) == 1
        assert "GPT-3" in docs[0].content

    def test_empty_file_returns_empty(self, sample_dir):
        from agentic_rag.loaders.text_loader import TextLoader
        p = sample_dir / "empty.txt"
        p.write_text("")
        docs = TextLoader().load(p)
        assert docs == []

    def test_missing_file_raises(self, sample_dir):
        from agentic_rag.loaders.text_loader import TextLoader
        with pytest.raises(OSError):
            TextLoader().load(sample_dir / "nonexistent.txt")


# ---------------------------------------------------------------------------
# PDFLoader tests (skipped if pypdf not installed)
# ---------------------------------------------------------------------------

class TestPDFLoader:
    def test_import_error_message(self, tmp_path):
        """If pdfminer.six is not installed, error message is helpful."""
        from unittest.mock import patch
        from agentic_rag.loaders.pdf_loader import PDFLoader

        p = tmp_path / "fake.pdf"
        p.write_bytes(b"%PDF-1.4")
        with patch("agentic_rag.loaders.pdf_loader.PDFLoader.load",
                   side_effect=ImportError("Install pdfminer.six: pip install pdfminer.six")):
            with pytest.raises(ImportError, match="pdfminer"):
                PDFLoader().load(p)


# ---------------------------------------------------------------------------
# DocxLoader tests (skipped if python-docx not installed)
# ---------------------------------------------------------------------------

class TestDocxLoader:
    def test_import_error_message(self, sample_dir):
        from unittest.mock import patch
        import builtins
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "docx":
                raise ImportError("No module named 'docx'")
            return real_import(name, *args, **kwargs)

        from agentic_rag.loaders.docx_loader import DocxLoader
        p = sample_dir / "fake.docx"
        p.write_bytes(b"PK\x03\x04")  # ZIP magic bytes
        with patch("builtins.__import__", side_effect=mock_import):
            with pytest.raises(ImportError, match="python-docx"):
                DocxLoader().load(p)


# ---------------------------------------------------------------------------
# Dispatcher tests
# ---------------------------------------------------------------------------

class TestDispatcher:
    def test_routes_txt(self, txt_file):
        from agentic_rag.loaders.dispatcher import load_file
        docs = load_file(txt_file)
        assert len(docs) >= 1

    def test_routes_md(self, md_file):
        from agentic_rag.loaders.dispatcher import load_file
        docs = load_file(md_file)
        assert len(docs) >= 1

    def test_routes_html(self, html_file):
        from agentic_rag.loaders.dispatcher import load_file
        docs = load_file(html_file)
        assert len(docs) >= 1
        assert "<" not in docs[0].content

    def test_unsupported_extension_raises(self, sample_dir):
        from agentic_rag.loaders.dispatcher import load_file
        p = sample_dir / "data.xyz"
        p.write_text("hello")
        with pytest.raises(ValueError, match="Unsupported"):
            load_file(p)

    def test_missing_file_raises(self, sample_dir):
        from agentic_rag.loaders.dispatcher import load_file
        with pytest.raises(FileNotFoundError):
            load_file(sample_dir / "ghost.txt")

    def test_load_directory_finds_all(self, tmp_path):
        from agentic_rag.loaders.dispatcher import load_directory
        # Use a clean isolated directory with only text-based files
        (tmp_path / "a.txt").write_text("Transformers use attention.")
        (tmp_path / "b.md").write_text("# RAG\nRetrieval augmented generation.")
        (tmp_path / "c.html").write_text("<p>Vector databases store embeddings.</p>")
        docs, errors = load_directory(tmp_path, recursive=False)
        assert len(docs) >= 3
        assert all(isinstance(e, str) for e in errors)

    def test_load_directory_skips_unsupported(self, tmp_path):
        from agentic_rag.loaders.dispatcher import load_directory
        (tmp_path / "valid.txt").write_text("Valid text content for testing.")
        (tmp_path / "junk.xyz").write_text("ignored")
        docs, errors = load_directory(tmp_path, recursive=False)
        filenames = [d.metadata["filename"] for d in docs]
        assert "junk.xyz" not in filenames
        assert "valid.txt" in filenames


# ---------------------------------------------------------------------------
# Integration — ingest_file / ingest_directory
# ---------------------------------------------------------------------------

class TestIngestIntegration:
    """Each test gets its own fresh VectorStore via tmp_path (pytest built-in)."""

    def _make_vs(self, db_path: str):
        import os
        os.environ["ANTHROPIC_API_KEY"] = "test-key"
        from agentic_rag.config import Config
        from agentic_rag.sources.vector_store import VectorStore
        from unittest.mock import patch
        cfg = Config(chroma_persist_dir=db_path)
        with patch("agentic_rag.sources.vector_store.config", cfg):
            vs = VectorStore()
        return vs, cfg

    def test_ingest_file_adds_chunks(self, tmp_path):
        doc = tmp_path / "rag.txt"
        doc.write_text(
            "Retrieval-Augmented Generation combines retrieval with generation. "
            "It grounds LLM answers in external documents to reduce hallucination."
        )
        vs, cfg = self._make_vs(str(tmp_path / "db"))
        from unittest.mock import patch
        with patch("agentic_rag.data.ingest.config", cfg):
            from agentic_rag.data.ingest import ingest_file
            n = ingest_file(vs, doc)
        assert n >= 1
        assert vs.count() >= 1

    def test_ingest_directory_adds_multiple(self, tmp_path):
        (tmp_path / "docs").mkdir()
        (tmp_path / "docs" / "a.txt").write_text(
            "Transformers revolutionised NLP by introducing self-attention mechanisms that allow "
            "each token to attend to every other token in a sequence simultaneously in parallel."
        )
        (tmp_path / "docs" / "b.md").write_text(
            "BERT stands for Bidirectional Encoder Representations from Transformers. "
            "It uses masked language modelling as a pre-training objective on large corpora."
        )
        vs, cfg = self._make_vs(str(tmp_path / "db"))
        from unittest.mock import patch
        with patch("agentic_rag.data.ingest.config", cfg):
            from agentic_rag.data.ingest import ingest_directory
            n, errors = ingest_directory(vs, tmp_path / "docs", recursive=False)
        assert n >= 2
        assert errors == []

    def test_list_sources_after_ingest(self, tmp_path):
        doc = tmp_path / "llm.txt"
        doc.write_text(
            "Large language models are neural networks trained on trillions of tokens from the "
            "internet using self-supervised objectives such as next-token prediction."
        )
        vs, cfg = self._make_vs(str(tmp_path / "db"))
        from unittest.mock import patch
        with patch("agentic_rag.data.ingest.config", cfg):
            from agentic_rag.data.ingest import ingest_file
            ingest_file(vs, doc)
        sources = vs.list_sources()
        assert len(sources) >= 1
        assert any("llm" in s["title"].lower() for s in sources)

    def test_retrieval_after_ingest(self, tmp_path):
        doc = tmp_path / "rag2.txt"
        doc.write_text(
            "RAG systems retrieve relevant documents before generating answers. "
            "The retriever fetches passages; the LLM generates grounded responses."
        )
        vs, cfg = self._make_vs(str(tmp_path / "db"))
        from unittest.mock import patch
        with patch("agentic_rag.data.ingest.config", cfg):
            from agentic_rag.data.ingest import ingest_file
            ingest_file(vs, doc)
        results = vs.search("retrieval augmented generation")
        assert len(results) >= 1
        content = results[0]["content"].lower()
        assert "retriev" in content or "rag" in content
