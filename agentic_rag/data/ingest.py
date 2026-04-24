"""
Corpus ingestion — local-first, Wikipedia-optional.

Load order:
  1. Local bundled corpus  (always available, no network required)
  2. Wikipedia             (augments the local corpus when network is available)

Call `ingest(vector_store)` to populate the VectorStore.
"""
from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path
from typing import Callable

from ..sources.vector_store import VectorStore
from ..config import config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_CORPUS_DIR = Path(__file__).parent / "corpus"

# Optional Wikipedia topics to fetch when network is available
WIKIPEDIA_TOPICS: list[str] = [
    "Artificial intelligence",
    "Machine learning",
    "Deep learning",
    "Natural language processing",
    "Large language model",
    "Retrieval-augmented generation",
    "Transformer (deep learning architecture)",
    "BERT (language model)",
    "Generative pre-trained transformer",
    "Reinforcement learning",
    "Neural network (machine learning)",
    "Knowledge graph",
    "Vector database",
    "Semantic search",
    "Prompt engineering",
    "Attention (machine learning)",
    "Diffusion model",
    "Hallucination (artificial intelligence)",
]


# ---------------------------------------------------------------------------
# Text chunking
# ---------------------------------------------------------------------------

def _chunk(text: str, size: int, overlap: int) -> list[str]:
    words = text.split()
    step = max(1, size - overlap)
    return [
        " ".join(words[i : i + size])
        for i in range(0, len(words), step)
        if len(words[i : i + size]) >= 30  # skip micro-chunks
    ]


def _uid(source: str, index: int) -> str:
    return hashlib.sha256(f"{source}||{index}".encode()).hexdigest()[:32]


# ---------------------------------------------------------------------------
# Local corpus loader
# ---------------------------------------------------------------------------

def _load_local_corpus(
    vector_store: VectorStore,
    progress_cb: Callable[[str, int, int], None] | None = None,
) -> int:
    files = sorted(_CORPUS_DIR.glob("*.txt"))
    if not files:
        logger.warning("No corpus files found in %s", _CORPUS_DIR)
        return 0

    documents: list[str] = []
    metadatas: list[dict] = []
    ids: list[str] = []

    for idx, path in enumerate(files):
        title = path.stem.replace("_", " ").title()
        if progress_cb:
            progress_cb(title, idx + 1, len(files))

        text = path.read_text(encoding="utf-8").strip()
        if not text:
            continue

        chunks = _chunk(text, config.chunk_size, config.chunk_overlap)
        for i, chunk in enumerate(chunks):
            uid = _uid(f"local:{path.stem}", i)
            documents.append(chunk)
            metadatas.append(
                {
                    "source": "local_corpus",
                    "title": title,
                    "file": path.name,
                    "chunk_index": i,
                }
            )
            ids.append(uid)

    _batch_upsert(vector_store, documents, metadatas, ids)
    logger.info("Local corpus: %d chunks from %d files", len(documents), len(files))
    return len(documents)


# ---------------------------------------------------------------------------
# Wikipedia loader (optional — requires network)
# ---------------------------------------------------------------------------

def _load_wikipedia(
    vector_store: VectorStore,
    topics: list[str],
    progress_cb: Callable[[str, int, int], None] | None = None,
) -> int:
    try:
        import wikipediaapi
    except ImportError:
        logger.warning("wikipedia-api not installed; skipping Wikipedia ingestion")
        return 0

    wiki = wikipediaapi.Wikipedia(
        language="en",
        user_agent="AgenticRAG/2.0 (open-source research project)",
    )

    documents: list[str] = []
    metadatas: list[dict] = []
    ids: list[str] = []
    fetched = 0

    for idx, topic in enumerate(topics):
        if progress_cb:
            progress_cb(topic, idx + 1, len(topics))
        try:
            page = wiki.page(topic)
            if not page.exists() or not page.text.strip():
                logger.debug("Wikipedia page missing or empty: %s", topic)
                continue

            chunks = _chunk(page.text, config.chunk_size, config.chunk_overlap)
            for i, chunk in enumerate(chunks):
                uid = _uid(f"wiki:{topic}", i)
                documents.append(chunk)
                metadatas.append(
                    {
                        "source": "wikipedia",
                        "title": topic,
                        "url": page.fullurl,
                        "chunk_index": i,
                    }
                )
                ids.append(uid)
            fetched += 1
        except Exception as exc:
            logger.warning("Failed to fetch Wikipedia page '%s': %s", topic, exc)

    _batch_upsert(vector_store, documents, metadatas, ids)
    logger.info("Wikipedia: %d chunks from %d/%d pages", len(documents), fetched, len(topics))
    return len(documents)


# ---------------------------------------------------------------------------
# Batch upsert helper
# ---------------------------------------------------------------------------

def _batch_upsert(
    vector_store: VectorStore,
    documents: list[str],
    metadatas: list[dict],
    ids: list[str],
    batch: int = 200,
) -> None:
    for start in range(0, len(documents), batch):
        vector_store.upsert(
            documents[start : start + batch],
            metadatas[start : start + batch],
            ids[start : start + batch],
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def ingest(
    vector_store: VectorStore,
    use_wikipedia: bool = True,
    wikipedia_topics: list[str] | None = None,
    progress_cb: Callable[[str, int, int], None] | None = None,
) -> int:
    """
    Populate *vector_store* with the bundled local corpus and optionally
    Wikipedia articles.

    Returns total chunks ingested.
    """
    total = 0
    total += _load_local_corpus(vector_store, progress_cb=progress_cb)

    if use_wikipedia:
        topics = wikipedia_topics or WIKIPEDIA_TOPICS
        wiki_count = _load_wikipedia(vector_store, topics, progress_cb=progress_cb)
        if wiki_count == 0:
            logger.info("Wikipedia unavailable — running on local corpus only")
        total += wiki_count

    return total
