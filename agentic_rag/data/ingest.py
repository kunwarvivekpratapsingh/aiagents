"""
Wikipedia-backed open-source corpus ingestion.

Fetches articles on AI/ML topics, splits them into overlapping chunks,
and upserts into the ChromaDB vector store.
"""
from __future__ import annotations

import hashlib
import logging
from typing import Callable

from ..sources.vector_store import VectorStore
from ..config import config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Corpus — 20 Wikipedia topics covering the knowledge base domain
# ---------------------------------------------------------------------------

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
    "Recurrent neural network",
    "Convolutional neural network",
    "Diffusion model",
    "Hallucination (artificial intelligence)",
]


# ---------------------------------------------------------------------------
# Text chunking
# ---------------------------------------------------------------------------

def _chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    words = text.split()
    step = max(1, chunk_size - overlap)
    chunks: list[str] = []
    for i in range(0, len(words), step):
        chunk = " ".join(words[i : i + chunk_size])
        if len(chunk.split()) >= 40:  # skip micro-chunks
            chunks.append(chunk)
    return chunks


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def ingest_wikipedia(
    vector_store: VectorStore,
    topics: list[str] | None = None,
    progress_cb: Callable[[str, int, int], None] | None = None,
) -> int:
    """
    Fetch Wikipedia articles and upsert them into *vector_store*.

    Args:
        vector_store: Target VectorStore instance.
        topics:       Override the default topic list.
        progress_cb:  Optional callback(topic, current, total) for UI progress.

    Returns:
        Total number of document chunks ingested.
    """
    try:
        import wikipediaapi
    except ImportError as exc:
        raise ImportError("Install wikipedia-api: pip install wikipedia-api") from exc

    topics = topics or WIKIPEDIA_TOPICS
    wiki = wikipediaapi.Wikipedia(
        language="en",
        user_agent="AgenticRAG/1.0 (open-source research project)",
    )

    documents: list[str] = []
    metadatas: list[dict] = []
    ids: list[str] = []

    for idx, topic in enumerate(topics):
        if progress_cb:
            progress_cb(topic, idx + 1, len(topics))

        page = wiki.page(topic)
        if not page.exists():
            logger.warning("Wikipedia page not found: %s", topic)
            continue

        text = page.text
        if not text.strip():
            continue

        chunks = _chunk_text(text, config.chunk_size, config.chunk_overlap)
        logger.info("  %s → %d chunks", topic, len(chunks))

        for i, chunk in enumerate(chunks):
            uid = hashlib.sha256(f"{topic}||{i}".encode()).hexdigest()[:32]
            documents.append(chunk)
            metadatas.append(
                {
                    "source": "wikipedia",
                    "title": topic,
                    "chunk_index": i,
                    "url": page.fullurl,
                }
            )
            ids.append(uid)

    # Upsert in batches of 200 to avoid memory spikes
    batch = 200
    for start in range(0, len(documents), batch):
        vector_store.upsert(
            documents[start : start + batch],
            metadatas[start : start + batch],
            ids[start : start + batch],
        )

    logger.info("Ingestion complete — %d chunks from %d topics", len(documents), len(topics))
    return len(documents)
