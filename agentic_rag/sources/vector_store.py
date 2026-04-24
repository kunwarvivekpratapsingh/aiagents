"""ChromaDB-backed vector store with cosine-similarity retrieval."""
from __future__ import annotations

import logging
from typing import Any

import chromadb
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

from ..config import config

logger = logging.getLogger(__name__)


class VectorStore:
    def __init__(self) -> None:
        self._client = chromadb.PersistentClient(path=config.chroma_persist_dir)
        # DefaultEmbeddingFunction uses a bundled ONNX MiniLM model —
        # no HuggingFace download required, works fully offline.
        self._ef = DefaultEmbeddingFunction()
        self._collection = self._client.get_or_create_collection(
            name=config.collection_name,
            embedding_function=self._ef,
            metadata={"hnsw:space": "cosine"},
        )

    # ------------------------------------------------------------------
    # Writes
    # ------------------------------------------------------------------

    def upsert(
        self,
        documents: list[str],
        metadatas: list[dict[str, Any]],
        ids: list[str],
    ) -> None:
        if not documents:
            return
        self._collection.upsert(documents=documents, metadatas=metadatas, ids=ids)

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------

    def search(self, query: str, n_results: int | None = None) -> list[dict[str, Any]]:
        n = min(n_results or config.top_k_retrieval, self.count())
        if n == 0:
            return []

        raw = self._collection.query(query_texts=[query], n_results=n)
        results: list[dict[str, Any]] = []
        for doc, meta, dist in zip(
            raw["documents"][0],
            raw["metadatas"][0],
            raw["distances"][0],
        ):
            results.append(
                {
                    "content": doc,
                    "metadata": meta,
                    "relevance_score": round(1.0 - dist, 4),
                }
            )
        return results

    def count(self) -> int:
        return self._collection.count()
