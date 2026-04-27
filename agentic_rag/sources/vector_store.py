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
        if config.chroma_host:
            # HTTP client mode — connects to a separate ChromaDB server.
            # Used in production (Cloud Run, Docker Compose with separate service).
            self._client = chromadb.HttpClient(
                host=config.chroma_host,
                port=config.chroma_port,
            )
            logger.info("ChromaDB HTTP client → %s:%d", config.chroma_host, config.chroma_port)
        else:
            # Embedded mode — local SQLite file.
            # Fine for dev and single-VM deployments.
            self._client = chromadb.PersistentClient(path=config.chroma_persist_dir)
            logger.info("ChromaDB embedded → %s", config.chroma_persist_dir)

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

    def list_sources(self) -> list[dict[str, Any]]:
        """
        Return a deduplicated list of indexed sources with chunk counts.
        Each entry: {"source": str, "title": str, "filetype": str, "chunks": int}
        """
        if self.count() == 0:
            return []

        # Fetch all metadata (no embeddings needed)
        raw = self._collection.get(include=["metadatas"])
        tally: dict[str, dict[str, Any]] = {}

        for meta in raw["metadatas"]:
            key = meta.get("source") or meta.get("filename") or meta.get("title", "unknown")
            if key not in tally:
                tally[key] = {
                    "source": key,
                    "title": meta.get("title") or meta.get("filename") or key,
                    "filetype": meta.get("filetype") or meta.get("source", "?"),
                    "count": 0,
                }
            tally[key]["count"] += 1

        return sorted(tally.values(), key=lambda x: x["title"].lower())

    def export_all(self) -> list[dict[str, Any]]:
        """Return all stored chunks as a list of {id, content, metadata} dicts."""
        if self.count() == 0:
            return []
        raw = self._collection.get(include=["documents", "metadatas"])
        return [
            {"id": uid, "content": doc, "metadata": meta}
            for uid, doc, meta in zip(raw["ids"], raw["documents"], raw["metadatas"])
        ]

    def import_all(self, chunks: list[dict[str, Any]]) -> int:
        """Upsert exported chunks back into the collection. Returns count upserted."""
        if not chunks:
            return 0
        self.upsert(
            documents=[c["content"] for c in chunks],
            metadatas=[c["metadata"] for c in chunks],
            ids=[c["id"] for c in chunks],
        )
        return len(chunks)

    def delete_source(self, source: str) -> int:
        """Delete all chunks whose 'source' metadata matches *source*. Returns deleted count."""
        raw = self._collection.get(include=["metadatas"])
        ids_to_delete = [
            uid
            for uid, meta in zip(raw["ids"], raw["metadatas"])
            if meta.get("source") == source
            or meta.get("filename") == source
            or meta.get("title") == source
        ]
        if ids_to_delete:
            self._collection.delete(ids=ids_to_delete)
        return len(ids_to_delete)


def create_vector_store() -> "VectorStore":
    """Return the configured vector store backend.

    USE_FIRESTORE=true → FirestoreVectorStore (GCP-native, always-free).
    Otherwise         → VectorStore (ChromaDB, for local dev).
    """
    if config.use_firestore:
        from .firestore_vector_store import FirestoreVectorStore  # lazy import
        return FirestoreVectorStore()
    return VectorStore()
