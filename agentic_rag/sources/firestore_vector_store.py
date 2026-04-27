"""
Firestore-backed vector store using Firestore Vector Search.

Replaces ChromaDB for GCP deployments — fully managed, always-free tier:
  - 1 GB storage free
  - 50,000 reads/day free
  - 20,000 writes/day free

Embeddings are generated locally using the bundled ONNX MiniLM model
(same as ChromaDB's DefaultEmbeddingFunction) — no external API calls.

Set USE_FIRESTORE=true to activate this backend.
"""
from __future__ import annotations

import logging
from typing import Any

from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

from ..config import config

logger = logging.getLogger(__name__)

_BATCH_SIZE = 400  # Firestore batch write limit is 500


class FirestoreVectorStore:
    """Same public interface as VectorStore — drop-in replacement."""

    def __init__(self) -> None:
        try:
            from google.cloud import firestore
            from google.cloud.firestore_v1.base_vector_query import DistanceMeasure
            from google.cloud.firestore_v1.vector import Vector
        except ImportError as exc:
            raise ImportError(
                "Install google-cloud-firestore: pip install google-cloud-firestore"
            ) from exc

        self._firestore = firestore
        self._DistanceMeasure = DistanceMeasure
        self._Vector = Vector

        self._db  = firestore.Client(project=config.gcp_project_id or None)
        self._col = self._db.collection(config.collection_name)
        self._ef  = DefaultEmbeddingFunction()

        logger.info(
            "FirestoreVectorStore ready (project=%s, collection=%s)",
            config.gcp_project_id or "default", config.collection_name,
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

        embeddings = self._ef(documents)

        for offset in range(0, len(documents), _BATCH_SIZE):
            batch = self._db.batch()
            for doc_id, content, meta, emb in zip(
                ids[offset:offset + _BATCH_SIZE],
                documents[offset:offset + _BATCH_SIZE],
                metadatas[offset:offset + _BATCH_SIZE],
                embeddings[offset:offset + _BATCH_SIZE],
            ):
                ref = self._col.document(doc_id)
                batch.set(ref, {
                    "content":  content,
                    "source":   meta.get("source", ""),   # top-level for WHERE filter
                    "metadata": meta,
                    "embedding": self._Vector(list(emb)),
                })
            batch.commit()

        logger.debug("Upserted %d chunks to Firestore", len(documents))

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------

    def search(self, query: str, n_results: int | None = None) -> list[dict[str, Any]]:
        n = min(n_results or config.top_k_retrieval, max(self.count(), 1))
        query_emb = self._ef([query])[0]

        vq = self._col.find_nearest(
            vector_field="embedding",
            query_vector=self._Vector(list(query_emb)),
            distance_measure=self._DistanceMeasure.COSINE,
            limit=n,
            distance_result_field="vector_distance",
        )

        results = []
        for doc in vq.get():
            distance = doc.get("vector_distance") or 0.0
            results.append({
                "content":         doc.get("content", ""),
                "metadata":        doc.get("metadata", {}),
                "relevance_score": round(1.0 - distance, 4),
            })
        return results

    def count(self) -> int:
        try:
            res = self._col.count().get()
            return res[0][0].value
        except Exception:
            return 0

    def list_sources(self) -> list[dict[str, Any]]:
        if self.count() == 0:
            return []

        tally: dict[str, dict[str, Any]] = {}
        for doc in self._col.select(["source", "metadata"]).stream():
            data = doc.to_dict() or {}
            key  = data.get("source") or "unknown"
            meta = data.get("metadata", {})
            if key not in tally:
                tally[key] = {
                    "source":   key,
                    "title":    meta.get("title") or meta.get("filename") or key,
                    "filetype": meta.get("filetype", "?"),
                    "count":    0,
                }
            tally[key]["count"] += 1

        return sorted(tally.values(), key=lambda x: x["title"].lower())

    def delete_source(self, source: str) -> int:
        docs = list(self._col.where("source", "==", source).stream())
        if not docs:
            return 0
        batch = self._db.batch()
        for doc in docs:
            batch.delete(doc.reference)
        batch.commit()
        return len(docs)

    def export_all(self) -> list[dict[str, Any]]:
        return [
            {
                "id":       doc.id,
                "content":  doc.get("content", ""),
                "metadata": doc.get("metadata", {}),
            }
            for doc in self._col.select(["content", "metadata"]).stream()
        ]

    def import_all(self, chunks: list[dict[str, Any]]) -> int:
        if not chunks:
            return 0
        self.upsert(
            documents=[c["content"] for c in chunks],
            metadatas=[c["metadata"] for c in chunks],
            ids=[c["id"] for c in chunks],
        )
        return len(chunks)
