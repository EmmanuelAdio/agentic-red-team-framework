from __future__ import annotations

from app.corpus.schemas import AttackLabel
from app.retrieval.embeddings import EmbeddingService
from app.retrieval.vector_index import VectorIndex
from app.rag.schemas import RetrievedChunk


class Retriever:
    """Retrieves the most relevant corpus chunks for a user query."""

    def __init__(self, embedding_service: EmbeddingService, vector_index: VectorIndex):
        self._embedding_service = embedding_service
        self._vector_index = vector_index

    def retrieve(
        self,
        query: str,
        top_k: int,
        corpus_version: str | None = None,
    ) -> list[RetrievedChunk]:
        query_vector = self._embedding_service.embed_text(query)
        rows = self._vector_index.search(query_vector=query_vector, top_k=top_k, corpus_version=corpus_version)
        return [
            RetrievedChunk(
                chunk_id=row["chunk_id"],
                doc_id=row["doc_id"],
                title=row["title"],
                chunk_text=row["chunk_text"],
                score=float(row.get("score", 0.0)),
                attack_label=AttackLabel(row["attack_label"]),
                corpus_version=row["corpus_version"],
            )
            for row in rows
        ]
