from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Any

from app.db.repositories import ChunkRepository


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity for two same-length vectors."""

    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class VectorIndex(ABC):
    """Interface for pluggable vector retrieval backends."""

    @abstractmethod
    def index_chunks(self, chunks: list[dict[str, Any]]) -> None:
        """Optional indexing hook after ingestion."""

    @abstractmethod
    def search(
        self,
        query_vector: list[float],
        top_k: int,
        corpus_version: str | None = None,
    ) -> list[dict[str, Any]]:
        """Return top-k chunk records with score metadata."""


class LocalVectorIndex(VectorIndex):
    """In-process cosine retrieval over persisted chunk embeddings."""

    def __init__(self, chunk_repo: ChunkRepository):
        self._chunk_repo = chunk_repo

    def index_chunks(self, chunks: list[dict[str, Any]]) -> None:
        # Embeddings are persisted in Mongo during ingestion, so no extra local state is required.
        return None

    def search(
        self,
        query_vector: list[float],
        top_k: int,
        corpus_version: str | None = None,
    ) -> list[dict[str, Any]]:
        if corpus_version:
            candidates = self._chunk_repo.list_with_embeddings_for_version(corpus_version)
        else:
            candidates = self._chunk_repo.list_all_with_embeddings()

        scored: list[dict[str, Any]] = []
        for candidate in candidates:
            score = cosine_similarity(query_vector, candidate.get("embedding", []))
            item = {**candidate, "score": float(score)}
            scored.append(item)

        scored.sort(key=lambda item: item["score"], reverse=True)
        return scored[:top_k]


class MongoVectorIndex(VectorIndex):
    """Mongo-backed retrieval abstraction.

    This baseline implementation computes cosine similarity in Python after fetching
    candidate vectors from Mongo, which keeps the data path Mongo-centric while
    remaining portable to local Mongo setups.

    TODO(provider): swap search() with Atlas `$vectorSearch` when available.
    """

    def __init__(self, chunk_repo: ChunkRepository):
        self._chunk_repo = chunk_repo

    def index_chunks(self, chunks: list[dict[str, Any]]) -> None:
        # Existing chunk records already include embeddings, so no additional action is needed.
        return None

    def search(
        self,
        query_vector: list[float],
        top_k: int,
        corpus_version: str | None = None,
    ) -> list[dict[str, Any]]:
        if corpus_version:
            candidates = self._chunk_repo.list_with_embeddings_for_version(corpus_version)
        else:
            candidates = self._chunk_repo.list_all_with_embeddings()

        scored: list[dict[str, Any]] = []
        for candidate in candidates:
            score = cosine_similarity(query_vector, candidate.get("embedding", []))
            scored.append({**candidate, "score": float(score)})

        scored.sort(key=lambda item: item["score"], reverse=True)
        return scored[:top_k]
