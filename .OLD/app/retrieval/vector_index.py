from __future__ import annotations

import math
import time
from abc import ABC, abstractmethod
from typing import Any

from OLD.app.core.settings import Settings
from OLD.app.db.repositories import ChunkRepository


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

    def diagnostics(self) -> dict[str, Any]:
        """Return backend-specific diagnostics for operational debugging."""

        return {"backend": "unknown"}


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

    def diagnostics(self) -> dict[str, Any]:
        return {
            "backend": "local",
            "uses_atlas_vector_search": False,
            "message": "Local backend computes cosine similarity in Python over Mongo-stored embeddings.",
        }


class MongoVectorIndex(VectorIndex):
    """Mongo-backed retrieval abstraction.

    Uses Atlas `$vectorSearch` when supported, with a local cosine fallback for
    portable local Mongo/mongomock setups.
    """

    def __init__(self, chunk_repo: ChunkRepository, settings: Settings):
        self._chunk_repo = chunk_repo
        self._atlas_vector_index = settings.atlas_vector_index_name
        self._embedding_dimension = settings.embedding_dimension
        self._vector_search_available: bool | None = None
        self._vector_index_checked = False
        self._next_vector_retry_after = 0.0

    def index_chunks(self, chunks: list[dict[str, Any]]) -> None:
        # Existing chunk records already include embeddings, so no additional action is needed.
        return None

    def search(
        self,
        query_vector: list[float],
        top_k: int,
        corpus_version: str | None = None,
    ) -> list[dict[str, Any]]:
        if time.time() < self._next_vector_retry_after:
            return self._search_with_python_cosine(query_vector, top_k, corpus_version)

        if self._vector_search_available is not False:
            try:
                self._ensure_vector_search_index()
                results = self._search_with_atlas_vector(query_vector, top_k, corpus_version)
                self._vector_search_available = True
                return results
            except Exception as exc:
                if self._is_vector_search_temporarily_unavailable(exc):
                    # Atlas index can be present but still building (e.g., INITIAL_SYNC).
                    # Fall back for now and retry Atlas after a short cooldown.
                    self._next_vector_retry_after = time.time() + 30.0
                    return self._search_with_python_cosine(query_vector, top_k, corpus_version)
                if self._is_vector_search_unavailable(exc):
                    self._vector_search_available = False
                else:
                    raise

        return self._search_with_python_cosine(query_vector, top_k, corpus_version)

    def _search_with_atlas_vector(
        self,
        query_vector: list[float],
        top_k: int,
        corpus_version: str | None,
    ) -> list[dict[str, Any]]:
        vector_search_stage: dict[str, Any] = {
            "index": self._atlas_vector_index,
            "path": "embedding",
            "queryVector": query_vector,
            "numCandidates": max(top_k * 10, top_k),
            "limit": top_k,
        }
        if corpus_version:
            vector_search_stage["filter"] = {"corpus_version": corpus_version}

        pipeline: list[dict[str, Any]] = [
            {"$vectorSearch": vector_search_stage}, 
            {
                "$project": {
                    "_id": 0,
                    "chunk_id": 1,
                    "doc_id": 1,
                    "title": 1,
                    "chunk_text": 1,
                    "attack_label": 1,
                    "corpus_version": 1,
                    "score": {"$meta": "vectorSearchScore"},
                }
            },
        ]
        return list(self._chunk_repo.collection.aggregate(pipeline))

    def _search_with_python_cosine(
        self,
        query_vector: list[float],
        top_k: int,
        corpus_version: str | None,
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

    def _ensure_vector_search_index(self) -> None:
        """Ensure Atlas vector index exists before running `$vectorSearch`."""

        if self._vector_index_checked:
            return
        self._vector_index_checked = True

        collection = self._chunk_repo.collection
        if not hasattr(collection, "list_search_indexes"):
            return

        try:
            existing = list(collection.list_search_indexes(self._atlas_vector_index))
            if existing:
                return
        except Exception as exc:
            if self._is_vector_search_unavailable(exc):
                return
            raise

        if self._embedding_dimension <= 0:
            return

        index_definition = {
            "name": self._atlas_vector_index,
            "type": "vectorSearch",
            "definition": {
                "fields": [
                    {
                        "type": "vector",
                        "path": "embedding",
                        "numDimensions": self._embedding_dimension,
                        "similarity": "cosine",
                    },
                    {
                        "type": "filter",
                        "path": "corpus_version",
                    },
                ]
            },
        }
        db = getattr(collection, "database", None)
        collection_name = getattr(collection, "name", None)
        if db is None or collection_name is None:
            return

        try:
            db.command(
                {
                    "createSearchIndexes": collection_name,
                    "indexes": [index_definition],
                }
            )
        except Exception as exc:
            if self._is_vector_search_unavailable(exc):
                return
            raise

    @staticmethod
    def _is_vector_search_unavailable(exc: Exception) -> bool:
        message = str(exc).lower()
        return (
            "$vectorsearch" in message
            or "unrecognized pipeline stage name" in message
            or "is not allowed" in message
            or "atlas search index" in message
            or "index not found" in message
            or "search index" in message
            or "command not found" in message
            or "not supported" in message
        )

    @staticmethod
    def _is_vector_search_temporarily_unavailable(exc: Exception) -> bool:
        message = str(exc).lower()
        return (
            "initial_sync" in message
            or "initial sync" in message
            or "cannot query vector index" in message and "state" in message
            or "queryable" in message and "false" in message
            or "building" in message and "index" in message
        )

    def diagnostics(self) -> dict[str, Any]:
        now = time.time()
        retry_after_seconds = max(0.0, self._next_vector_retry_after - now)
        payload: dict[str, Any] = {
            "backend": "mongo",
            "uses_atlas_vector_search": True,
            "atlas_vector_index_name": self._atlas_vector_index,
            "embedding_dimension": self._embedding_dimension,
            "vector_search_available_cache": self._vector_search_available,
            "vector_index_checked": self._vector_index_checked,
            "fallback_cooldown_active": retry_after_seconds > 0,
            "retry_atlas_after_seconds": round(retry_after_seconds, 3),
        }

        collection = self._chunk_repo.collection
        if not hasattr(collection, "list_search_indexes"):
            payload["atlas_search_api_available"] = False
            payload["message"] = "Collection driver does not expose list_search_indexes."
            return payload

        payload["atlas_search_api_available"] = True
        try:
            indexes = list(collection.list_search_indexes(self._atlas_vector_index))
        except Exception as exc:
            payload["search_index_lookup_error"] = str(exc)
            return payload

        payload["search_index_exists"] = len(indexes) > 0
        if not indexes:
            payload["search_index_state"] = "missing"
            return payload

        index_doc = indexes[0]
        status = index_doc.get("status")
        queryable = index_doc.get("queryable")
        payload["search_index_state"] = status or "unknown"
        payload["search_index_queryable"] = queryable if isinstance(queryable, bool) else None
        payload["search_index_latest_definition_version"] = index_doc.get("latestDefinitionVersion")
        payload["search_index_latest_queryable_version"] = index_doc.get("latestQueryableVersion")
        return payload
