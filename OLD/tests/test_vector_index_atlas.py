from __future__ import annotations

from typing import Any

from OLD.app.core.settings import Settings
from OLD.app.retrieval.vector_index import MongoVectorIndex


class FakeCollection:
    def __init__(self):
        self.pipeline_calls: list[list[dict[str, Any]]] = []
        self.results: list[dict[str, Any]] = []
        self.error: Exception | None = None
        self.name = "chunks"
        self.database = FakeDatabase()
        self.search_indexes: list[dict[str, Any]] = [
            {
                "name": "chunks_embedding_index",
                "status": "READY",
                "queryable": True,
                "latestDefinitionVersion": {"version": 1},
                "latestQueryableVersion": {"version": 1},
            }
        ]
        self.list_search_indexes_calls: list[str] = []

    def aggregate(self, pipeline: list[dict[str, Any]]):
        self.pipeline_calls.append(pipeline)
        if self.error is not None:
            raise self.error
        return self.results

    def list_search_indexes(self, name: str):
        self.list_search_indexes_calls.append(name)
        return [idx for idx in self.search_indexes if idx.get("name") == name]


class FakeDatabase:
    def __init__(self):
        self.command_calls: list[dict[str, Any]] = []

    def command(self, payload: dict[str, Any]):
        self.command_calls.append(payload)
        return {"ok": 1}


class FakeChunkRepo:
    def __init__(self):
        self.collection = FakeCollection()
        self.version_candidates: list[dict[str, Any]] = []
        self.all_candidates: list[dict[str, Any]] = []
        self.version_calls = 0
        self.all_calls = 0

    def list_with_embeddings_for_version(self, corpus_version: str):
        self.version_calls += 1
        return self.version_candidates

    def list_all_with_embeddings(self):
        self.all_calls += 1
        return self.all_candidates


def test_mongo_vector_index_uses_atlas_vector_search_when_available():
    repo = FakeChunkRepo()
    repo.collection.results = [
        {
            "chunk_id": "c1",
            "doc_id": "d1",
            "title": "t1",
            "chunk_text": "text",
            "attack_label": "benign",
            "corpus_version": "v1",
            "score": 0.91,
        }
    ]

    settings = Settings(atlas_vector_index_name="chunks_embedding_index")
    index = MongoVectorIndex(repo, settings)
    result = index.search(query_vector=[0.1, 0.2], top_k=3, corpus_version="v1")

    assert result[0]["chunk_id"] == "c1"
    assert repo.version_calls == 0
    assert repo.all_calls == 0
    pipeline = repo.collection.pipeline_calls[0]
    vector_stage = pipeline[0]["$vectorSearch"]
    assert vector_stage["index"] == "chunks_embedding_index"
    assert vector_stage["filter"] == {"corpus_version": "v1"}
    assert vector_stage["limit"] == 3
    assert repo.collection.list_search_indexes_calls == ["chunks_embedding_index"]
    assert repo.collection.database.command_calls == []


def test_mongo_vector_index_falls_back_when_vector_search_unavailable_and_caches():
    repo = FakeChunkRepo()
    repo.collection.error = Exception("Unrecognized pipeline stage name: '$vectorSearch'")
    repo.all_candidates = [
        {"chunk_id": "a", "embedding": [1.0, 0.0], "corpus_version": "v1"},
        {"chunk_id": "b", "embedding": [0.0, 1.0], "corpus_version": "v1"},
    ]
    settings = Settings(atlas_vector_index_name="chunks_embedding_index")
    index = MongoVectorIndex(repo, settings)

    first = index.search(query_vector=[1.0, 0.0], top_k=1)
    second = index.search(query_vector=[1.0, 0.0], top_k=1)

    assert first[0]["chunk_id"] == "a"
    assert second[0]["chunk_id"] == "a"
    assert len(repo.collection.pipeline_calls) == 1
    assert repo.all_calls == 2


def test_mongo_vector_index_creates_index_when_missing():
    repo = FakeChunkRepo()
    repo.collection.search_indexes = []
    repo.collection.results = [
        {
            "chunk_id": "c1",
            "doc_id": "d1",
            "title": "t1",
            "chunk_text": "text",
            "attack_label": "benign",
            "corpus_version": "v1",
            "score": 0.9,
        }
    ]

    settings = Settings(atlas_vector_index_name="chunks_embedding_index", embedding_dimension=1536)
    index = MongoVectorIndex(repo, settings)
    index.search(query_vector=[0.1, 0.2], top_k=1)

    assert len(repo.collection.database.command_calls) == 1
    payload = repo.collection.database.command_calls[0]
    assert payload["createSearchIndexes"] == "chunks"
    assert payload["indexes"][0]["name"] == "chunks_embedding_index"
    vector_field = payload["indexes"][0]["definition"]["fields"][0]
    assert vector_field["path"] == "embedding"
    assert vector_field["numDimensions"] == 1536
    assert vector_field["similarity"] == "cosine"


def test_mongo_vector_index_raises_non_availability_errors():
    repo = FakeChunkRepo()
    repo.collection.error = RuntimeError("network timeout")
    settings = Settings(atlas_vector_index_name="chunks_embedding_index")
    index = MongoVectorIndex(repo, settings)

    try:
        index.search(query_vector=[0.2, 0.8], top_k=2)
    except RuntimeError as exc:
        assert "network timeout" in str(exc)
    else:
        raise AssertionError("Expected RuntimeError to be raised")


def test_mongo_vector_index_diagnostics_reports_ready_index():
    repo = FakeChunkRepo()
    settings = Settings(atlas_vector_index_name="chunks_embedding_index", embedding_dimension=1536)
    index = MongoVectorIndex(repo, settings)

    diagnostics = index.diagnostics()

    assert diagnostics["backend"] == "mongo"
    assert diagnostics["atlas_vector_index_name"] == "chunks_embedding_index"
    assert diagnostics["search_index_exists"] is True
    assert diagnostics["search_index_state"] == "READY"
    assert diagnostics["search_index_queryable"] is True


def test_mongo_vector_index_diagnostics_reports_missing_index():
    repo = FakeChunkRepo()
    repo.collection.search_indexes = []
    settings = Settings(atlas_vector_index_name="chunks_embedding_index", embedding_dimension=1536)
    index = MongoVectorIndex(repo, settings)

    diagnostics = index.diagnostics()

    assert diagnostics["search_index_exists"] is False
    assert diagnostics["search_index_state"] == "missing"
