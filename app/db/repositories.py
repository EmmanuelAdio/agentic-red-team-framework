from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pymongo.database import Database

from app.corpus.schemas import ChunkRecord, CorpusVersionRecord, DocumentRecord
from app.rag.schemas import QueryTrace, RAGResponse


@dataclass
class MongoRepositories:
    """Bundle of Mongo repositories used by ingestion and RAG services."""

    document_repo: "DocumentRepository"
    chunk_repo: "ChunkRepository"
    corpus_version_repo: "CorpusVersionRepository"
    query_trace_repo: "QueryTraceRepository"
    rag_response_repo: "RAGResponseRepository"


class DocumentRepository:
    """CRUD operations for corpus document records."""

    def __init__(self, db: Database):
        self.collection = db["documents"]
        self.collection.create_index([("doc_id", 1), ("corpus_version", 1)], unique=True)
        self.collection.create_index([("corpus_version", 1), ("created_at", -1)])

    def insert_many(self, records: list[DocumentRecord]) -> None:
        if not records:
            return
        self.collection.insert_many([record.model_dump(mode="json") for record in records])

    def list_documents(self, corpus_version: str | None = None) -> list[dict[str, Any]]:
        query: dict[str, Any] = {}
        if corpus_version:
            query["corpus_version"] = corpus_version
        cursor = self.collection.find(query, {"_id": 0}).sort("created_at", -1)
        return list(cursor)


class ChunkRepository:
    """Storage and query operations for chunk-level records."""

    def __init__(self, db: Database):
        self.collection = db["chunks"]
        self.collection.create_index([("chunk_id", 1), ("corpus_version", 1)], unique=True)
        self.collection.create_index([("corpus_version", 1), ("doc_id", 1)])

    def insert_many(self, records: list[ChunkRecord]) -> None:
        if not records:
            return
        self.collection.insert_many([record.model_dump(mode="json") for record in records])

    def list_by_corpus_version(self, corpus_version: str) -> list[dict[str, Any]]:
        return list(self.collection.find({"corpus_version": corpus_version}, {"_id": 0}))

    def list_all_with_embeddings(self) -> list[dict[str, Any]]:
        return list(self.collection.find({"embedding.0": {"$exists": True}}, {"_id": 0}))

    def list_with_embeddings_for_version(self, corpus_version: str) -> list[dict[str, Any]]:
        return list(
            self.collection.find(
                {"corpus_version": corpus_version, "embedding.0": {"$exists": True}},
                {"_id": 0},
            )
        )


class CorpusVersionRepository:
    """Manages immutable ingestion version snapshots."""

    def __init__(self, db: Database):
        self.collection = db["corpus_versions"]
        self.collection.create_index([("corpus_version", 1)], unique=True)
        self.collection.create_index([("created_at", -1)])

    def insert_one(self, record: CorpusVersionRecord) -> None:
        self.collection.insert_one(record.model_dump(mode="json"))

    def latest(self) -> dict[str, Any] | None:
        return self.collection.find_one({}, {"_id": 0}, sort=[("created_at", -1)])

    def get(self, corpus_version: str) -> dict[str, Any] | None:
        return self.collection.find_one({"corpus_version": corpus_version}, {"_id": 0})


class QueryTraceRepository:
    """Persists query-time retrieval traces for evaluation later."""

    def __init__(self, db: Database):
        self.collection = db["query_traces"]
        self.collection.create_index([("trace_id", 1)], unique=True)
        self.collection.create_index([("created_at", -1)])

    def insert_one(self, trace: QueryTrace) -> None:
        self.collection.insert_one(trace.model_dump(mode="json"))

    def get(self, trace_id: str) -> dict[str, Any] | None:
        return self.collection.find_one({"trace_id": trace_id}, {"_id": 0})


class RAGResponseRepository:
    """Stores full RAG responses for experiment logging and replay."""

    def __init__(self, db: Database):
        self.collection = db["rag_responses"]
        self.collection.create_index([("trace_id", 1)], unique=True)
        self.collection.create_index([("created_at", -1)])

    def insert_one(self, response: RAGResponse) -> None:
        self.collection.insert_one(response.model_dump(mode="json"))

    def get(self, trace_id: str) -> dict[str, Any] | None:
        return self.collection.find_one({"trace_id": trace_id}, {"_id": 0})


def build_repositories(db: Database) -> MongoRepositories:
    """Construct all repositories from a shared database handle."""

    return MongoRepositories(
        document_repo=DocumentRepository(db),
        chunk_repo=ChunkRepository(db),
        corpus_version_repo=CorpusVersionRepository(db),
        query_trace_repo=QueryTraceRepository(db),
        rag_response_repo=RAGResponseRepository(db),
    )
