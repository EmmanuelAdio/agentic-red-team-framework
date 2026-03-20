from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from app.corpus.schemas import AttackLabel


class QueryRequest(BaseModel):
    """Incoming user query for baseline RAG."""

    query: str
    top_k: int | None = None
    corpus_version: str | None = None


class RetrievedChunk(BaseModel):
    """Single retrieval result returned to clients and traces."""

    chunk_id: str
    doc_id: str
    title: str
    chunk_text: str
    score: float
    attack_label: AttackLabel
    corpus_version: str


class ModelMetadata(BaseModel):
    """Details of embedding/generation modes used for a query."""

    embedding_provider: str
    embedding_model: str
    llm_provider: str
    llm_model: str
    query_mode: str
    generation_skipped_reason: str | None = None


class RAGResponse(BaseModel):
    """Structured baseline RAG response contract."""

    trace_id: str
    query: str
    final_answer: str | None
    retrieved_chunks: list[RetrievedChunk]
    prompt_context: str
    prompt_used: str
    model_metadata: ModelMetadata
    corpus_version: str
    retrieval_backend: str
    created_at: datetime


class QueryTrace(BaseModel):
    """Persisted trace data for reproducible query auditing."""

    trace_id: str
    query: str
    top_k: int
    corpus_version: str
    retrieval_backend: str
    retrieved_chunks: list[RetrievedChunk]
    prompt_context: str
    prompt_used: str
    model_metadata: ModelMetadata
    created_at: datetime


class TraceResponse(BaseModel):
    """API response wrapper for trace inspection endpoint."""

    trace: QueryTrace


class StoredResponseRecord(BaseModel):
    """Database shape for stored RAG responses."""

    trace_id: str
    response: dict[str, Any]
    created_at: datetime
