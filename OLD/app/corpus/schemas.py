from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class AttackLabel(str, Enum):
    """Classification label used to tag corpus records for experiments."""

    benign = "benign"
    poisoned = "poisoned"
    misleading = "misleading"


class SourceType(str, Enum):
    """Source type for raw corpus files."""

    txt = "txt"
    json = "json"


class DocumentRecord(BaseModel):
    """Stored document-level corpus record."""

    doc_id: str
    title: str
    source_type: SourceType
    attack_label: AttackLabel
    corpus_version: str
    raw_text: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime


class ChunkRecord(BaseModel):
    """Stored chunk-level record with retrieval metadata."""

    chunk_id: str
    doc_id: str
    title: str
    chunk_text: str
    chunk_index: int
    source_type: SourceType
    attack_label: AttackLabel
    corpus_version: str
    embedding: list[float] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime


class CorpusVersionRecord(BaseModel):
    """Snapshot metadata for a single ingestion run."""

    corpus_version: str
    source_path: str
    document_count: int
    chunk_count: int
    created_at: datetime


class IngestRequest(BaseModel):
    """API request for ingesting a local corpus folder."""

    source_path: str | None = None


class IngestResponse(BaseModel):
    """API response describing an ingestion snapshot."""

    corpus_version: str
    source_path: str
    document_count: int
    chunk_count: int


class DocumentSummary(BaseModel):
    """Lightweight document metadata for list endpoints."""

    doc_id: str
    title: str
    source_type: SourceType
    attack_label: AttackLabel
    corpus_version: str
    created_at: datetime


class DocumentListResponse(BaseModel):
    """List response for corpus documents endpoint."""

    documents: list[DocumentSummary]


class PruneCorpusVersionsRequest(BaseModel):
    """Request payload for deleting historical corpus versions."""

    delete_count: int = Field(ge=1)
    keep_latest: bool = True


class PruneCorpusVersionsResponse(BaseModel):
    """Summary of deleted corpus-version records and related data."""

    requested_delete_count: int
    deleted_version_count: int
    deleted_versions: list[str] = Field(default_factory=list)
    deleted_document_count: int
    deleted_chunk_count: int
    deleted_corpus_version_count: int
