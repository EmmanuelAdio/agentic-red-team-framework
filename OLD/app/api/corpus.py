from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query

from OLD.app.api.deps import get_services
from OLD.app.core.container import ServiceContainer
from OLD.app.corpus.schemas import (
    DocumentListResponse,
    DocumentSummary,
    IngestRequest,
    IngestResponse,
    PruneCorpusVersionsRequest,
    PruneCorpusVersionsResponse,
)

router = APIRouter(prefix="/corpus", tags=["corpus"])


@router.post("/ingest", response_model=IngestResponse)
def ingest_corpus(
    payload: IngestRequest,
    services: ServiceContainer = Depends(get_services),
) -> IngestResponse:
    """Ingest local corpus files into versioned document/chunk stores."""

    source_path = Path(payload.source_path) if payload.source_path else services.settings.default_corpus_path
    try:
        return services.ingestion_service.ingest_from_path(source_path)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@router.get("/documents", response_model=DocumentListResponse)
def list_documents(
    corpus_version: str | None = Query(default=None),
    services: ServiceContainer = Depends(get_services),
) -> DocumentListResponse:
    """List stored corpus documents, optionally filtered by corpus version."""

    rows = services.repositories.document_repo.list_documents(corpus_version=corpus_version)
    return DocumentListResponse(documents=[DocumentSummary.model_validate(row) for row in rows])


@router.post("/versions/prune", response_model=PruneCorpusVersionsResponse)
def prune_corpus_versions(
    payload: PruneCorpusVersionsRequest,
    services: ServiceContainer = Depends(get_services),
) -> PruneCorpusVersionsResponse:
    """Delete a user-requested number of older corpus versions to reclaim storage."""

    versions = services.repositories.corpus_version_repo.list_versions()
    eligible = versions[1:] if payload.keep_latest else versions
    # Delete oldest eligible versions first so the most recent data stays available longer.
    to_delete = eligible[-payload.delete_count :] if eligible else []
    delete_versions = [str(row["corpus_version"]) for row in to_delete]

    deleted_document_count = services.repositories.document_repo.delete_by_corpus_versions(delete_versions)
    deleted_chunk_count = services.repositories.chunk_repo.delete_by_corpus_versions(delete_versions)
    deleted_corpus_version_count = services.repositories.corpus_version_repo.delete_by_corpus_versions(delete_versions)

    # Safety net: remove any orphan docs/chunks whose corpus_version no longer exists.
    remaining_versions = services.repositories.corpus_version_repo.list_version_ids()
    deleted_document_count += services.repositories.document_repo.delete_not_in_corpus_versions(remaining_versions)
    deleted_chunk_count += services.repositories.chunk_repo.delete_not_in_corpus_versions(remaining_versions)

    return PruneCorpusVersionsResponse(
        requested_delete_count=payload.delete_count,
        deleted_version_count=len(delete_versions),
        deleted_versions=delete_versions,
        deleted_document_count=deleted_document_count,
        deleted_chunk_count=deleted_chunk_count,
        deleted_corpus_version_count=deleted_corpus_version_count,
    )
