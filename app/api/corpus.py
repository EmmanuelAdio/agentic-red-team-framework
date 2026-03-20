from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query

from app.api.deps import get_services
from app.core.container import ServiceContainer
from app.corpus.schemas import DocumentListResponse, DocumentSummary, IngestRequest, IngestResponse

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
