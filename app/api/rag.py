from fastapi import APIRouter, Depends, HTTPException

from app.api.deps import get_services
from app.core.container import ServiceContainer
from app.rag.schemas import QueryRequest, RAGResponse

router = APIRouter(prefix="/rag", tags=["rag"])


@router.post("/query", response_model=RAGResponse)
def query_rag(
    payload: QueryRequest,
    services: ServiceContainer = Depends(get_services),
) -> RAGResponse:
    """Run baseline retrieval + optional generation and return full traceable response."""

    try:
        return services.rag_service.answer_query(
            query=payload.query,
            top_k=payload.top_k,
            corpus_version=payload.corpus_version,
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
