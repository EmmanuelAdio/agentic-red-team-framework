from fastapi import APIRouter, Depends, HTTPException

from OLD.app.api.deps import get_services
from OLD.app.core.container import ServiceContainer
from OLD.app.rag.schemas import TraceResponse

router = APIRouter(prefix="/traces", tags=["traces"])


@router.get("/{trace_id}", response_model=TraceResponse)
def get_trace(
    trace_id: str,
    services: ServiceContainer = Depends(get_services),
) -> TraceResponse:
    """Fetch a persisted query retrieval trace by trace ID."""

    try:
        trace = services.rag_service.get_trace(trace_id)
    except ValueError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    return TraceResponse(trace=trace)
