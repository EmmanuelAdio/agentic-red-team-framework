from fastapi import APIRouter

from OLD.app.api.corpus import router as corpus_router
from OLD.app.api.debug import router as debug_router
from OLD.app.api.rag import router as rag_router
from OLD.app.api.traces import router as traces_router


def build_api_router() -> APIRouter:
    """Build top-level API router for all baseline endpoints."""

    router = APIRouter()
    router.include_router(corpus_router)
    router.include_router(debug_router)
    router.include_router(rag_router)
    router.include_router(traces_router)
    return router
