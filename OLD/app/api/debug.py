from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends

from OLD.app.api.deps import get_services
from OLD.app.core.container import ServiceContainer

router = APIRouter(prefix="/debug", tags=["debug"])


@router.get("/vector-index")
def debug_vector_index(
    services: ServiceContainer = Depends(get_services),
) -> dict[str, Any]:
    """Return runtime diagnostics for the configured retrieval backend."""

    diagnostics = services.vector_index.diagnostics()
    diagnostics["configured_vector_backend"] = services.settings.vector_backend
    diagnostics["mongodb_db_name"] = services.settings.mongodb_db_name
    return diagnostics
