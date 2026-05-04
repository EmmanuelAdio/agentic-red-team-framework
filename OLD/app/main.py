from contextlib import asynccontextmanager

from fastapi import FastAPI

from OLD.app.api.router import build_api_router
from OLD.app.core.container import get_container
from OLD.app.core.settings import get_settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Ensure DB client is closed when the application shuts down."""

    container = get_container()
    try:
        yield
    finally:
        container.db_factory.close()


def create_app() -> FastAPI:
    """FastAPI application factory for the baseline RAG service."""

    settings = get_settings()
    app = FastAPI(title=settings.app_name, lifespan=lifespan)
    app.include_router(build_api_router(), prefix=settings.api_prefix)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    return app


app = create_app()
