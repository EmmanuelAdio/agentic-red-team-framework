from __future__ import annotations

from dataclasses import dataclass
from threading import RLock

from OLD.app.core.settings import Settings, clear_settings_cache, get_settings, get_settings_cache_key
from OLD.app.corpus.chunker import TextChunker
from OLD.app.corpus.loader import CorpusLoader
from OLD.app.corpus.service import CorpusIngestionService
from OLD.app.db.client import MongoDatabaseFactory
from OLD.app.db.repositories import MongoRepositories, build_repositories
from OLD.app.rag.generator import LLMGenerator, build_generator
from OLD.app.rag.service import BaselineRAGService
from OLD.app.retrieval.embeddings import EmbeddingService, build_embedding_service
from OLD.app.retrieval.retriever import Retriever
from OLD.app.retrieval.vector_index import LocalVectorIndex, MongoVectorIndex, VectorIndex


@dataclass
class ServiceContainer:
    """Shared service registry for FastAPI dependencies."""

    settings: Settings
    db_factory: MongoDatabaseFactory
    repositories: MongoRepositories
    embedding_service: EmbeddingService
    vector_index: VectorIndex
    retriever: Retriever
    generator: LLMGenerator
    ingestion_service: CorpusIngestionService
    rag_service: BaselineRAGService


def _build_vector_index(settings: Settings, repositories: MongoRepositories) -> VectorIndex:
    if settings.vector_backend == "local":
        return LocalVectorIndex(repositories.chunk_repo)
    if settings.vector_backend == "mongo":
        return MongoVectorIndex(repositories.chunk_repo, settings)
    raise ValueError(f"Unsupported vector backend: {settings.vector_backend}")


def get_container() -> ServiceContainer:
    """Construct and cache the dependency graph, refreshing when `.env` changes."""

    global _cached_container, _cached_container_key

    cache_key = get_settings_cache_key()
    with _container_lock:
        if _cached_container is None or _cached_container_key != cache_key:
            if _cached_container is not None:
                _cached_container.db_factory.close()

            # Refresh settings first so downstream services use the latest env.
            clear_settings_cache()
            settings = get_settings()
            db_factory = MongoDatabaseFactory(settings)
            repositories = build_repositories(db_factory.database)

            embedding_service = build_embedding_service(settings)
            vector_index = _build_vector_index(settings, repositories)
            retriever = Retriever(embedding_service=embedding_service, vector_index=vector_index)
            generator = build_generator(settings)

            ingestion_service = CorpusIngestionService(
                repositories=repositories,
                loader=CorpusLoader(),
                chunker=TextChunker(settings),
                embedding_service=embedding_service,
                vector_index=vector_index,
            )
            rag_service = BaselineRAGService(
                settings=settings,
                repositories=repositories,
                retriever=retriever,
                generator=generator,
            )

            _cached_container = ServiceContainer(
                settings=settings,
                db_factory=db_factory,
                repositories=repositories,
                embedding_service=embedding_service,
                vector_index=vector_index,
                retriever=retriever,
                generator=generator,
                ingestion_service=ingestion_service,
                rag_service=rag_service,
            )
            _cached_container_key = cache_key

        return _cached_container


def clear_container_cache() -> None:
    """Close and clear the cached container graph."""

    global _cached_container, _cached_container_key

    with _container_lock:
        if _cached_container is not None:
            _cached_container.db_factory.close()
        _cached_container = None
        _cached_container_key = None


_container_lock = RLock()
_cached_container: ServiceContainer | None = None
_cached_container_key: int | None = None

# Backwards-compatible test API: existing tests call `get_container.cache_clear()`.
get_container.cache_clear = clear_container_cache  # type: ignore[attr-defined]
