from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from app.core.settings import Settings, get_settings
from app.corpus.chunker import TextChunker
from app.corpus.loader import CorpusLoader
from app.corpus.service import CorpusIngestionService
from app.db.client import MongoDatabaseFactory
from app.db.repositories import MongoRepositories, build_repositories
from app.rag.generator import LLMGenerator, build_generator
from app.rag.service import BaselineRAGService
from app.retrieval.embeddings import EmbeddingService, build_embedding_service
from app.retrieval.retriever import Retriever
from app.retrieval.vector_index import LocalVectorIndex, MongoVectorIndex, VectorIndex


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
        return MongoVectorIndex(repositories.chunk_repo)
    raise ValueError(f"Unsupported vector backend: {settings.vector_backend}")


@lru_cache
def get_container() -> ServiceContainer:
    """Construct and cache the full dependency graph."""

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

    return ServiceContainer(
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
