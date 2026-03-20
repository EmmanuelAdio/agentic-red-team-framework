from __future__ import annotations

from pathlib import Path

from app.core.utils import make_corpus_version, utc_now
from app.corpus.chunker import TextChunker
from app.corpus.loader import CorpusLoader
from app.corpus.schemas import ChunkRecord, CorpusVersionRecord, DocumentRecord, IngestResponse
from app.db.repositories import MongoRepositories
from app.retrieval.embeddings import EmbeddingService
from app.retrieval.vector_index import VectorIndex


class CorpusIngestionService:
    """Orchestrates corpus loading, chunking, embedding, and persistence."""

    def __init__(
        self,
        repositories: MongoRepositories,
        loader: CorpusLoader,
        chunker: TextChunker,
        embedding_service: EmbeddingService,
        vector_index: VectorIndex,
    ):
        self._repositories = repositories
        self._loader = loader
        self._chunker = chunker
        self._embedding_service = embedding_service
        self._vector_index = vector_index

    def ingest_from_path(self, source_path: Path) -> IngestResponse:
        if not source_path.exists() or not source_path.is_dir():
            raise ValueError(f"Invalid corpus path: {source_path}")

        now = utc_now()
        corpus_version = make_corpus_version()
        loaded_docs = self._loader.load_documents(source_path)

        document_records: list[DocumentRecord] = []
        chunk_records: list[ChunkRecord] = []

        for doc in loaded_docs:
            document_records.append(
                DocumentRecord(
                    doc_id=doc.doc_id,
                    title=doc.title,
                    source_type=doc.source_type,
                    attack_label=doc.attack_label,
                    corpus_version=corpus_version,
                    raw_text=doc.raw_text,
                    metadata=doc.metadata,
                    created_at=now,
                )
            )

            chunks = self._chunker.split(doc.raw_text)
            for index, chunk_text in enumerate(chunks):
                chunk_id = f"{doc.doc_id}::chunk::{index}"
                embedding = self._embedding_service.embed_text(chunk_text)
                chunk_records.append(
                    ChunkRecord(
                        chunk_id=chunk_id,
                        doc_id=doc.doc_id,
                        title=doc.title,
                        chunk_text=chunk_text,
                        chunk_index=index,
                        source_type=doc.source_type,
                        attack_label=doc.attack_label,
                        corpus_version=corpus_version,
                        embedding=embedding,
                        metadata=doc.metadata,
                        created_at=now,
                    )
                )

        self._repositories.document_repo.insert_many(document_records)
        self._repositories.chunk_repo.insert_many(chunk_records)
        self._repositories.corpus_version_repo.insert_one(
            CorpusVersionRecord(
                corpus_version=corpus_version,
                source_path=str(source_path),
                document_count=len(document_records),
                chunk_count=len(chunk_records),
                created_at=now,
            )
        )

        self._vector_index.index_chunks([record.model_dump(mode="python") for record in chunk_records])

        return IngestResponse(
            corpus_version=corpus_version,
            source_path=str(source_path),
            document_count=len(document_records),
            chunk_count=len(chunk_records),
        )
