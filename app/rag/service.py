from __future__ import annotations

from app.core.settings import Settings
from app.core.utils import make_trace_id, utc_now
from app.db.repositories import MongoRepositories
from app.rag.generator import LLMGenerator
from app.rag.prompting import build_prompt, build_prompt_context
from app.rag.schemas import ModelMetadata, QueryTrace, RAGResponse
from app.retrieval.retriever import Retriever


class BaselineRAGService:
    """Runs retrieval-augmented answering and persists full query traces."""

    def __init__(
        self,
        settings: Settings,
        repositories: MongoRepositories,
        retriever: Retriever,
        generator: LLMGenerator,
    ):
        self._settings = settings
        self._repositories = repositories
        self._retriever = retriever
        self._generator = generator

    def answer_query(
        self,
        query: str,
        top_k: int | None = None,
        corpus_version: str | None = None,
    ) -> RAGResponse:
        selected_version = corpus_version or self._latest_corpus_version()
        if selected_version is None:
            raise ValueError("No corpus version found. Ingest corpus before querying.")

        effective_top_k = top_k or self._settings.default_top_k
        retrieved = self._retriever.retrieve(
            query=query,
            top_k=effective_top_k,
            corpus_version=selected_version,
        )

        prompt_context = build_prompt_context(retrieved)
        prompt_used = build_prompt(query=query, prompt_context=prompt_context)

        generation_skipped_reason: str | None = None
        final_answer: str | None = None

        if self._settings.query_mode == "retrieve_only" or self._settings.llm_provider == "none":
            generation_skipped_reason = "retrieve_only_mode"
        else:
            final_answer = self._generator.generate(prompt_used)

        now = utc_now()
        trace_id = make_trace_id()

        model_metadata = ModelMetadata(
            embedding_provider=self._settings.embedding_provider,
            embedding_model=self._settings.embedding_model,
            llm_provider=self._settings.llm_provider,
            llm_model=self._settings.llm_model,
            query_mode=self._settings.query_mode,
            generation_skipped_reason=generation_skipped_reason,
        )

        trace = QueryTrace(
            trace_id=trace_id,
            query=query,
            top_k=effective_top_k,
            corpus_version=selected_version,
            retrieval_backend=self._settings.vector_backend,
            retrieved_chunks=retrieved,
            prompt_context=prompt_context,
            prompt_used=prompt_used,
            model_metadata=model_metadata,
            created_at=now,
        )

        response = RAGResponse(
            trace_id=trace_id,
            query=query,
            final_answer=final_answer,
            retrieved_chunks=retrieved,
            prompt_context=prompt_context,
            prompt_used=prompt_used,
            model_metadata=model_metadata,
            corpus_version=selected_version,
            retrieval_backend=self._settings.vector_backend,
            created_at=now,
        )

        self._repositories.query_trace_repo.insert_one(trace)
        self._repositories.rag_response_repo.insert_one(response)
        return response

    def get_trace(self, trace_id: str) -> QueryTrace:
        payload = self._repositories.query_trace_repo.get(trace_id)
        if payload is None:
            raise ValueError(f"Trace not found: {trace_id}")
        return QueryTrace.model_validate(payload)

    def _latest_corpus_version(self) -> str | None:
        latest = self._repositories.corpus_version_repo.latest()
        if latest is None:
            return None
        return str(latest["corpus_version"])
