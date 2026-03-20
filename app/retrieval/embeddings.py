from __future__ import annotations

import hashlib
import math
import os
from abc import ABC, abstractmethod

from app.core.settings import Settings


class EmbeddingService(ABC):
    """Interface for embedding providers used by retrieval."""

    @abstractmethod
    def embed_text(self, text: str) -> list[float]:
        """Embed one text string into a dense vector."""


class DeterministicStubEmbeddingService(EmbeddingService):
    """Deterministic token hashing embedding for local reproducible baselines."""

    def __init__(self, settings: Settings):
        self._dimension = settings.embedding_dimension

    def embed_text(self, text: str) -> list[float]:
        vector = [0.0] * self._dimension
        for token in text.lower().split():
            digest = hashlib.sha256(token.encode("utf-8")).hexdigest()
            index = int(digest[:8], 16) % self._dimension
            vector[index] += 1.0
        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0:
            return vector
        return [value / norm for value in vector]


class OpenAIEmbeddingService(EmbeddingService):
    """OpenAI embeddings adapter."""

    def __init__(self, settings: Settings):
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "OpenAI embedding provider selected but package 'openai' is not installed. "
                "Install it with: pip install openai"
            ) from exc

        self._model = settings.embedding_model
        self._dimensions = settings.embedding_dimension if settings.embedding_dimension > 0 else None
        self._client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def embed_text(self, text: str) -> list[float]:
        kwargs: dict[str, object] = {"model": self._model, "input": text}
        if self._dimensions is not None:
            kwargs["dimensions"] = self._dimensions
        response = self._client.embeddings.create(**kwargs)
        return [float(value) for value in response.data[0].embedding]


class SentenceTransformersEmbeddingService(EmbeddingService):
    """Local sentence-transformers embeddings adapter."""

    def __init__(self, settings: Settings):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "Local embedding provider selected but package 'sentence-transformers' is not installed. "
                "Install it with: pip install sentence-transformers"
            ) from exc

        self._model = SentenceTransformer(settings.embedding_model)

    def embed_text(self, text: str) -> list[float]:
        vector = self._model.encode(text, normalize_embeddings=True)
        return [float(value) for value in vector]


def build_embedding_service(settings: Settings) -> EmbeddingService:
    """Factory for embedding service implementations."""

    provider = settings.embedding_provider.lower()

    if provider == "deterministic_stub":
        return DeterministicStubEmbeddingService(settings)
    if provider == "openai":
        return OpenAIEmbeddingService(settings)
    if provider in {"sentence_transformers", "local_sentence_transformers", "local_model"}:
        return SentenceTransformersEmbeddingService(settings)

    raise ValueError(f"Unsupported embedding provider: {settings.embedding_provider}")
