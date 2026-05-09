from __future__ import annotations

import hashlib
import math
from abc import ABC, abstractmethod

from OLD.app.core.settings import Settings


class EmbeddingService(ABC):
    """Interface for embedding providers used by retrieval."""

    @abstractmethod
    def embed_text(self, text: str) -> list[float]:
        """Embed one text string into a dense vector."""

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple text strings; default implementation calls single-text embedding."""

        return [self.embed_text(text) for text in texts]


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

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_text(text) for text in texts]


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
        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required when EMBEDDING_PROVIDER=openai")
        self._client = OpenAI(api_key=settings.openai_api_key)

    def embed_text(self, text: str) -> list[float]:
        kwargs: dict[str, object] = {"model": self._model, "input": text}
        if self._dimensions is not None:
            kwargs["dimensions"] = self._dimensions
        response = self._client.embeddings.create(**kwargs)
        return [float(value) for value in response.data[0].embedding]

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        # Keep requests bounded to reduce per-request payload and avoid upstream limits.
        batch_size = 64
        vectors: list[list[float]] = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            kwargs: dict[str, object] = {"model": self._model, "input": batch}
            if self._dimensions is not None:
                kwargs["dimensions"] = self._dimensions
            response = self._client.embeddings.create(**kwargs)
            vectors.extend([[float(value) for value in row.embedding] for row in response.data])
        return vectors


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

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self._model.encode(texts, normalize_embeddings=True)
        return [[float(value) for value in vector] for vector in vectors]


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
