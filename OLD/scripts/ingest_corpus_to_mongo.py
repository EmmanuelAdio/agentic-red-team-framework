from __future__ import annotations

import argparse
import hashlib
import math
import os
from collections import Counter
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.collection import Collection

try:
    from OLD.scripts.common import read_jsonl
except ModuleNotFoundError:  # pragma: no cover - support direct script execution
    from OLD.scripts.common import read_jsonl


class EmbeddingBackend:
    """Pluggable embedding backend for corpus ingestion."""

    def embed(self, text: str) -> list[float]:
        raise NotImplementedError

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        """Embed many texts; default implementation maps over single-text embedding."""

        return [self.embed(text) for text in texts]


class DeterministicStubEmbeddingBackend(EmbeddingBackend):
    """Deterministic baseline embedding for reproducible local experiments."""

    def __init__(self, dimension: int = 256):
        self._dimension = dimension

    def embed(self, text: str) -> list[float]:
        vector = [0.0] * self._dimension
        for token in str(text or "").lower().split():
            digest = hashlib.sha256(token.encode("utf-8")).hexdigest()
            index = int(digest[:8], 16) % self._dimension
            vector[index] += 1.0

        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0:
            return vector
        return [value / norm for value in vector]

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(text) for text in texts]


class OpenAIEmbeddingBackend(EmbeddingBackend):
    """OpenAI embedding backend loaded from environment variables."""

    def __init__(self, model: str, dimension: int | None = None):
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("Install openai to use --embedding-provider openai") from exc

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is required for OpenAI embeddings")

        self._model = model
        self._dimension = dimension if dimension and dimension > 0 else None
        self._client = OpenAI(api_key=api_key)

    def embed(self, text: str) -> list[float]:
        kwargs: dict[str, Any] = {"model": self._model, "input": text}
        if self._dimension is not None:
            kwargs["dimensions"] = self._dimension
        response = self._client.embeddings.create(**kwargs)
        return [float(value) for value in response.data[0].embedding]

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        kwargs: dict[str, Any] = {"model": self._model, "input": texts}
        if self._dimension is not None:
            kwargs["dimensions"] = self._dimension
        response = self._client.embeddings.create(**kwargs)
        return [[float(value) for value in row.embedding] for row in response.data]


def build_embedding_backend(provider: str, model: str, dimension: int) -> EmbeddingBackend:
    selected = provider.lower()
    if selected == "deterministic_stub":
        return DeterministicStubEmbeddingBackend(dimension=dimension)
    if selected == "openai":
        return OpenAIEmbeddingBackend(model=model, dimension=dimension)
    raise ValueError(f"Unsupported embedding provider: {provider}")


def ensure_indexes(collection: Collection) -> None:
    collection.create_index([("doc_id", 1)], unique=True)
    collection.create_index([("entity_id", 1)])
    collection.create_index([("entity_type", 1)])
    collection.create_index([("doc_type", 1)])
    collection.create_index([("attack_type", 1)])
    collection.create_index([("source_type", 1)])


def ingest_rows(
    collection: Collection,
    rows: list[dict[str, Any]],
    embedding_backend: EmbeddingBackend,
    clear_first: bool = False,
    batch_size: int = 100,
) -> dict[str, Any]:
    if clear_first:
        collection.delete_many({})

    ensure_indexes(collection)

    inserted_count = 0
    failed_count = 0

    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        payload: list[dict[str, Any]] = []
        texts = [str(row.get("content") or row.get("raw_text") or "") for row in batch]
        vectors: list[list[float]]
        try:
            vectors = embedding_backend.embed_many(texts)
            if len(vectors) != len(texts):
                raise RuntimeError("Embedding backend returned mismatched vector count")
        except Exception:
            vectors = []
            for text in texts:
                try:
                    vectors.append(embedding_backend.embed(text))
                except Exception:
                    failed_count += 1
                    vectors.append([])

        for row, vector in zip(batch, vectors):
            try:
                if not vector:
                    failed_count += 1
                    continue
                doc = dict(row)
                doc["embedding"] = vector
                payload.append(doc)
            except Exception:
                failed_count += 1

        if payload:
            try:
                result = collection.insert_many(payload, ordered=False)
                inserted_count += len(result.inserted_ids)
            except Exception:
                # Fallback to per-doc insert to keep progress for non-duplicate rows.
                for doc in payload:
                    try:
                        collection.insert_one(doc)
                        inserted_count += 1
                    except Exception:
                        failed_count += 1

    stats = {
        "total_input": len(rows),
        "inserted_count": inserted_count,
        "failed_count": failed_count,
        "entity_type": dict(Counter(str(row.get("entity_type")) for row in rows)),
        "doc_type": dict(Counter(str(row.get("doc_type")) for row in rows)),
        "attack_type": dict(Counter(str(row.get("attack_type")) for row in rows if row.get("attack_type") is not None)),
    }
    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest retrieval corpus JSONL into MongoDB with embeddings.")
    parser.add_argument("--input-path", type=Path, default=Path("data/corpus_retrieval.jsonl"))
    parser.add_argument("--mongo-uri", default=os.getenv("MONGODB_URI", "mongodb://localhost:27017"))
    parser.add_argument("--db-name", default=os.getenv("MONGODB_DB_NAME", "agentic_red_team_baseline"))
    parser.add_argument("--collection", default="rag_documents")
    parser.add_argument("--clear-first", action="store_true")
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--embedding-provider", default=os.getenv("EMBEDDING_PROVIDER", "deterministic_stub"))
    parser.add_argument("--embedding-model", default=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"))
    parser.add_argument("--embedding-dimension", type=int, default=int(os.getenv("EMBEDDING_DIMENSION", "256")))
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()

    rows = read_jsonl(args.input_path)
    backend = build_embedding_backend(
        provider=args.embedding_provider,
        model=args.embedding_model,
        dimension=args.embedding_dimension,
    )

    client = MongoClient(args.mongo_uri)
    collection = client[args.db_name][args.collection]

    stats = ingest_rows(
        collection=collection,
        rows=rows,
        embedding_backend=backend,
        clear_first=args.clear_first,
        batch_size=args.batch_size,
    )

    print(f"Ingested retrieval corpus from {args.input_path} into {args.db_name}.{args.collection}")
    print(f"Summary: {stats}")


if __name__ == "__main__":
    main()
