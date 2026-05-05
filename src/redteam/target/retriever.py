"""Dense retriever wrapping Chroma + a sentence-transformer embedding model.

Top-k cosine retrieval over chunked NQ docs. Supports runtime add/remove for
corpus-poisoning attacks, plus a SHA-256 state hash that goes into the exploit bundle.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings


@dataclass
class RetrievedDoc:
    """One result from a top-k retrieval."""
    doc_id: str
    content: str
    score: float  # similarity in [0,1]; higher = more relevant
    rank: int     # 1-based


def _chunk_id(doc_id: str, chunk_index: int) -> str:
    # Stable, unique per chunk so Chroma can address chunks individually.
    return f"{doc_id}::{chunk_index}"


class Retriever:
    """Persistent Chroma collection with bge-small embeddings."""

    def __init__(self, persist_dir: Path, embedding_model_name: str) -> None:
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
        # Chroma loads existing data from `persist_directory` if present.
        self.store = Chroma(
            collection_name="redteam_nq",
            embedding_function=self.embeddings,
            persist_directory=str(self.persist_dir),
        )

    def _count(self) -> int:
        return self.store._collection.count()

    def index(self, docs: list[Document]) -> None:
        """Index `docs` if not already present. Idempotent on chunk count."""
        if self._count() == len(docs):
            return
        ids = [
            _chunk_id(d.metadata["doc_id"], d.metadata.get("chunk_index", 0))
            for d in docs
        ]
        self.store.add_documents(docs, ids=ids)

    def query(self, text: str, k: int = 5) -> list[RetrievedDoc]:
        """Top-k cosine retrieval. Returns rank-ordered RetrievedDoc list."""
        results = self.store.similarity_search_with_relevance_scores(text, k=k)
        return [
            RetrievedDoc(
                doc_id=doc.metadata["doc_id"],
                content=doc.page_content,
                score=float(score),
                rank=rank,
            )
            for rank, (doc, score) in enumerate(results, start=1)
        ]

    def add_documents(self, docs: list[Document]) -> None:
        """Insert documents at runtime (used by corpus-poisoning attacks)."""
        ids = [
            _chunk_id(d.metadata["doc_id"], d.metadata.get("chunk_index", 0))
            for d in docs
        ]
        self.store.add_documents(docs, ids=ids)

    def remove_documents(self, doc_ids: list[str]) -> None:
        """Remove every chunk whose source `doc_id` is in `doc_ids`."""
        if not doc_ids:
            return
        # Chroma's where-filter takes $in for list membership.
        self.store._collection.delete(where={"doc_id": {"$in": list(doc_ids)}})

    def get_state_hash(self) -> str:
        """SHA-256 over the sorted unique doc_id list — pins index state per run."""
        records = self.store._collection.get(include=["metadatas"])
        unique_ids = sorted({m["doc_id"] for m in records["metadatas"]})
        joined = "\n".join(unique_ids).encode("utf-8")
        return "sha256:" + hashlib.sha256(joined).hexdigest()
