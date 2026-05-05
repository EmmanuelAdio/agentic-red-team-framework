"""Build the target corpus index.

Loads a 1k-doc slice of NQ (Natural Questions), chunks it, and indexes the
chunks into a persistent Chroma collection. Idempotent — re-runs are no-ops
once the collection size matches.

Run from repo root:
    python scripts/01_build_corpus.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running before `pip install -e .` by adding src/ to sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from redteam.config import CHROMA_DIR, EMBEDDING_MODEL, load_env
from redteam.target.corpus import chunk_documents, load_nq_slice
from redteam.target.retriever import Retriever


def main() -> None:
    load_env()

    print("Loading NQ slice (1000 docs)…")
    docs = load_nq_slice(n_docs=1000, seed=42)
    print(f"  -> {len(docs)} docs")

    print("Chunking…")
    chunks = chunk_documents(docs, chunk_size=512, chunk_overlap=64)
    print(f"  -> {len(chunks)} chunks")

    print(f"Indexing into Chroma at {CHROMA_DIR}…")
    retriever = Retriever(persist_dir=CHROMA_DIR, embedding_model_name=EMBEDDING_MODEL)
    retriever.index(chunks)
    print(f"  -> collection size: {retriever._count()}")
    print(f"  -> index_state_hash: {retriever.get_state_hash()}")


if __name__ == "__main__":
    main()
