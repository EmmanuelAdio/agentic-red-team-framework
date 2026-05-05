"""Build the target corpus index.

Stratified 1k-doc slice of NQ (Natural Questions): gold docs for 50 test queries
guaranteed in slice + uniform random fill. Then chunked and indexed into a
persistent Chroma collection. Idempotent on chunk count.

Run from repo root:
    python scripts/01_build_corpus.py
    python scripts/01_build_corpus.py --rebuild   # force re-index after a slice change
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from redteam.config import CHROMA_DIR, EMBEDDING_MODEL, load_env
from redteam.target.corpus import chunk_documents, load_nq_slice
from redteam.target.retriever import Retriever


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Delete .chroma/ first (use after changing the slice composition).",
    )
    parser.add_argument("--n-docs", type=int, default=1000)
    parser.add_argument("--n-queries", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    load_env()

    if args.rebuild and CHROMA_DIR.exists():
        print(f"Removing stale Chroma at {CHROMA_DIR}…")
        shutil.rmtree(CHROMA_DIR)

    print(f"Loading stratified NQ slice ({args.n_docs} docs, {args.n_queries} queries)…")
    docs = load_nq_slice(n_docs=args.n_docs, n_queries=args.n_queries, seed=args.seed)
    n_gold = sum(1 for d in docs if d.metadata["is_gold"])
    print(f"  -> {len(docs)} docs ({n_gold} gold + {len(docs) - n_gold} random fill)")

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
