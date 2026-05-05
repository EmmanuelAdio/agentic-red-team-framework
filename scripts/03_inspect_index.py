"""Browse the Chroma index — see what's in the corpus and try ad-hoc queries.

Use this to pick test questions that the index can actually answer. Default mode
prints index size, the state hash, and 10 random doc titles. With `--query "..."`
it also runs a top-5 retrieval.

Run from repo root:
    python scripts/03_inspect_index.py
    python scripts/03_inspect_index.py --query "who was the first president"
    python scripts/03_inspect_index.py --titles 25
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from redteam.config import CHROMA_DIR, EMBEDDING_MODEL, load_env
from redteam.target.retriever import Retriever


def sample_titles(retriever: Retriever, n: int) -> list[tuple[str, str]]:
    """Return n random (doc_id, first-line) pairs from the index."""
    records = retriever.store._collection.get(include=["documents", "metadatas"])
    pairs: dict[str, str] = {}
    for content, meta in zip(records["documents"], records["metadatas"]):
        # Keep only the first chunk per doc (chunk_index 0) so we get titles, not body.
        if meta.get("chunk_index", 0) == 0:
            first_line = (content or "").splitlines()[0][:120]
            pairs[meta["doc_id"]] = first_line
    items = list(pairs.items())
    random.shuffle(items)
    return items[:n]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, default=None, help="Run a top-5 retrieval.")
    parser.add_argument("--titles", type=int, default=10, help="How many doc titles to sample.")
    parser.add_argument("--seed", type=int, default=0, help="Seed for title sampling.")
    args = parser.parse_args()

    load_env()
    random.seed(args.seed)

    retriever = Retriever(persist_dir=CHROMA_DIR, embedding_model_name=EMBEDDING_MODEL)
    chunk_count = retriever._count()
    print(f"Index location:    {CHROMA_DIR}")
    print(f"Chunk count:       {chunk_count}")
    print(f"State hash:        {retriever.get_state_hash()}")
    print()

    if chunk_count == 0:
        print("Index is empty. Run scripts/01_build_corpus.py first.")
        return

    print(f"--- {args.titles} sample doc titles (chunk 0 first line) ---")
    for doc_id, title in sample_titles(retriever, args.titles):
        print(f"  {doc_id}: {title}")
    print()

    if args.query:
        print(f"--- top-5 retrieval for: {args.query!r} ---")
        for d in retriever.query(args.query, k=5):
            preview = d.content.replace("\n", " ")[:140]
            print(f"  rank {d.rank}  score={d.score:.3f}  doc_id={d.doc_id}")
            print(f"    {preview}")


if __name__ == "__main__":
    main()
