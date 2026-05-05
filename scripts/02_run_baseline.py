"""Baseline RAG smoke run over the 50-query test set.

Reads `data/queries.json`, runs each query through the clean pipeline, and
reports whether the gold doc appears in the top-k. This is the ASR-r baseline
(Attack Success Rate, retrieval) under no attack — a sanity number that should
be high; if it isn't, retrieval is the bottleneck.

Run from repo root:
    python scripts/02_run_baseline.py
    python scripts/02_run_baseline.py --limit 5       # just first 5 queries
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from redteam.config import CHROMA_DIR, DATA_DIR, EMBEDDING_MODEL, load_env
from redteam.target.generator import LLMGenerator
from redteam.target.pipeline import RAGPipeline
from redteam.target.retriever import Retriever

QUERIES_PATH = DATA_DIR / "queries.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Only run the first N queries.")
    args = parser.parse_args()

    load_env()

    if not QUERIES_PATH.exists():
        raise SystemExit(
            f"{QUERIES_PATH} not found. Run `python scripts/04_build_query_set.py` first."
        )
    queries = json.loads(QUERIES_PATH.read_text(encoding="utf-8"))
    if args.limit is not None:
        queries = queries[: args.limit]

    retriever = Retriever(persist_dir=CHROMA_DIR, embedding_model_name=EMBEDDING_MODEL)
    if retriever._count() == 0:
        raise SystemExit("Chroma is empty. Run `python scripts/01_build_corpus.py` first.")

    pipeline = RAGPipeline(retriever=retriever, generator=LLMGenerator())

    n_gold_in_topk = 0
    n_top1_is_gold = 0

    for q in queries:
        gold = set(q["gold_doc_ids"])
        result = pipeline.run(q["query_text"])
        retrieved_ids = [d["doc_id"] for d in result["retrieved_docs"]]

        in_topk = bool(set(retrieved_ids) & gold)
        top1_gold = retrieved_ids[:1] and retrieved_ids[0] in gold
        n_gold_in_topk += int(in_topk)
        n_top1_is_gold += int(bool(top1_gold))

        top = result["retrieved_docs"][0] if result["retrieved_docs"] else None
        print("-" * 72)
        print(f"[{q['query_id']}] {q['query_text']}")
        print(f"  gold={sorted(gold)}  in_top5={in_topk}  top1_is_gold={bool(top1_gold)}")
        if top:
            print(f"  top1: doc_id={top['doc_id']}  score={top['score']:.3f}")
        print(f"  latency_ms={result['generator_latency_ms']:.1f}")
        print(f"  A: {result['generator_output'][:300]}")

    n = len(queries)
    print("=" * 72)
    print(f"Baseline ASR-r (gold in top-5):   {n_gold_in_topk}/{n}  ({n_gold_in_topk / n:.1%})")
    print(f"Baseline top-1 == gold:           {n_top1_is_gold}/{n}  ({n_top1_is_gold / n:.1%})")


if __name__ == "__main__":
    main()
