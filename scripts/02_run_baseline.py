"""Baseline RAG smoke test.

Runs five hardcoded queries through the clean (un-attacked) pipeline and prints
the top-1 retrieved doc plus the LLM's answer. Day 1 success = this script
exits 0 and shows five answers.

Run from repo root:
    python scripts/02_run_baseline.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from redteam.config import CHROMA_DIR, EMBEDDING_MODEL, load_env
from redteam.target.generator import LLMGenerator
from redteam.target.pipeline import RAGPipeline
from redteam.target.retriever import Retriever

TEST_QUERIES: list[str] = [
    "Who wrote Pride and Prejudice?",
    "When did the Second World War end?",
    "What is the capital of Australia?",
    "Who painted the Mona Lisa?",
    "What is the chemical symbol for gold?",
    "Who is Thomas Jefferson?"
]


def main() -> None:
    load_env()

    retriever = Retriever(persist_dir=CHROMA_DIR, embedding_model_name=EMBEDDING_MODEL)
    if retriever._count() == 0:
        raise SystemExit(
            "Chroma collection is empty. Run `python scripts/01_build_corpus.py` first."
        )

    pipeline = RAGPipeline(retriever=retriever, generator=LLMGenerator())

    for q in TEST_QUERIES:
        result = pipeline.run(q)
        top = result["retrieved_docs"][0] if result["retrieved_docs"] else None
        print("-" * 72)
        print(f"Q: {q}")
        if top:
            print(f"  top-1 doc_id={top['doc_id']}  score={top['score']:.3f}")
        print(f"  latency_ms={result['generator_latency_ms']:.1f}")
        print(f"  A: {result['generator_output']}")


if __name__ == "__main__":
    main()
