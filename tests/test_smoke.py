"""End-to-end smoke test for the target RAG pipeline.

Runs one real query through the clean pipeline against the persistent Chroma
index. Does NOT mock the LLM (per spec rule); relies on SQLiteCache for speed.

Prereq: corpus built and queries.json written:
    python scripts/01_build_corpus.py
    python scripts/04_build_query_set.py

Target: completes in <60 s on a cache hit.
"""

from __future__ import annotations

import json
import time

import pytest

from redteam.config import CHROMA_DIR, DATA_DIR, EMBEDDING_MODEL, load_env
from redteam.target.generator import LLMGenerator
from redteam.target.pipeline import RAGPipeline
from redteam.target.retriever import Retriever


@pytest.fixture(scope="module")
def pipeline() -> RAGPipeline:
    load_env()
    retriever = Retriever(persist_dir=CHROMA_DIR, embedding_model_name=EMBEDDING_MODEL)
    if retriever._count() == 0:
        pytest.skip("Chroma index empty — run `python scripts/01_build_corpus.py` first.")
    return RAGPipeline(retriever=retriever, generator=LLMGenerator())


def test_pipeline_runs_under_60s(pipeline: RAGPipeline) -> None:
    """Single end-to-end run must produce an answer + retrieved docs in <60 s."""
    queries_path = DATA_DIR / "queries.json"
    if not queries_path.exists():
        pytest.skip("queries.json missing — run `python scripts/04_build_query_set.py` first.")

    queries = json.loads(queries_path.read_text(encoding="utf-8"))
    assert queries, "queries.json is empty"
    q = queries[0]

    start = time.perf_counter()
    result = pipeline.run(q["query_text"])
    elapsed = time.perf_counter() - start

    assert elapsed < 60.0, f"pipeline took {elapsed:.1f}s (>60s budget)"
    assert result["generator_output"], "generator returned empty answer"
    assert len(result["retrieved_docs"]) > 0, "retriever returned zero docs"
    assert result["index_state_hash"].startswith("sha256:"), "missing index_state_hash"
    assert result["prompt_template_hash"].startswith("sha256:"), "missing prompt_template_hash"
