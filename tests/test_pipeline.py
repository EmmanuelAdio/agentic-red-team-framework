"""Unit test: pipeline.run() output shape matches the exploit-bundle execution block.

Every field below is referenced by PROJECT_SPEC.md §7 — if they go missing or
get renamed, the bundle JSON will fail validation later.
"""

from __future__ import annotations

import pytest

from redteam.config import CHROMA_DIR, EMBEDDING_MODEL, load_env
from redteam.target.generator import LLMGenerator
from redteam.target.pipeline import RAGPipeline
from redteam.target.retriever import Retriever


@pytest.fixture(scope="module")
def pipeline() -> RAGPipeline:
    load_env()
    r = Retriever(persist_dir=CHROMA_DIR, embedding_model_name=EMBEDDING_MODEL)
    if r._count() == 0:
        pytest.skip("Chroma empty — run `python scripts/01_build_corpus.py` first.")
    return RAGPipeline(retriever=r, generator=LLMGenerator())


REQUIRED_KEYS = {
    "query",
    "retrieved_docs",
    "generator_output",
    "generator_latency_ms",
    "prompt_template_hash",
    "index_state_hash",
}
REQUIRED_DOC_KEYS = {"doc_id", "rank", "score", "content", "is_poisoned"}


def test_pipeline_run_returns_bundle_shape(pipeline: RAGPipeline) -> None:
    result = pipeline.run("Who was Thomas Jefferson?")
    assert REQUIRED_KEYS <= set(result.keys())
    assert isinstance(result["retrieved_docs"], list) and result["retrieved_docs"]
    for d in result["retrieved_docs"]:
        assert REQUIRED_DOC_KEYS <= set(d.keys())
        assert d["is_poisoned"] is False  # baseline: nothing poisoned yet
    assert result["index_state_hash"].startswith("sha256:")
    assert result["prompt_template_hash"].startswith("sha256:")
