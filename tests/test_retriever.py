"""Unit tests for the Retriever (Chroma + bge-small wrapper).

Uses the persistent index built by `scripts/01_build_corpus.py`. Skips if not built.
"""

from __future__ import annotations

import pytest
from langchain_core.documents import Document

from redteam.config import CHROMA_DIR, EMBEDDING_MODEL, load_env
from redteam.target.retriever import Retriever


@pytest.fixture(scope="module")
def retriever() -> Retriever:
    load_env()
    r = Retriever(persist_dir=CHROMA_DIR, embedding_model_name=EMBEDDING_MODEL)
    if r._count() == 0:
        pytest.skip("Chroma empty — run `python scripts/01_build_corpus.py` first.")
    return r


def test_query_returns_ranked_topk(retriever: Retriever) -> None:
    """Top-k retrieval returns rank-ordered results with correct shape."""
    results = retriever.query("Thomas Jefferson founding father", k=5)
    assert len(results) <= 5
    assert results, "expected at least one result"
    # Ranks are 1-based and contiguous.
    assert [d.rank for d in results] == list(range(1, len(results) + 1))
    # Scores are floats.
    for d in results:
        assert isinstance(d.score, float)
        assert d.doc_id and d.content


def test_state_hash_is_stable(retriever: Retriever) -> None:
    """Same index -> same hash on repeated calls. Differs after add/remove."""
    h1 = retriever.get_state_hash()
    h2 = retriever.get_state_hash()
    assert h1 == h2
    assert h1.startswith("sha256:")


def test_add_then_remove_round_trips(retriever: Retriever) -> None:
    """add_documents + remove_documents must leave the collection unchanged."""
    pre_count = retriever._count()
    pre_hash = retriever.get_state_hash()

    fake_doc_id = "test_fake_doc_xyz"
    fake = Document(
        page_content="This is a synthetic test document inserted by the retriever round-trip test.",
        metadata={"doc_id": fake_doc_id, "source": "test", "chunk_index": 0},
    )
    retriever.add_documents([fake])
    try:
        assert retriever._count() == pre_count + 1
        # New doc should be retrievable by its content.
        hits = retriever.query("synthetic test document round-trip", k=5)
        assert any(h.doc_id == fake_doc_id for h in hits)
    finally:
        # Always remove, even if assertions above fail, so the index isn't left dirty.
        retriever.remove_documents([fake_doc_id])

    assert retriever._count() == pre_count
    assert retriever.get_state_hash() == pre_hash
