"""Unit tests for the corpus loader.

These call HF (Hugging Face) `load_dataset` but rely on the local cache after
Day 2's first run, so they're fast on subsequent invocations.
"""

from __future__ import annotations

from redteam.target.corpus import chunk_documents, load_nq_slice, select_test_queries


def test_select_test_queries_is_deterministic() -> None:
    """Same seed -> same query list (required for reproducibility)."""
    a = select_test_queries(n_queries=10, seed=42)
    b = select_test_queries(n_queries=10, seed=42)
    assert a == b
    assert len(a) == 10
    # Each entry: (query_id, query_text, gold_doc_ids)
    for qid, text, golds in a:
        assert isinstance(qid, str) and qid
        assert isinstance(text, str) and text
        assert isinstance(golds, list) and golds


def test_load_nq_slice_includes_every_gold_doc() -> None:
    """The stratified slice must contain *all* gold docs for the selected queries."""
    n_queries = 10
    selected = select_test_queries(n_queries=n_queries, seed=42)
    expected_gold = {gid for _, _, golds in selected for gid in golds}

    docs = load_nq_slice(n_docs=200, n_queries=n_queries, seed=42)
    sliced_ids = {d.metadata["doc_id"] for d in docs}
    missing = expected_gold - sliced_ids
    assert not missing, f"missing gold docs from slice: {missing}"


def test_chunk_documents_preserves_doc_id() -> None:
    """Every chunk inherits its parent's doc_id (so chunks group back to source)."""
    docs = load_nq_slice(n_docs=20, n_queries=2, seed=42)
    chunks = chunk_documents(docs)
    assert len(chunks) >= len(docs)
    parent_ids = {d.metadata["doc_id"] for d in docs}
    for c in chunks:
        assert c.metadata["doc_id"] in parent_ids
        assert "chunk_index" in c.metadata
