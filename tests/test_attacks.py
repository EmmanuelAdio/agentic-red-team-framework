"""Unit tests for the IPI (Indirect Prompt Injection) attack family.

Two tests:
- structural test (no LLM, no Chroma) — fast, deterministic.
- round-trip test against the live Chroma index — proves the topical-anchor
  heuristic places the payload in top-k and that cleanup leaves the index
  in its original state. We deliberately do NOT assert ASR-a here; whether
  gpt-4o-mini complies with the override is an end-to-end concern shown in
  the Jupyter demo, not a unit-test invariant.
"""

from __future__ import annotations

import json

import pytest

from redteam.attacks.prompt_injection import (
    DEFAULT_TARGET_STRING,
    IPIPayload,
    generate_ipi_payload,
)
from redteam.config import CHROMA_DIR, DATA_DIR, EMBEDDING_MODEL, load_env
from redteam.target.retriever import Retriever


@pytest.mark.parametrize("strategy", ["instruction_override", "role_reassignment"])
def test_generate_ipi_payload_structure(strategy: str) -> None:
    """Payload object is well-formed and deterministic per (query, target, strategy, seed)."""
    query = "Who was Thomas Jefferson?"
    target = DEFAULT_TARGET_STRING

    p1 = generate_ipi_payload(query, target_string=target, strategy=strategy, seed=42)
    p2 = generate_ipi_payload(query, target_string=target, strategy=strategy, seed=42)

    assert isinstance(p1, IPIPayload)
    assert p1.strategy == strategy
    assert p1.target_string == target
    assert p1.query_text == query
    assert p1.doc_id == p2.doc_id, "doc_id must be deterministic"
    assert p1.doc_id.startswith(f"ipi_{strategy}_")
    # Target string appears verbatim in the payload body — required for ASR-a substring matching.
    assert target in p1.document.page_content
    # Document metadata mirrors the payload doc_id (so add/remove via Retriever round-trips).
    assert p1.document.metadata["doc_id"] == p1.doc_id
    assert p1.document.metadata["is_poison"] is True
    assert p1.document.metadata["attack_strategy"] == strategy


def test_ipi_attack_round_trips() -> None:
    """Add → query → remove leaves the collection state_hash unchanged.

    Also asserts the topical anchor brings the payload into top-k retrieval —
    the *mechanism* of the IPI attack. Whether the LLM complies (ASR-a) is not
    asserted here; that is the demo notebook's job.
    """
    load_env()

    retriever = Retriever(persist_dir=CHROMA_DIR, embedding_model_name=EMBEDDING_MODEL)
    if retriever._count() == 0:
        pytest.skip("Chroma empty — run `python scripts/01_build_corpus.py` first.")

    queries_path = DATA_DIR / "queries.json"
    if not queries_path.exists():
        pytest.skip("queries.json missing — run `python scripts/04_build_query_set.py` first.")
    queries = json.loads(queries_path.read_text(encoding="utf-8"))
    assert queries
    demo_query = queries[0]

    pre_count = retriever._count()
    pre_hash = retriever.get_state_hash()

    payload = generate_ipi_payload(demo_query["query_text"], strategy="instruction_override")

    retriever.add_documents([payload.document])
    try:
        assert retriever._count() == pre_count + 1
        hits = retriever.query(demo_query["query_text"], k=5)
        retrieved_ids = {h.doc_id for h in hits}
        assert payload.doc_id in retrieved_ids, (
            f"topical anchor failed — payload {payload.doc_id} not in top-5 "
            f"(got {sorted(retrieved_ids)})"
        )
    finally:
        retriever.remove_documents([payload.doc_id])

    assert retriever._count() == pre_count
    assert retriever.get_state_hash() == pre_hash, "index state_hash drifted after cleanup"
