"""Unit tests for the two attack families: IPI and corpus poisoning.

For each family:
- a structural test (no LLM, no Chroma) — fast, deterministic.
- a round-trip test against the live Chroma index — proves the topical-anchor
  heuristic places the payload in top-k and that cleanup leaves the index
  in its original state.

We deliberately do NOT assert ASR-a in any test. Whether gpt-4o-mini complies
with an override (IPI) or repeats a planted false answer (poisoning) is an
end-to-end concern shown in the Jupyter demo — not a unit-test invariant.
"""

from __future__ import annotations

import json

import pytest

from redteam.attacks.corpus_poisoning import (
    PoisonPayload,
    generate_jamming_payload,
    generate_poison_payload,
    generate_poison_payloads,
)
from redteam.attacks.prompt_injection import (
    DEFAULT_TARGET_STRING,
    IPIPayload,
    generate_ipi_payload,
)
from redteam.attacks.query_injection import (
    QueryInjectionPayload,
    generate_query_injection_payload,
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


# ---------------------------------------------------------------------------
# Corpus-poisoning tests
# ---------------------------------------------------------------------------
#
# Day 4 covered single-doc answer-replacement. Day 7.5 adds three further
# tests for the pulled-in extensions:
#
#   * ``test_generate_poison_payload_backward_compat_v0`` — pinning the
#     doc_id and document body produced at ``variant_idx=0`` so the new
#     multi-doc parameter cannot silently break existing experiment
#     bundles.
#   * ``test_generate_poison_payloads_multi_doc`` — N near-duplicates with
#     unique doc_ids but a shared target_answer (PoisonedRAG ref [6] §4.2).
#   * ``test_generate_jamming_payload_structure`` — availability-attack
#     payload, no target_answer, refusal-pressure body.


def test_generate_poison_payload_structure() -> None:
    """Payload object well-formed and deterministic per (query, target, strategy, seed)."""
    query = "Who was Thomas Jefferson?"
    target = "Benjamin Franklin"  # plausible-but-wrong answer for the query

    p1 = generate_poison_payload(query, target_answer=target, seed=42)
    p2 = generate_poison_payload(query, target_answer=target, seed=42)

    assert isinstance(p1, PoisonPayload)
    assert p1.strategy == "answer_replacement"
    assert p1.target_answer == target
    assert p1.query_text == query
    assert p1.doc_id == p2.doc_id, "doc_id must be deterministic"
    assert p1.doc_id.startswith("poison_answer_replacement_")
    # Target answer appears verbatim in the payload body — required for ASR-a substring matching.
    assert target in p1.document.page_content
    # Document metadata mirrors the payload doc_id and flags the document as poison.
    assert p1.document.metadata["doc_id"] == p1.doc_id
    assert p1.document.metadata["is_poison"] is True
    assert p1.document.metadata["attack_strategy"] == "answer_replacement"
    assert p1.document.metadata["source"] == "poison_attack"


def test_corpus_poisoning_round_trips() -> None:
    """Add → query → remove leaves the collection state_hash unchanged.

    Asserts the topical anchor brings the poisoned doc into top-k. Whether the
    LLM repeats the planted false answer (ASR-a) is the demo notebook's job.
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

    payload = generate_poison_payload(
        demo_query["query_text"],
        target_answer="TestPlantedFalseAnswer",
    )

    retriever.add_documents([payload.document])
    try:
        assert retriever._count() == pre_count + 1
        hits = retriever.query(demo_query["query_text"], k=5)
        retrieved_ids = {h.doc_id for h in hits}
        assert payload.doc_id in retrieved_ids, (
            f"topical anchor failed — poison payload {payload.doc_id} not in top-5 "
            f"(got {sorted(retrieved_ids)})"
        )
    finally:
        retriever.remove_documents([payload.doc_id])

    assert retriever._count() == pre_count
    assert retriever.get_state_hash() == pre_hash, "index state_hash drifted after cleanup"


def test_generate_poison_payload_backward_compat_v0() -> None:
    """variant_idx=0 reproduces the Day-4 single-doc behaviour exactly.

    This pins the contract that adding the multi-doc machinery cannot have
    silently changed the doc_id, body, or anchor length for existing
    callers — important because Day-9's experiment matrix and any pre-
    Day 7.5 exploit bundle reference the v0 ids.
    """
    query = "Who was Thomas Jefferson?"
    target = "Benjamin Franklin"

    p_default = generate_poison_payload(query, target_answer=target, seed=42)
    p_v0 = generate_poison_payload(query, target_answer=target, seed=42, variant_idx=0)

    # variant_idx defaults to 0; explicit 0 must match exactly.
    assert p_default.doc_id == p_v0.doc_id
    assert p_default.document.page_content == p_v0.document.page_content
    assert p_default.variant_idx == 0
    # Variant index recorded in metadata for downstream auditing.
    assert p_v0.document.metadata["variant_idx"] == 0


def test_generate_poison_payloads_multi_doc() -> None:
    """N near-duplicate payloads — unique doc_ids, shared target_answer.

    Replicates the PoisonedRAG ref [6] §4.2 setup. Each variant should:
      1. carry the same target_answer (so they reinforce each other in the
         retrieval neighbourhood);
      2. have a distinct doc_id (so add_documents doesn't dedupe them);
      3. produce a distinct document body (so the bge-small embedder
         doesn't collapse them onto one retrieval position).
    """
    query = "Who was Thomas Jefferson?"
    target = "Benjamin Franklin"
    N = 5

    payloads = generate_poison_payloads(
        query, target_answer=target, n_docs=N, seed=42
    )

    assert isinstance(payloads, list)
    assert len(payloads) == N
    # Shared target_answer.
    assert all(p.target_answer == target for p in payloads)
    # Unique doc_ids.
    doc_ids = [p.doc_id for p in payloads]
    assert len(set(doc_ids)) == N, f"doc_id collision in N={N} batch: {doc_ids}"
    # Distinct bodies (variant templates differ).
    bodies = {p.document.page_content for p in payloads}
    assert len(bodies) == N, "variant templates produced identical bodies"
    # All variants flag is_poison and carry their variant_idx in metadata.
    for i, p in enumerate(payloads):
        assert p.variant_idx == i
        assert p.document.metadata["variant_idx"] == i
        assert p.document.metadata["is_poison"] is True
        assert p.document.metadata["attack_strategy"] == "answer_replacement"
        # Target answer appears verbatim in every variant — required for ASR-a.
        assert target in p.document.page_content

    # n_docs < 1 is a programmer error, not a runtime edge case.
    with pytest.raises(ValueError):
        generate_poison_payloads(query, target_answer=target, n_docs=0)
    # Empty target_answer is rejected — no false fact to plant.
    with pytest.raises(ValueError):
        generate_poison_payloads(query, target_answer="", n_docs=3)


def test_generate_jamming_payload_structure() -> None:
    """Jamming payload has no target_answer; body asserts unanswerability.

    Jamming is an availability attack: success is measured by
    ``compute_asr_deny`` (refusal-pattern match) on the generator output,
    not by ``compute_asr_answer`` (substring match). The structural test
    pins the no-target-answer contract and a few keywords from the
    refusal-pressure template.
    """
    query = "Who was Thomas Jefferson?"

    p = generate_jamming_payload(query, seed=42)

    assert isinstance(p, PoisonPayload)
    assert p.strategy == "jamming"
    assert p.target_answer == ""
    assert p.query_text == query
    assert p.doc_id.startswith("poison_jamming_")
    assert p.document.metadata["attack_strategy"] == "jamming"
    assert p.document.metadata["is_poison"] is True
    # Refusal-pressure keywords appear in the body — this is what
    # `compute_asr_deny` will substring-match against the LLM output.
    body_lc = p.document.page_content.lower()
    assert "cannot be answered" in body_lc or "decline to answer" in body_lc
    assert "refuse" in body_lc

    # Determinism — same query + seed → same doc_id.
    p2 = generate_jamming_payload(query, seed=42)
    assert p.doc_id == p2.doc_id


# ---------------------------------------------------------------------------
# Query-injection tests (Day 7.5)
# ---------------------------------------------------------------------------
#
# Query-side injection attacks the input channel rather than the corpus
# channel — there is no Document, no `add_documents`, and no Chroma round
# trip. The payload is purely a rewritten query string. Tests pin the
# structural contract; the orchestration tests in `test_orchestration.py`
# cover the end-to-end LangGraph behaviour for this family.


@pytest.mark.parametrize("strategy", ["prefix_injection", "suffix_injection"])
def test_generate_query_injection_payload_structure(strategy: str) -> None:
    """Query-injection payload is well-formed and deterministic."""
    query = "Who was Thomas Jefferson?"
    target = DEFAULT_TARGET_STRING

    p1 = generate_query_injection_payload(
        query, target_string=target, strategy=strategy, seed=42
    )
    p2 = generate_query_injection_payload(
        query, target_string=target, strategy=strategy, seed=42
    )

    assert isinstance(p1, QueryInjectionPayload)
    assert p1.strategy == strategy
    assert p1.target_string == target
    assert p1.original_query == query
    assert p1.query_text == query
    # Deterministic id + body.
    assert p1.payload_id == p2.payload_id
    assert p1.modified_query == p2.modified_query
    # Id namespace.
    assert p1.payload_id.startswith(f"qinject_{strategy}_")
    # The user's original query MUST still appear inside the rewrite — otherwise
    # the retrieval pass would lose all topical anchoring and ASR-r-equivalent
    # would become a no-op.
    assert query in p1.modified_query
    # The target string MUST be present (this is what the LLM is asked to emit).
    assert target in p1.modified_query
    # The rewrite is materially different from the original — sanity-check
    # that the template injected something non-trivial.
    assert len(p1.modified_query) > len(query) + len(target)


def test_generate_query_injection_payload_rejects_bad_inputs() -> None:
    """Empty query / target / unknown strategy raise ValueError."""
    with pytest.raises(ValueError):
        generate_query_injection_payload("", target_string="x")
    with pytest.raises(ValueError):
        generate_query_injection_payload("q", target_string="")
    with pytest.raises(ValueError):
        generate_query_injection_payload(
            "q", target_string="x", strategy="not_a_strategy"  # type: ignore[arg-type]
        )
