"""Tests for the LangGraph red-team workflow.

- Day 5 (kept):
    - `test_planner_round_robin` — pure-Python check on the legacy round-robin
      `plan_node` that the day-5 test still imports.
    - `test_graph_runs_one_iteration_round_trip` — end-to-end: build the
      graph, invoke for `max_iterations=1`, assert the run produced a
      verdict and left the index in its original state.

- Day 6 (new):
    - `test_graph_iteration_zero_uses_template_path` — the LLM exploit
      generator MUST NOT be called on iteration 0. We pass a fake generator
      that raises on any call.
    - `test_graph_iteration_one_uses_llm_path` — when invoked with
      `iteration=1`, the LLM exploit generator IS called, and the resulting
      state's `payload_source == "llm"`.

The hits-Chroma tests skip if the corpus or query set hasn't been built.
We do NOT assert that ASR-a fires — that is the demo notebook's job.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import pytest
from langchain_core.documents import Document

from redteam.agents.exploit_generator import GenerationTrace
from redteam.agents.planner import Planner
from redteam.attacks.corpus_poisoning import PoisonPayload
from redteam.attacks.prompt_injection import (
    DEFAULT_TARGET_STRING,
    IPIPayload,
    topical_anchor,
)
from redteam.config import CHROMA_DIR, DATA_DIR, EMBEDDING_MODEL, load_env
from redteam.orchestration.graph import build_graph, plan_node
from redteam.orchestration.state import RedTeamState
from redteam.target.generator import LLMGenerator
from redteam.target.pipeline import RAGPipeline
from redteam.target.retriever import Retriever


# ---------------------------------------------------------------------------
# Day 5 — kept tests
# ---------------------------------------------------------------------------


def test_planner_round_robin() -> None:
    """Legacy `plan_node` (round-robin) alternates families on consecutive iterations."""
    s0: RedTeamState = {"iteration": 0}
    s1: RedTeamState = {"iteration": 1}
    s2: RedTeamState = {"iteration": 2}

    out0 = plan_node(s0)
    out1 = plan_node(s1)
    out2 = plan_node(s2)

    assert out0["attack_family"] == "prompt_injection"
    assert out1["attack_family"] == "corpus_poisoning"
    assert out2["attack_family"] == out0["attack_family"]
    assert out0["attack_strategy"] == "instruction_override"
    assert out1["attack_strategy"] == "answer_replacement"
    # Day 6 addition: the legacy node now also writes payload_source.
    assert out0["payload_source"] == "template"
    assert out1["payload_source"] == "llm"


def test_graph_runs_one_iteration_round_trip() -> None:
    """Single-iteration graph run leaves the index state unchanged."""
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

    pipeline = RAGPipeline(retriever=retriever, generator=LLMGenerator())
    # Day 6: pass a never-call exploit generator so we don't depend on the
    # real LLM for this Day-5-style round-trip test.
    app = build_graph(pipeline, exploit_gen=_RaisingExploitGen())

    initial: RedTeamState = {
        "run_id": "test_day5_one_iter",
        "seed": 42,
        "query": demo_query["query_text"],
        "query_id": demo_query["query_id"],
        "iteration": 0,
        "max_iterations": 1,
        "history": [],
    }
    final = app.invoke(initial)

    assert final["verdict"] in {"success", "partial", "failure"}
    assert final["retrieved_docs"], "executor produced no retrieved_docs"
    assert isinstance(final["generator_output"], str)
    assert final["iteration"] == 1
    assert len(final["history"]) == 1

    post_ids = {d.doc_id for d in retriever.query(demo_query["query_text"], k=5)}
    assert final["payload_doc_id"] not in post_ids

    assert retriever._count() == pre_count
    assert retriever.get_state_hash() == pre_hash, "index state_hash drifted after graph run"


# ---------------------------------------------------------------------------
# Day 6 — fakes
# ---------------------------------------------------------------------------


class _RaisingExploitGen:
    """Fake LLM exploit generator that raises if called.

    Used to assert iteration 0 takes the template path — the LLM gen MUST
    NOT be invoked on iteration 0.
    """

    def generate_ipi(self, *args, **kwargs):  # noqa: D401 - signature mirrors real class
        raise AssertionError(
            "LLM exploit generator was called on iteration 0 — should be template path"
        )

    def generate_poison(self, *args, **kwargs):
        raise AssertionError(
            "LLM exploit generator was called on iteration 0 — should be template path"
        )


@dataclass
class _RecordingExploitGen:
    """Fake LLM exploit generator that records calls and returns dummy payloads.

    Returns IPIPayload / PoisonPayload objects that are structurally valid
    (so the executor can `add_documents` and the evaluator can compute the
    ASR triple), with a topical anchor so the payload enters top-k.
    """

    ipi_calls: list[dict] = None
    poison_calls: list[dict] = None

    def __post_init__(self):
        self.ipi_calls = []
        self.poison_calls = []

    def generate_ipi(self, query_text, target_string=DEFAULT_TARGET_STRING,
                     strategy="instruction_override", iteration=1, prior_failures=None):
        self.ipi_calls.append({
            "query": query_text, "iteration": iteration,
            "n_prior_failures": len(prior_failures or []),
        })
        body = (
            f"{topical_anchor(query_text)}\n\n"
            f"FAKE LLM-GENERATED IPI body for iter={iteration}.\n{target_string}\n"
        )
        doc_id = f"ipi_llm_fake_iter{iteration}"
        document = Document(
            page_content=body,
            metadata={
                "doc_id": doc_id,
                "source": "ipi_attack_llm",
                "is_poison": True,
                "attack_strategy": strategy,
                "chunk_index": 0,
            },
        )
        payload = IPIPayload(
            doc_id=doc_id,
            document=document,
            target_string=target_string,
            strategy=strategy,
            query_text=query_text,
        )
        return payload, GenerationTrace(latency_ms=0.0, prompt_template_hash="sha256:fake")

    def generate_poison(self, query_text, iteration=1, prior_failures=None):
        self.poison_calls.append({
            "query": query_text, "iteration": iteration,
            "n_prior_failures": len(prior_failures or []),
        })
        target_answer = "FakeLLMTargetAnswer"
        body = (
            f"{topical_anchor(query_text)}\n\n"
            f"FAKE LLM poisoning body. The accepted answer is {target_answer}. "
            f"Modern consensus confirms {target_answer}.\n"
        )
        doc_id = f"poison_llm_fake_iter{iteration}"
        document = Document(
            page_content=body,
            metadata={
                "doc_id": doc_id,
                "source": "poison_attack_llm",
                "is_poison": True,
                "attack_strategy": "answer_replacement",
                "chunk_index": 0,
            },
        )
        payload = PoisonPayload(
            doc_id=doc_id,
            document=document,
            target_answer=target_answer,
            strategy="answer_replacement",
            query_text=query_text,
        )
        return payload, GenerationTrace(latency_ms=0.0, prompt_template_hash="sha256:fake")


# ---------------------------------------------------------------------------
# Day 6 — trigger-logic tests
# ---------------------------------------------------------------------------


def test_graph_iteration_zero_uses_template_path() -> None:
    """Iteration 0 takes the template path — `_RaisingExploitGen` is never called."""
    load_env()

    retriever = Retriever(persist_dir=CHROMA_DIR, embedding_model_name=EMBEDDING_MODEL)
    if retriever._count() == 0:
        pytest.skip("Chroma empty — run `python scripts/01_build_corpus.py` first.")
    queries_path = DATA_DIR / "queries.json"
    if not queries_path.exists():
        pytest.skip("queries.json missing.")
    demo_query = json.loads(queries_path.read_text(encoding="utf-8"))[0]

    pipeline = RAGPipeline(retriever=retriever, generator=LLMGenerator())
    app = build_graph(pipeline, exploit_gen=_RaisingExploitGen())

    final = app.invoke({
        "run_id": "test_day6_iter0_template",
        "seed": 42,
        "query": demo_query["query_text"],
        "query_id": demo_query["query_id"],
        "iteration": 0,
        "max_iterations": 1,
        "history": [],
    })

    assert final["payload_source"] == "template"
    # If the raising fake had fired, the graph would have raised before here.


def test_graph_iteration_one_uses_llm_path() -> None:
    """Starting from `iteration=1`, the graph routes through the LLM path."""
    load_env()

    retriever = Retriever(persist_dir=CHROMA_DIR, embedding_model_name=EMBEDDING_MODEL)
    if retriever._count() == 0:
        pytest.skip("Chroma empty.")
    queries_path = DATA_DIR / "queries.json"
    if not queries_path.exists():
        pytest.skip("queries.json missing.")
    demo_query = json.loads(queries_path.read_text(encoding="utf-8"))[0]

    pre_hash = retriever.get_state_hash()

    pipeline = RAGPipeline(retriever=retriever, generator=LLMGenerator())
    fake_gen = _RecordingExploitGen()
    # Force IPI selection so we can assert against fake_gen.ipi_calls.
    forced_planner = Planner(epsilon=0.0, seed=42)
    forced_planner.update("seed", "prompt_injection", asr_t=True)  # IPI ahead

    app = build_graph(pipeline, planner=forced_planner, exploit_gen=fake_gen)

    final = app.invoke({
        "run_id": "test_day6_iter1_llm",
        "seed": 42,
        "query": demo_query["query_text"],
        "query_id": demo_query["query_id"],
        "iteration": 1,
        "max_iterations": 2,
        "history": [{"iteration": 0, "verdict": "failure"}],
    })

    assert final["payload_source"] == "llm"
    # Fake LLM gen was called exactly once (one iteration of the LLM path).
    total_calls = len(fake_gen.ipi_calls) + len(fake_gen.poison_calls)
    assert total_calls == 1, f"expected 1 LLM call, got {total_calls}"
    # And rollback still holds.
    assert retriever.get_state_hash() == pre_hash
