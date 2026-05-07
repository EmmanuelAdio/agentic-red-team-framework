"""Tests for the LangGraph red-team workflow (Day 5).

Two tests:

- `test_planner_round_robin` — pure-Python check on the planner's family
  rotation. No LLM, no Chroma, deterministic.
- `test_graph_runs_one_iteration_round_trip` — end-to-end: build the graph,
  invoke it for `max_iterations=1`, assert the run produced a verdict and
  left the index in its original state. Hits the live Chroma index and the
  cached LLM. Skips if the corpus or query set hasn't been built yet.

We do NOT assert that ASR-a fires — that is the demo notebook's job.
"""

from __future__ import annotations

import json

import pytest

from redteam.config import CHROMA_DIR, DATA_DIR, EMBEDDING_MODEL, load_env
from redteam.orchestration.graph import build_graph, plan_node
from redteam.orchestration.state import RedTeamState
from redteam.target.generator import LLMGenerator
from redteam.target.pipeline import RAGPipeline
from redteam.target.retriever import Retriever


def test_planner_round_robin() -> None:
    """`plan_node` alternates families on consecutive iterations."""
    s0: RedTeamState = {"iteration": 0}
    s1: RedTeamState = {"iteration": 1}
    s2: RedTeamState = {"iteration": 2}

    out0 = plan_node(s0)
    out1 = plan_node(s1)
    out2 = plan_node(s2)

    assert out0["attack_family"] == "prompt_injection"
    assert out1["attack_family"] == "corpus_poisoning"
    # Schedule cycles back round.
    assert out2["attack_family"] == out0["attack_family"]
    # Each family carries its default strategy through.
    assert out0["attack_strategy"] == "instruction_override"
    assert out1["attack_strategy"] == "answer_replacement"


def test_graph_runs_one_iteration_round_trip() -> None:
    """Single-iteration graph run leaves the index state unchanged.

    This is the Day-5 acceptance test: the graph wires the four nodes, the
    executor's `try/finally` cleanup runs, and `evaluate` writes a verdict.
    Whether ASR-a fires is out of scope (see module docstring).
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

    pipeline = RAGPipeline(retriever=retriever, generator=LLMGenerator())
    app = build_graph(pipeline)

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

    # Graph populated the executor + evaluator fields.
    assert "verdict" in final
    assert final["verdict"] in {"success", "partial", "failure"}
    assert final["retrieved_docs"], "executor produced no retrieved_docs"
    assert isinstance(final["generator_output"], str)
    # Iteration counter advanced past the loop bound.
    assert final["iteration"] == 1
    assert len(final["history"]) == 1

    # The executor's try/finally must have removed the payload chunk.
    post_ids = {d.doc_id for d in retriever.query(demo_query["query_text"], k=5)}
    assert final["payload_doc_id"] not in post_ids

    # Index state hash restored.
    assert retriever._count() == pre_count
    assert retriever.get_state_hash() == pre_hash, "index state_hash drifted after graph run"
