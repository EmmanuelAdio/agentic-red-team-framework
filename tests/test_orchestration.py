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
from redteam.attacks.query_injection import QueryInjectionPayload
from redteam.config import CHROMA_DIR, DATA_DIR, EMBEDDING_MODEL, load_env
from redteam.orchestration.graph import (
    ForcedCellPlanner,
    build_graph,
    make_plan_node,
    plan_node,
    should_continue,
)
from redteam.orchestration.state import RedTeamState
from redteam.target.generator import LLMGenerator
from redteam.target.pipeline import RAGPipeline
from redteam.target.retriever import Retriever


# ---------------------------------------------------------------------------
# Day 5 — kept tests
# ---------------------------------------------------------------------------


def test_planner_round_robin() -> None:
    """Legacy `plan_node` (round-robin) cycles families on consecutive iterations.

    Day 7.5 expanded the family set from 2 to 3 (added `query_injection`),
    so the cycle length is now 3: iter 3 returns to iter 0's family rather
    than iter 2 doing so.
    """
    s0: RedTeamState = {"iteration": 0}
    s1: RedTeamState = {"iteration": 1}
    s2: RedTeamState = {"iteration": 2}
    s3: RedTeamState = {"iteration": 3}

    out0 = plan_node(s0)
    out1 = plan_node(s1)
    out2 = plan_node(s2)
    out3 = plan_node(s3)

    assert out0["attack_family"] == "prompt_injection"
    assert out1["attack_family"] == "corpus_poisoning"
    assert out2["attack_family"] == "query_injection"
    # Cycle wraps: iter 3 % 3 == 0, so family matches iter 0.
    assert out3["attack_family"] == out0["attack_family"]
    assert out0["attack_strategy"] == "instruction_override"
    assert out1["attack_strategy"] == "answer_replacement"
    assert out2["attack_strategy"] == "prefix_injection"
    # Day 6 addition: the legacy node also writes payload_source.
    # iter 0 → template; iter ≥ 1 → llm.
    assert out0["payload_source"] == "template"
    assert out1["payload_source"] == "llm"
    assert out2["payload_source"] == "llm"


# ---------------------------------------------------------------------------
# Day 10 — `should_continue` honours the cell's success metric
# ---------------------------------------------------------------------------
#
# These four tests pin the early-exit predicate's behaviour across the
# Day-10 fix. The pre-fix predicate keyed off `verdict == "success"`
# which made the jamming cell (whose success is `asr_deny`) never
# early-exit, leading to the 69-of-150 silent-overwrite anomaly logged
# in the Day-10 lab note. The predicate now reads `state["success_metric"]`
# (default `"asr_target"` for backwards-compat) and treats the named
# boolean as the terminal signal.


def test_should_continue_defaults_to_asr_target() -> None:
    """Backwards-compat: when no ``success_metric`` is present in state,
    the predicate exits on ``asr_target=True`` exactly like the pre-fix
    behaviour. Existing ε-greedy / round-robin paths (which never set the
    metric) are unaffected.
    """
    state: RedTeamState = {
        "iteration": 0,
        "max_iterations": 3,
        "asr_target": True,
        "asr_deny": False,
        "verdict": "success",
    }
    assert should_continue(state) == "end"


def test_should_continue_loops_when_no_signal_yet() -> None:
    """No success fired and budget remains → keep looping."""
    state: RedTeamState = {
        "iteration": 1,
        "max_iterations": 3,
        "asr_target": False,
        "asr_deny": False,
    }
    assert should_continue(state) == "loop"


def test_should_continue_exits_on_asr_deny_for_jamming_cell() -> None:
    """Day-10 fix: the jamming cell carries ``success_metric="asr_deny"``;
    ``asr_deny=True`` terminates the loop on iteration 0 instead of
    continuing into iter 1 where the LLM exploit generator would silently
    switch attack mode and overwrite the availability win.
    """
    state: RedTeamState = {
        "iteration": 0,
        "max_iterations": 3,
        "success_metric": "asr_deny",
        "asr_target": False,  # jamming doesn't fire ASR-t — the bug was
                              # that the pre-fix predicate only looked here
        "asr_deny": True,
    }
    assert should_continue(state) == "end"


def test_should_continue_jamming_does_not_exit_on_asr_target() -> None:
    """A jamming-cell run with ``asr_target=True`` but ``asr_deny=False``
    is NOT a jamming win — it means the iteration-1+ LLM produced an
    answer-replacement-style payload that happened to land an ASR-t hit.
    The loop must keep iterating (or hit max_iter) rather than treating
    that as success for an availability cell.
    """
    state: RedTeamState = {
        "iteration": 1,
        "max_iterations": 3,
        "success_metric": "asr_deny",
        "asr_target": True,
        "asr_deny": False,
    }
    assert should_continue(state) == "loop"


def test_should_continue_exits_on_max_iter_regardless_of_metric() -> None:
    """Budget exhaustion is independent of the success-metric path."""
    state: RedTeamState = {
        "iteration": 3,
        "max_iterations": 3,
        "success_metric": "asr_deny",
        "asr_target": False,
        "asr_deny": False,
    }
    assert should_continue(state) == "end"


def test_forced_cell_planner_surfaces_success_metric_to_state() -> None:
    """The ``ForcedCellPlanner.success_metric`` value reaches
    ``plan_node``'s output dict, which the graph wires into state. This
    is the seam that lets ``should_continue`` see the metric on the
    correct iteration without coupling the predicate to the planner.
    """
    forced = ForcedCellPlanner(
        family="corpus_poisoning",
        strategy="jamming",
        success_metric="asr_deny",
    )
    plan = make_plan_node(forced)
    out = plan({"iteration": 0, "query": "anything"})
    assert out["success_metric"] == "asr_deny"
    assert out["attack_family"] == "corpus_poisoning"
    assert out["attack_strategy"] == "jamming"


def test_forced_cell_planner_default_metric_is_integrity() -> None:
    """``ForcedCellPlanner`` without an explicit metric defaults to
    ``asr_target`` (the integrity cells' metric). Confirms the dataclass
    default and that the field reaches state.
    """
    forced = ForcedCellPlanner(
        family="prompt_injection",
        strategy="instruction_override",
    )
    plan = make_plan_node(forced)
    out = plan({"iteration": 0, "query": "anything"})
    assert out["success_metric"] == "asr_target"


def test_make_plan_node_omits_metric_for_legacy_planners() -> None:
    """Round-robin and ε-greedy planners don't carry ``success_metric``.
    The plan-node output omits the field so the predicate falls back to
    the default ``"asr_target"`` at read time — keeping the
    ``PlannerLike`` protocol thin.
    """

    class _BarePlanner:
        """Has no ``success_metric`` attribute; mimics the ε-greedy /
        round-robin protocol surface."""

        def select(self, query_text: str) -> str:  # type: ignore[override]
            del query_text
            return "prompt_injection"

        def update(self, query_text: str, family: str, asr_t: bool) -> None:
            del query_text, family, asr_t

    plan = make_plan_node(_BarePlanner())  # type: ignore[arg-type]
    out = plan({"iteration": 0, "query": "anything"})
    assert "success_metric" not in out


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
    app = build_graph(pipeline, exploit_gen=_RaisingExploitGen(), run_ragas=False)

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

    def generate_query_injection(self, *args, **kwargs):
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
    query_injection_calls: list[dict] = None

    def __post_init__(self):
        self.ipi_calls = []
        self.poison_calls = []
        self.query_injection_calls = []

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

    def generate_query_injection(self, query_text, target_string=DEFAULT_TARGET_STRING,
                                 strategy="prefix_injection", iteration=1, prior_failures=None):
        self.query_injection_calls.append({
            "query": query_text, "iteration": iteration,
            "n_prior_failures": len(prior_failures or []),
        })
        modified = (
            f"FAKE LLM-rewritten query (iter={iteration}, strategy={strategy}). "
            f"Original was: {query_text}. Required reply: {target_string}"
        )
        payload_id = f"qinject_llm_fake_iter{iteration}"
        payload = QueryInjectionPayload(
            payload_id=payload_id,
            original_query=query_text,
            modified_query=modified,
            target_string=target_string,
            strategy=strategy,
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
    app = build_graph(pipeline, exploit_gen=_RaisingExploitGen(), run_ragas=False)

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

    app = build_graph(pipeline, planner=forced_planner, exploit_gen=fake_gen, run_ragas=False)

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
    # Sum across all three families (Day 7.5 added query_injection_calls).
    total_calls = (
        len(fake_gen.ipi_calls)
        + len(fake_gen.poison_calls)
        + len(fake_gen.query_injection_calls)
    )
    assert total_calls == 1, f"expected 1 LLM call, got {total_calls}"
    # And rollback still holds.
    assert retriever.get_state_hash() == pre_hash


# ---------------------------------------------------------------------------
# Day 7 — metrics fields wired through the graph
# ---------------------------------------------------------------------------


def test_graph_populates_metric_fields() -> None:
    """The Day-5 round-trip test now asserts the Day-7 metric fields are present.

    `asr_target` and `rank_shift_at_k` come from the metrics modules;
    `baseline_retrieved_docs` is captured by the executor's baseline pass;
    `ragas_*` are None here because we pass `run_ragas=False` to keep the
    test offline (covered by the dedicated RAGAS test instead).
    """
    load_env()

    retriever = Retriever(persist_dir=CHROMA_DIR, embedding_model_name=EMBEDDING_MODEL)
    if retriever._count() == 0:
        pytest.skip("Chroma empty.")
    queries_path = DATA_DIR / "queries.json"
    if not queries_path.exists():
        pytest.skip("queries.json missing.")
    demo_query = json.loads(queries_path.read_text(encoding="utf-8"))[0]

    pipeline = RAGPipeline(retriever=retriever, generator=LLMGenerator())
    app = build_graph(pipeline, exploit_gen=_RaisingExploitGen(), run_ragas=False)

    final = app.invoke({
        "run_id": "test_day7_metrics",
        "seed": 42,
        "query": demo_query["query_text"],
        "query_id": demo_query["query_id"],
        "iteration": 0,
        "max_iterations": 1,
        "history": [],
    })

    # ASR triple now includes asr_target as an explicit field.
    assert isinstance(final["asr_target"], bool)
    assert final["asr_target"] == (final["asr_retrieval"] and final["asr_answer"])
    # Day 8 wire-up: `asr_deny` is now populated for every run.
    assert isinstance(final["asr_deny"], bool)
    # rank_shift_at_k is now a real int (was a 0 placeholder pre-Day-7).
    assert isinstance(final["rank_shift_at_k"], int)
    # Baseline pass populated.
    assert final["baseline_retrieved_docs"], "executor did not capture baseline"
    assert isinstance(final["baseline_generator_output"], str)
    # RAGAS disabled → all None with a notes flag.
    assert final["ragas_faithfulness"] is None
    assert final["ragas_answer_relevance"] is None
    assert final["ragas_context_relevance"] is None
    assert final["ragas_notes"] and "disabled" in final["ragas_notes"]


# ---------------------------------------------------------------------------
# Day 7.5 — query-channel orchestration
# ---------------------------------------------------------------------------


def test_graph_query_channel_skips_corpus_writes() -> None:
    """Query-injection routes through the query channel without touching the index.

    Pins the Day-7.5 contract that the executor's query-channel branch:
      1. does NOT call `add_documents` / `remove_documents` (so Chroma's
         `state_hash` is identical pre- and post-execution);
      2. feeds the modified query into the pipeline, not the original;
      3. populates `attack_channel == "query"` and a non-empty
         `modified_query` in the final state;
      4. yields `asr_retrieval == True` trivially in the evaluator (the
         malicious instruction is delivered through the prompt, not via
         retrieval gating).
    """
    load_env()

    retriever = Retriever(persist_dir=CHROMA_DIR, embedding_model_name=EMBEDDING_MODEL)
    if retriever._count() == 0:
        pytest.skip("Chroma empty — run `python scripts/01_build_corpus.py` first.")
    queries_path = DATA_DIR / "queries.json"
    if not queries_path.exists():
        pytest.skip("queries.json missing.")
    demo_query = json.loads(queries_path.read_text(encoding="utf-8"))[0]

    pre_count = retriever._count()
    pre_hash = retriever.get_state_hash()

    # Force the planner to pick `query_injection` by giving it the only
    # success record. Greedy (ε=0) → planner always picks this family.
    forced_planner = Planner(epsilon=0.0, seed=42)
    forced_planner.update("seed", "query_injection", asr_t=True)

    pipeline = RAGPipeline(retriever=retriever, generator=LLMGenerator())
    app = build_graph(
        pipeline,
        planner=forced_planner,
        exploit_gen=_RaisingExploitGen(),  # iter 0 → template path; never call LLM
        run_ragas=False,
    )

    final = app.invoke({
        "run_id": "test_day7_5_query_channel",
        "seed": 42,
        "query": demo_query["query_text"],
        "query_id": demo_query["query_id"],
        "iteration": 0,
        "max_iterations": 1,
        "history": [],
    })

    # Channel-level invariants.
    assert final["attack_family"] == "query_injection"
    assert final["attack_channel"] == "query"
    assert final["modified_query"], "modified_query must be populated for query channel"
    assert demo_query["query_text"] in final["modified_query"], (
        "modified_query lost the original user question — retrieval would drift"
    )
    assert DEFAULT_TARGET_STRING in final["modified_query"], (
        "modified_query missing the target string"
    )

    # Index untouched — query-channel does not write to Chroma.
    assert retriever._count() == pre_count
    assert retriever.get_state_hash() == pre_hash, (
        "query-channel attack mutated the index — should be no-op"
    )
    # No retrieved chunk should be flagged poisoned.
    assert all(not d["is_poisoned"] for d in final["retrieved_docs"])

    # ASR-r is trivially True for the query channel.
    assert final["asr_retrieval"] is True
    # ASR-a may or may not fire — model behaviour is not the unit invariant.


# ---------------------------------------------------------------------------
# Day 9 — forced-cell planner (drives the 600-run experiment matrix)
# ---------------------------------------------------------------------------


def test_forced_cell_planner_drives_graph() -> None:
    """A ``ForcedCellPlanner(family, strategy)`` pins both axes of the cell.

    The Day-9 experiment matrix is a forced Cartesian over (seed × query ×
    cell) where one of the cells is *corpus_poisoning + jamming* — an
    availability attack scored by ASR-deny. Pinning only the family would
    route this cell through ``_DEFAULT_STRATEGY[corpus_poisoning] ==
    "answer_replacement"`` and silently collapse it into the integrity cell.
    This test pins the contract: when the planner carries a ``strategy``
    attribute, ``plan_node`` honours it; the resulting payload metadata
    records the forced strategy verbatim.
    """
    load_env()

    retriever = Retriever(persist_dir=CHROMA_DIR, embedding_model_name=EMBEDDING_MODEL)
    if retriever._count() == 0:
        pytest.skip("Chroma empty — run `python scripts/01_build_corpus.py` first.")
    queries_path = DATA_DIR / "queries.json"
    if not queries_path.exists():
        pytest.skip("queries.json missing.")
    demo_query = json.loads(queries_path.read_text(encoding="utf-8"))[0]

    pre_count = retriever._count()
    pre_hash = retriever.get_state_hash()

    pipeline = RAGPipeline(retriever=retriever, generator=LLMGenerator())
    forced = ForcedCellPlanner(family="corpus_poisoning", strategy="jamming")
    app = build_graph(
        pipeline,
        planner=forced,
        exploit_gen=_RaisingExploitGen(),  # iter 0 → template path
        run_ragas=False,
    )

    final = app.invoke({
        "run_id": "test_day9_forced_cell_jamming",
        "seed": 42,
        "query": demo_query["query_text"],
        "query_id": demo_query["query_id"],
        "iteration": 0,
        "max_iterations": 1,
        "history": [],
    })

    # The forced cell pins both axes.
    assert final["attack_family"] == "corpus_poisoning"
    assert final["attack_strategy"] == "jamming"
    # And the payload metadata round-trips the strategy so the bundle reader
    # can distinguish jamming bundles from answer_replacement bundles
    # without re-deriving from the body text.
    assert final["payload_metadata"]["strategy"] == "jamming"
    # ASR-deny is wired into evaluate_node as of Day 8 and is the success
    # metric for the jamming cell — it must be a bool here regardless of
    # whether the LLM actually refused.
    assert isinstance(final["asr_deny"], bool)

    # Index rollback still holds — corpus channel adds + removes the payload.
    assert retriever._count() == pre_count
    assert retriever.get_state_hash() == pre_hash, (
        "forced jamming cell leaked state — Day-9 batch would be contaminated"
    )
