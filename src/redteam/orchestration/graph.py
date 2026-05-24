"""LangGraph 4-node red-team workflow: plan -> generate -> execute -> evaluate -> loop.

Day 6 evolution of the Day-5 skeleton:

- `plan_node` is no longer a deterministic round-robin. It is a closure over
  a `Planner` instance (default: ε-greedy with global success-rate memory,
  per spec §4.2). Backwards-compatible with the round-robin behaviour: pass
  `planner=None` and a `_RoundRobinPlanner` is used.
- `generate_node` dispatches on `payload_source`: `"template"` uses the
  hand-templated helpers from `redteam.attacks.*` (Day-3/Day-4 path);
  `"llm"` uses an `LLMExploitGenerator`. Trigger logic: iteration 0 always
  uses templates; iteration ≥ 1 uses the LLM, conditioned on
  `state["history"]` (the previous attempts' verdicts and outputs).
- `evaluate_node` now feeds back into the planner via `planner.update`.

The graph is built around a single shared `RAGPipeline` instance so we don't
re-load the embedding model or re-open Chroma on every iteration. The
planner and `LLMExploitGenerator` instances are likewise shared across
iterations within one query *and* across queries within one compiled graph.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Protocol

from langgraph.graph import END, StateGraph

from redteam.agents.exploit_generator import LLMExploitGenerator
from redteam.agents.planner import ATTACK_FAMILIES, Planner
from redteam.attacks.corpus_poisoning import generate_poison_payload
from redteam.attacks.prompt_injection import (
    DEFAULT_TARGET_STRING,
    generate_ipi_payload,
)
from redteam.attacks.query_injection import generate_query_injection_payload
from redteam.metrics.asr import compute_asr, compute_asr_deny
from redteam.metrics.rank_shift import compute_rank_shift
from redteam.metrics.ragas_wrapper import compute_ragas_scores
from redteam.metrics.verdict import compute_verdict
from redteam.orchestration.state import AttackChannel, AttackFamily, RedTeamState
from redteam.target.pipeline import RAGPipeline


# Default strategy per family. The planner picks the family; this map picks
# the strategy. A more sophisticated planner could pick (family, strategy)
# pairs — listed in FUTURE_WORKS.md §6 as a refinement.
_DEFAULT_STRATEGY: dict[AttackFamily, str] = {
    "prompt_injection": "instruction_override",
    "corpus_poisoning": "answer_replacement",
    "query_injection": "prefix_injection",
}

# Which delivery channel each family uses. The executor reads this to decide
# whether to call `Retriever.add_documents` (corpus channel) or to swap in a
# rewritten query string (query channel). Mapping kept here rather than on
# the family enum so the orchestration is the single source of truth for
# the channel split — adding a new family in future means adding one line
# here, one line in `_DEFAULT_STRATEGY`, and one branch in the generator.
_FAMILY_CHANNEL: dict[AttackFamily, AttackChannel] = {
    "prompt_injection": "corpus",
    "corpus_poisoning": "corpus",
    "query_injection": "query",
}

# Day-5 sentinel false answer for the corpus-poisoning *template* branch.
# Day 6's LLM branch picks query-specific plausible false answers, which
# was the missing ingredient for ASR-a to fire under poisoning (see Day 4
# lab note).
DEFAULT_POISON_TARGET = "PoisonedAnswer1985"


class PlannerLike(Protocol):
    """Minimal interface graph nodes need from a planner."""

    def select(self, query_text: str) -> AttackFamily: ...
    def update(self, query_text: str, family: AttackFamily, asr_t: bool) -> None: ...


@dataclass
class _RoundRobinPlanner:
    """Deterministic round-robin fallback (Day 5's behaviour).

    Kept as a no-LLM, no-RNG planner for tests that want to predict the
    exact sequence of (family) selections.
    """

    _counter: int = 0
    _last_query: str | None = field(default=None, repr=False)

    def select(self, query_text: str) -> AttackFamily:
        # Reset counter on a new query so within-query iterations alternate.
        if query_text != self._last_query:
            self._counter = 0
            self._last_query = query_text
        family = ATTACK_FAMILIES[self._counter % len(ATTACK_FAMILIES)]
        self._counter += 1
        return family

    def update(self, query_text: str, family: AttackFamily, asr_t: bool) -> None:
        # No-op: round-robin has no memory.
        del query_text, family, asr_t


@dataclass
class ForcedCellPlanner:
    """Always returns a fixed (family, strategy) pair; ``update()`` is a no-op.

    Used by ``scripts/06_run_experiments.py`` (the Day-9 experiment driver) to
    force the (seed × query × cell) Cartesian product the dissertation's
    headline matrix needs. The ε-greedy :class:`Planner` is preserved
    elsewhere as a *sidecar log* — its ``select()`` sequence is recorded
    per-seed for the RQ2 (planner-adaptivity) discussion, but it does not
    drive the headline runs.

    Carrying the strategy as well as the family is the reason this class
    exists rather than a thinner ``ForcedFamilyPlanner``: cell 3 of the
    Day-9 matrix is *corpus_poisoning + jamming*, which is structurally
    distinct from cell 2 (*corpus_poisoning + answer_replacement*). Pinning
    only the family would route both cells through ``_DEFAULT_STRATEGY`` and
    collapse them into one. The strategy attribute is duck-typed —
    :func:`make_plan_node` reads it via ``getattr(planner, "strategy",
    None)`` and falls through to ``_DEFAULT_STRATEGY`` for planners that
    don't carry one (the ε-greedy ``Planner`` and ``_RoundRobinPlanner``
    both behave exactly as before).

    The ``success_metric`` attribute drives the loop's early-exit
    predicate (:func:`should_continue`). Default ``"asr_target"`` matches
    the integrity-cell behaviour the graph has always had; the jamming
    cell sets ``"asr_deny"`` so its runs exit when the availability win
    fires rather than running iter 1+ which would have the LLM exploit
    generator silently switch to a different attack mode. Duck-typed for
    the same reason ``strategy`` is — ε-greedy and round-robin planners
    don't set it, and :func:`make_plan_node` falls through to the
    ``"asr_target"`` default when absent.
    """

    family: AttackFamily
    strategy: str
    success_metric: str = "asr_target"

    def select(self, query_text: str) -> AttackFamily:
        del query_text
        return self.family

    def update(self, query_text: str, family: AttackFamily, asr_t: bool) -> None:
        # No memory — the cell is fixed by construction.
        del query_text, family, asr_t


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------


def make_plan_node(planner: PlannerLike):
    """Closure over the planner. Returns the plan node function.

    Trigger logic for `payload_source`:
        - iteration 0  -> "template"  (cheap, deterministic, the proven path)
        - iteration >= 1 -> "llm"     (variant generation conditioned on history)
    """

    def plan_node(state: RedTeamState) -> dict[str, Any]:
        iteration = state.get("iteration", 0)
        family = planner.select(state["query"])
        # Day 9: a planner may optionally carry a `strategy` attribute to
        # pin both axes of the (family, strategy) cell — see
        # `ForcedCellPlanner`. Duck-typed to keep the `PlannerLike`
        # protocol unchanged: planners without a `strategy` attribute fall
        # through to the per-family default exactly as before.
        forced_strategy = getattr(planner, "strategy", None)
        strategy = forced_strategy if forced_strategy else _DEFAULT_STRATEGY[family]
        out: dict[str, Any] = {
            "attack_family": family,
            "attack_strategy": strategy,
            "payload_source": "template" if iteration == 0 else "llm",
        }
        # Day-10 fix (jamming early-exit): a planner may optionally carry a
        # `success_metric` attribute naming which state-boolean
        # `should_continue` should treat as the terminal condition. The
        # `poiJ` cell sets `"asr_deny"`; the three integrity cells leave
        # it at the default `"asr_target"`. Surfacing it via state (rather
        # than closing the planner over the predicate) keeps the
        # `PlannerLike` protocol thin and means the field appears in the
        # bundle JSON for audit. Duck-typed so the ε-greedy and
        # round-robin planners remain protocol-compatible without code
        # changes — `should_continue` defaults to `"asr_target"` when the
        # field is absent.
        forced_metric = getattr(planner, "success_metric", None)
        if forced_metric:
            out["success_metric"] = forced_metric
        return out

    return plan_node


def make_generate_node(exploit_gen: LLMExploitGenerator):
    """Closure over the LLM exploit generator. Dispatches on `payload_source`.

    Day 7.5 changes: dispatches *also* on the family's delivery channel.
    For corpus-channel attacks (IPI, poisoning) the return shape is
    unchanged from Day 6 — `payload` carries a document body, `payload_doc_id`
    is the Chroma id the executor will add/remove. For the query channel
    (query_injection) `payload` carries the rewritten query, `payload_doc_id`
    is a synthetic id used only for bundle audit, and the new
    `modified_query` field carries the same string so the executor can use
    it directly without parsing.
    """

    def generate_node(state: RedTeamState) -> dict[str, Any]:
        family: AttackFamily = state["attack_family"]
        strategy = state["attack_strategy"]
        query = state["query"]
        seed = state.get("seed", 42)
        source = state.get("payload_source", "template")
        iteration = state.get("iteration", 0)
        history = list(state.get("history", []))
        channel = _FAMILY_CHANNEL[family]

        if source == "template":
            payload, marker, payload_template_hash = _generate_via_template(
                query, family, strategy, seed
            )
        else:
            payload, marker, payload_template_hash = _generate_via_llm(
                exploit_gen, query, family, strategy, iteration, history
            )

        # The two channels have structurally different payload objects:
        # corpus payloads carry a `Document`; query payloads carry a
        # `modified_query` string. Branch the state-shape accordingly.
        if channel == "corpus":
            return {
                "payload": payload.document.page_content,
                "payload_doc_id": payload.doc_id,
                "attack_channel": "corpus",
                "modified_query": "",
                "payload_metadata": {
                    "doc_id": payload.doc_id,
                    "family": family,
                    "strategy": strategy,
                    "marker": marker,
                    "source": source,
                    "channel": "corpus",
                    "exploit_prompt_template_hash": payload_template_hash,
                    "document_metadata": dict(payload.document.metadata),
                },
            }
        # channel == "query"
        return {
            "payload": payload.modified_query,
            "payload_doc_id": payload.payload_id,
            "attack_channel": "query",
            "modified_query": payload.modified_query,
            "payload_metadata": {
                "doc_id": payload.payload_id,
                "family": family,
                "strategy": strategy,
                "marker": marker,
                "source": source,
                "channel": "query",
                "exploit_prompt_template_hash": payload_template_hash,
                # No `document_metadata` — query-channel payloads have no
                # corpus footprint. Executor checks `attack_channel` and
                # skips the add/remove path entirely.
                "document_metadata": None,
            },
        }

    return generate_node


def _generate_via_template(
    query: str,
    family: AttackFamily,
    strategy: str,
    seed: int,
) -> tuple[Any, str, Optional[str]]:
    """Hand-templated path (Day-3/Day-4/Day-7.5 helpers).

    Returns ``(payload, marker, prompt_template_hash)``. The hand-templated
    path has no LLM prompt template, so ``prompt_template_hash`` is
    ``None`` — the bundle reader infers the template from the attack
    family + strategy by reading the source.
    """
    if family == "prompt_injection":
        payload = generate_ipi_payload(
            query_text=query,
            target_string=DEFAULT_TARGET_STRING,
            strategy=strategy,  # type: ignore[arg-type]
            seed=seed,
        )
        return payload, payload.target_string, None
    if family == "corpus_poisoning":
        payload = generate_poison_payload(
            query_text=query,
            target_answer=DEFAULT_POISON_TARGET,
            strategy=strategy,  # type: ignore[arg-type]
            seed=seed,
        )
        return payload, payload.target_answer, None
    if family == "query_injection":
        payload = generate_query_injection_payload(
            query_text=query,
            target_string=DEFAULT_TARGET_STRING,
            strategy=strategy,  # type: ignore[arg-type]
            seed=seed,
        )
        return payload, payload.target_string, None
    raise ValueError(f"Unknown attack_family {family!r}")


def _generate_via_llm(
    exploit_gen: LLMExploitGenerator,
    query: str,
    family: AttackFamily,
    strategy: str,
    iteration: int,
    history: list[dict[str, Any]],
) -> tuple[Any, str, str]:
    """LLM-driven path. Returns (payload, marker, prompt_template_hash)."""
    if family == "prompt_injection":
        payload, trace = exploit_gen.generate_ipi(
            query_text=query,
            strategy=strategy,
            iteration=iteration,
            prior_failures=history,
        )
        return payload, payload.target_string, trace.prompt_template_hash
    if family == "corpus_poisoning":
        payload, trace = exploit_gen.generate_poison(
            query_text=query,
            iteration=iteration,
            prior_failures=history,
        )
        return payload, payload.target_answer, trace.prompt_template_hash
    if family == "query_injection":
        payload, trace = exploit_gen.generate_query_injection(
            query_text=query,
            strategy=strategy,
            iteration=iteration,
            prior_failures=history,
        )
        return payload, payload.target_string, trace.prompt_template_hash
    raise ValueError(f"Unknown attack_family {family!r}")


def make_execute_node(pipeline: RAGPipeline):
    """Closure over the shared RAG pipeline. Returns the executor node.

    Day 7: runs a *baseline* (clean) pass before the attacked pass so the
    evaluator can compute `rank_shift_at_k`. The baseline result is cached
    per-query inside this closure — iterating the same query multiple
    times runs the baseline only once. The cache is process-local; Day 9's
    experiment runs all queries in one process.
    """
    baseline_cache: dict[str, dict[str, Any]] = {}

    def execute_node(state: RedTeamState) -> dict[str, Any]:
        from langchain_core.documents import Document

        payload_doc_id = state["payload_doc_id"]
        query = state["query"]
        # Default to corpus channel when the field is absent — preserves
        # behaviour for any caller that initialised state pre-Day-7.5.
        channel = state.get("attack_channel", "corpus")

        # ---- baseline pass (cached, runs the ORIGINAL query in both channels)
        # The baseline is the user's clean query — we want rank_shift_at_k
        # to compare attacked retrieval against the clean retrieval, not
        # against a different attacked retrieval. For query-channel attacks
        # this still uses the unmodified `query`, which is the right
        # counterfactual ("what would the user have got without injection?").
        if query not in baseline_cache:
            baseline_cache[query] = pipeline.run(query)
        baseline = baseline_cache[query]

        if channel == "corpus":
            # Reconstruct the Document from state (TypedDict-friendly: avoids
            # carrying rich Python objects through state). Generator already
            # produced a deterministic doc_id, so this round-trip is exact.
            meta = state["payload_metadata"]
            doc = Document(
                page_content=state["payload"],
                metadata=meta["document_metadata"],
            )
            pipeline.retriever.add_documents([doc])
            try:
                run_record = pipeline.run(query)
            finally:
                pipeline.retriever.remove_documents([payload_doc_id])
            # Flag any retrieved chunk whose doc_id matches the payload's.
            retrieved = [
                {**d, "is_poisoned": d["doc_id"] == payload_doc_id}
                for d in run_record["retrieved_docs"]
            ]
        else:
            # Query channel: feed the modified query to the pipeline. Nothing
            # is added to or removed from the retriever, so the
            # `index_state_hash` returned to state matches the baseline's
            # automatically (same Chroma collection, same content).
            modified_query = state.get("modified_query") or state["payload"]
            run_record = pipeline.run(modified_query)
            # No poisoned doc to flag — the input channel doesn't add to
            # the corpus, so every retrieved chunk is legitimate. We still
            # surface the rank-1 doc_id etc. for the bundle audit.
            retrieved = [
                {**d, "is_poisoned": False} for d in run_record["retrieved_docs"]
            ]

        return {
            "retrieved_docs": retrieved,
            "generator_output": run_record["generator_output"],
            "generator_latency_ms": run_record["generator_latency_ms"],
            "index_state_hash": run_record["index_state_hash"],
            "baseline_retrieved_docs": baseline["retrieved_docs"],
            "baseline_generator_output": baseline["generator_output"],
        }

    return execute_node


def make_evaluate_node(planner: PlannerLike, run_ragas: bool = True):
    """Closure over the planner. Day 7: computes ASR + rank-shift + RAGAS.

    `run_ragas` exists so tests can disable the RAGAS path (which is the
    only LLM-call branch inside `evaluate_node`) and stay fast/offline.
    Production runs and the notebook leave it on.
    """

    def evaluate_node(state: RedTeamState) -> dict[str, Any]:
        payload_doc_id = state["payload_doc_id"]
        marker = state["payload_metadata"]["marker"]
        retrieved = state.get("retrieved_docs", [])
        output = state.get("generator_output", "") or ""
        channel = state.get("attack_channel", "corpus")

        # ---- ASR triple ---------------------------------------------------
        # ASR-r semantics depend on the channel:
        #   * corpus channel — was the payload doc in retrieved top-k?
        #   * query channel  — trivially True (the malicious instruction is
        #     embedded in the prompt itself; retrieval has no gating role).
        # Keeping the field name uniform across channels means Day 9's
        # experiment matrix can aggregate ASR-r across all three families
        # without per-family branching at analysis time.
        if channel == "query":
            from redteam.metrics.asr import ASRTriple, compute_asr_answer

            asr_a = compute_asr_answer(output, marker)
            asr = ASRTriple(retrieval=True, answer=asr_a, target=asr_a)
        else:
            asr = compute_asr(
                retrieved_docs=retrieved,
                payload_doc_id=payload_doc_id,
                generator_output=output,
                marker=marker,
            )

        # ASR-deny — availability metric (Day 7.5 lexicon, wired Day 8).
        # Fires when the LLM refused / declined to answer; orthogonal to
        # the integrity triple so a jamming attack registers a positive
        # `asr_deny` even when ASR-t is False. Computed for every run
        # (corpus and query channel alike) so the bundle's evaluation
        # block is uniform across the experiment matrix.
        asr_deny = compute_asr_deny(output)

        # Verdict is keyed off the *cell's headline success metric* — the
        # state-boolean named in ``success_metric`` (set by
        # ``make_plan_node`` for the Day-9 forced-cell matrix, defaulting
        # to ``"asr_target"`` for legacy planners). Jamming cells
        # (``"asr_deny"``) are binary on the refusal signal; integrity
        # cells keep the original three-way scheme. See
        # ``redteam.metrics.verdict.compute_verdict`` for the rationale —
        # importantly, this is the same helper the bundle-rewrite
        # migration calls, so on-disk verdicts cannot drift away from
        # newly-written ones.
        verdict = compute_verdict(
            success_metric=state.get("success_metric", "asr_target"),
            asr_retrieval=asr.retrieval,
            asr_target=asr.target,
            asr_deny=asr_deny,
        )

        # ---- rank_shift@k -------------------------------------------------
        baseline_retrieved = state.get("baseline_retrieved_docs", []) or []
        if baseline_retrieved:
            rs = compute_rank_shift(baseline_retrieved, retrieved, k=5)
            rank_shift_at_k = rs.rank_shift
        else:
            # Defensive: if executor didn't capture a baseline (shouldn't
            # happen post-Day-7 but kept so legacy tests pass), report 0.
            rank_shift_at_k = 0

        # ---- RAGAS triple --------------------------------------------------
        if run_ragas:
            ragas = compute_ragas_scores(
                query=state["query"],
                retrieved_contexts=[d.get("content", "") for d in retrieved],
                answer=output,
            )
        else:
            from redteam.metrics.ragas_wrapper import RagasScores
            ragas = RagasScores(
                faithfulness=None,
                answer_relevance=None,
                context_relevance=None,
                notes="ragas disabled in this graph",
            )

        # Day 6: feed the verdict back into the planner's success memory.
        planner.update(state["query"], state["attack_family"], asr.target)

        history = list(state.get("history", []))
        history.append({
            "iteration": state.get("iteration", 0),
            "attack_family": state.get("attack_family"),
            "attack_strategy": state.get("attack_strategy"),
            "payload_source": state.get("payload_source"),
            "payload_doc_id": payload_doc_id,
            "asr_retrieval": asr.retrieval,
            "asr_answer": asr.answer,
            "asr_target": asr.target,
            "asr_deny": asr_deny,
            "rank_shift_at_k": rank_shift_at_k,
            "verdict": verdict,
            # Truncated output snapshot — useful for the LLM exploit-gen
            # prompt on the next iteration without bloating state.
            "generator_output": output[:500],
        })

        # Increment iteration here so the conditional edge sees the post-iteration
        # counter, and so plan_node on the next loop sees the new index.
        return {
            "ragas_faithfulness": ragas.faithfulness,
            "ragas_answer_relevance": ragas.answer_relevance,
            "ragas_context_relevance": ragas.context_relevance,
            "ragas_notes": ragas.notes,
            "asr_retrieval": asr.retrieval,
            "asr_answer": asr.answer,
            "asr_target": asr.target,
            "asr_deny": asr_deny,
            "rank_shift_at_k": rank_shift_at_k,
            "verdict": verdict,
            "history": history,
            "iteration": state.get("iteration", 0) + 1,
        }

    return evaluate_node


def should_continue(state: RedTeamState) -> str:
    """Loop while iterations remain and we haven't already succeeded.

    Day-10 fix: terminate on the *cell's own* success metric rather than
    on ``verdict == "success"`` (which is keyed off ``asr_target`` and
    therefore mis-handles the jamming cell, whose success is
    ``asr_deny``). The ``success_metric`` field names the state-boolean
    to consult — set by :func:`make_plan_node` from
    :class:`ForcedCellPlanner.success_metric` for the Day-9 matrix; falls
    back to ``"asr_target"`` for any caller that doesn't set it (the
    ε-greedy and round-robin planners, and any existing test that
    initialises state without a planner). All three of
    ``asr_target``/``asr_answer``/``asr_deny`` are populated by
    :func:`evaluate_node` before this predicate runs, so the read is
    always defined.
    """
    iteration = state.get("iteration", 0)
    max_iter = state.get("max_iterations", 1)
    metric = state.get("success_metric", "asr_target")
    if state.get(metric):
        return "end"
    if iteration >= max_iter:
        return "end"
    return "loop"


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


def build_graph(
    pipeline: RAGPipeline,
    planner: Optional[PlannerLike] = None,
    exploit_gen: Optional[LLMExploitGenerator] = None,
    run_ragas: bool = True,
):
    """Compile the 4-node red-team graph.

    Defaults: ε-greedy `Planner` (ε=0.3) and an `LLMExploitGenerator` sharing
    the global SQLite cache. Tests inject the round-robin planner and a
    dummy exploit generator to keep them deterministic + offline.

    `run_ragas=False` short-circuits the RAGAS triple inside `evaluate_node`
    (the only LLM-call branch in evaluation). Tests pass `False` to stay
    fast/offline; production runs and the notebook leave it on.
    """
    if planner is None:
        planner = Planner(epsilon=0.3, seed=42)
    if exploit_gen is None:
        exploit_gen = LLMExploitGenerator()

    g: StateGraph = StateGraph(RedTeamState)
    g.add_node("plan", make_plan_node(planner))
    g.add_node("generate", make_generate_node(exploit_gen))
    g.add_node("execute", make_execute_node(pipeline))
    g.add_node("evaluate", make_evaluate_node(planner, run_ragas=run_ragas))

    g.set_entry_point("plan")
    g.add_edge("plan", "generate")
    g.add_edge("generate", "execute")
    g.add_edge("execute", "evaluate")
    g.add_conditional_edges(
        "evaluate",
        should_continue,
        {"loop": "plan", "end": END},
    )
    return g.compile()


# ---------------------------------------------------------------------------
# Day-5 backwards-compat: keep the old `plan_node` as a free function for
# tests that import it directly. Internally now defers to the round-robin
# planner instance.
# ---------------------------------------------------------------------------

_legacy_round_robin = _RoundRobinPlanner()


def plan_node(state: RedTeamState) -> dict[str, Any]:
    """round-robin planner — preserved for the existing planner test.

    New code should call `make_plan_node(planner)`; this free function is a
    thin wrapper using a module-level round-robin instance.
    """
    iteration = state.get("iteration", 0)
    # Use the legacy round-robin instance, but compute family from iteration
    # directly so the test's expectations (iter 0 -> PI, iter 1 -> poisoning)
    # remain stable regardless of prior calls.
    family: AttackFamily = ATTACK_FAMILIES[iteration % len(ATTACK_FAMILIES)]
    return {
        "attack_family": family,
        "attack_strategy": _DEFAULT_STRATEGY[family],
        "payload_source": "template" if iteration == 0 else "llm",
    }
