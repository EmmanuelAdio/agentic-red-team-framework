"""Build an :class:`ExploitBundle` from a finished :class:`RedTeamState`.

The bundle layer is kept *downstream* of the orchestration graph: nodes
write to ``RedTeamState`` (the live working set the LangGraph passes
between nodes); the bundle is a *materialisation* of that state plus
environment metadata for archival. Keeping the projection one-way (state
→ bundle, never bundle → state) means the graph can evolve its state
shape with only the builder needing to track it.

Why a separate module?

* :mod:`redteam.bundles.schema` is pure shape — no imports of the rest
  of the project. That keeps schema definitions easy to review and means
  bundle JSON files can be loaded without instantiating a RAG pipeline.
* :mod:`redteam.bundles.builder` is the place where business logic lives:
  reading the right field from state, picking ``injection_stage`` from
  ``attack_channel``, capturing git + Python version metadata, etc.
* This separation matches the existing project pattern (e.g. the metrics
  module: pure dataclasses on one side, orchestration glue on the other).
"""

from __future__ import annotations

import platform
import subprocess
from typing import Any

from redteam.bundles.schema import (
    BUNDLE_VERSION,
    FRAMEWORK_VERSION,
    AttackBlock,
    BundleSummary,
    EvaluationBlock,
    ExecutionBlock,
    ExploitBundle,
    Reproducibility,
    RetrievedDocRecord,
    TargetSystem,
    utc_now_iso,
)
from redteam.config import (
    EMBEDDING_MODEL,
    LLM_MODEL,
    LLM_TEMPERATURE,
    PROJECT_ROOT,
    RETRIEVER_TOP_K,
)
from redteam.orchestration.state import RedTeamState
from redteam.target.generator import PROMPT_TEMPLATE_HASH

# Subset of installed packages whose versions matter for replay. The full
# `pip freeze` is overkill — bundles are meant to be readable. A reader
# wanting full env reproducibility can consult `requirements.txt` pinned
# at the same git_commit recorded below.
_KEY_DEPENDENCIES: tuple[str, ...] = (
    "langchain",
    "langgraph",
    "langchain-core",
    "langchain-openai",
    "langchain-chroma",
    "chromadb",
    "ragas",
    "pydantic",
    "openai",
    "sentence-transformers",
)


# ---------------------------------------------------------------------------
# Environment capture
# ---------------------------------------------------------------------------


def _git_short_sha() -> str | None:
    """Return the current short SHA, or ``None`` if not in a git repo / no git.

    Resilient to: not being in a repo, ``git`` not on PATH, and detached-HEAD
    states (still produces a SHA). We do *not* embed dirty-state info
    because the spec example shows a clean SHA — a separate `dirty: bool`
    field could be added in a future bundle_version bump.
    """
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(PROJECT_ROOT),
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5.0,
        )
        return out.strip() or None
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return None


def _key_dependency_versions() -> dict[str, str]:
    """Best-effort version capture for the listed packages.

    Uses :func:`importlib.metadata.version`; missing packages are simply
    omitted rather than raising, so the bundle layer remains robust to
    optional-dep installs (e.g. the bundle module is import-safe even if
    `ragas` is uninstalled).
    """
    from importlib.metadata import PackageNotFoundError, version

    out: dict[str, str] = {}
    for name in _KEY_DEPENDENCIES:
        try:
            out[name] = version(name)
        except PackageNotFoundError:
            continue
    return out


def _python_version() -> str:
    """`major.minor.patch` triple — matches spec §7's example shape."""
    return platform.python_version()


# ---------------------------------------------------------------------------
# State → bundle projection
# ---------------------------------------------------------------------------


def _injection_stage(attack_channel: str) -> str:
    """Map ``attack_channel`` → spec §7's ``injection_stage`` vocabulary.

    Day 7.5 added the channel split (`corpus` vs `query`); spec §7's example
    used `injection_stage = "indexing"` for the corpus channel. We keep the
    spec field name for downstream compatibility and add a `query` value
    for the new family.
    """
    if attack_channel == "query":
        return "query"
    # Default for `corpus` and any legacy state without `attack_channel`.
    return "indexing"


def _retrieved_doc_records(
    retrieved: list[dict[str, Any]],
) -> list[RetrievedDocRecord]:
    """Strip the heavy `content` field; keep only the audit-relevant columns.

    The full chunk text is recoverable from the indexed corpus + the
    chunk_index encoded in `doc_id`, so omitting it keeps bundles
    compact (typical bundle ~3-5 KiB instead of ~30+ KiB once 5 retrieved
    chunks of NQ text are inlined).
    """
    return [
        RetrievedDocRecord(
            doc_id=str(d.get("doc_id", "")),
            rank=int(d.get("rank", 0)),
            score=float(d.get("score", 0.0)),
            is_poisoned=bool(d.get("is_poisoned", False)),
        )
        for d in retrieved
    ]


def _baseline_top1_doc_id(state: RedTeamState) -> str | None:
    """Pull the rank-1 doc id from the baseline (clean) retrieval pass."""
    baseline = state.get("baseline_retrieved_docs") or []
    if not baseline:
        return None
    # Prefer rank==1 if available; fall back to the first list element.
    for d in baseline:
        if d.get("rank") == 1:
            return str(d.get("doc_id", ""))
    return str(baseline[0].get("doc_id", ""))


def build_bundle(state: RedTeamState) -> ExploitBundle:
    """Project a finished :class:`RedTeamState` into an :class:`ExploitBundle`.

    Caller is responsible for invoking this *after* the graph has run to
    completion (i.e. on the dict returned by ``app.invoke(initial_state)``):
    incomplete states are detected via the strict pydantic validation —
    missing required fields raise ``pydantic.ValidationError`` with a
    pointer to the offending key.

    Notes
    -----
    The bundle's ``timestamp_utc`` is captured at *build time*, not at
    execution time. For the Day-8 single-process dry run this is within
    a few hundred milliseconds of the actual run, which is the resolution
    the bundle JSON example in spec §7 shows. If higher-fidelity timing
    is needed later, a `started_at` / `finished_at` pair can be added in
    a future ``bundle_version`` bump.
    """
    payload_metadata = state.get("payload_metadata") or {}
    attack_channel = state.get("attack_channel", "corpus")

    target_system = TargetSystem(
        embedding_model=EMBEDDING_MODEL,
        vector_store="chroma",
        retriever_top_k=RETRIEVER_TOP_K,
        llm_model=LLM_MODEL,
        llm_temperature=LLM_TEMPERATURE,
        prompt_template_hash=PROMPT_TEMPLATE_HASH,
    )

    attack = AttackBlock(
        family=state["attack_family"],
        strategy=state["attack_strategy"],
        payload=state["payload"],
        payload_id=state["payload_doc_id"],
        injection_stage=_injection_stage(attack_channel),
        iteration=int(state.get("iteration", 0)),
        payload_source=state.get("payload_source", "template"),
        attack_channel=attack_channel,
        modified_query=(state.get("modified_query") or None) if attack_channel == "query" else None,
        exploit_prompt_template_hash=payload_metadata.get("exploit_prompt_template_hash"),
    )

    execution = ExecutionBlock(
        query=state["query"],
        query_id=state["query_id"],
        index_state_hash=state["index_state_hash"],
        retrieved_docs=_retrieved_doc_records(state.get("retrieved_docs", [])),
        generator_output=state.get("generator_output", "") or "",
        generator_latency_ms=float(state.get("generator_latency_ms", 0.0)),
        baseline_top1_doc_id=_baseline_top1_doc_id(state),
    )

    evaluation = EvaluationBlock(
        ragas_faithfulness=state.get("ragas_faithfulness"),
        ragas_answer_relevance=state.get("ragas_answer_relevance"),
        ragas_context_relevance=state.get("ragas_context_relevance"),
        asr_retrieval=bool(state["asr_retrieval"]),
        asr_answer=bool(state["asr_answer"]),
        asr_target=bool(state["asr_target"]),
        # ASR-deny is wired into `evaluate_node` from Day 8 onwards, so it
        # is always populated as a bool. The schema field is kept Optional
        # for backwards compatibility with any pre-Day-8 bundles still on
        # disk; for fresh runs the field is never None.
        asr_deny=state.get("asr_deny"),
        rank_shift_at_k=int(state.get("rank_shift_at_k", 0)),
        verdict=state["verdict"],
        evaluator_notes=state.get("ragas_notes"),
        iteration_history=list(state.get("history", [])),
    )

    repro = Reproducibility(
        git_commit=_git_short_sha(),
        python_version=_python_version(),
        key_dependencies=_key_dependency_versions(),
    )

    # Build the headline summary block from the same fields the
    # detailed blocks read. This is the *only* place the summary is
    # constructed, so it cannot drift out of sync with the rest of the
    # bundle.
    summary = BundleSummary(
        verdict=evaluation.verdict,
        query_id=execution.query_id,
        attack_family=attack.family,
        attack_strategy=attack.strategy,
        attack_channel=attack.attack_channel,
        asr_retrieval=evaluation.asr_retrieval,
        asr_answer=evaluation.asr_answer,
        asr_target=evaluation.asr_target,
        asr_deny=evaluation.asr_deny,
        rank_shift_at_k=evaluation.rank_shift_at_k,
        ragas_faithfulness=evaluation.ragas_faithfulness,
        ragas_answer_relevance=evaluation.ragas_answer_relevance,
        ragas_context_relevance=evaluation.ragas_context_relevance,
        generator_latency_ms=execution.generator_latency_ms,
    )

    return ExploitBundle(
        bundle_version=BUNDLE_VERSION,
        summary=summary,
        run_id=state["run_id"],
        timestamp_utc=utc_now_iso(),
        seed=int(state.get("seed", 0)),
        framework_version=FRAMEWORK_VERSION,
        target_system=target_system,
        attack=attack,
        execution=execution,
        evaluation=evaluation,
        reproducibility=repro,
    )
