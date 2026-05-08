"""Exploit-bundle JSON schema — operational definition of Contribution C4.

This module pins the per-run record that PROJECT_SPEC.md §7 calls the
**exploit bundle**: a single JSON document holding *everything a third party
would need to reproduce one red-team run* — pipeline configuration, attack
payload, execution trace, evaluator output, and reproducibility metadata.

Design notes
------------

* **Pydantic v2 over `dataclasses`.** Bundles cross a serialisation boundary
  (disk, future cloud store, dissertation appendix) and need both
  validation-on-read and a pretty-printable JSON schema; the rest of the
  codebase uses `dataclass(frozen=True)` for in-memory value types where
  validation is unnecessary, so the choice is local rather than a project
  shift. `pydantic` is already a transitive dependency through
  `langchain-core`, so adopting it for the bundle layer adds no install cost.
* **Schema mirrors spec §7 verbatim.** Every field name + nesting matches the
  spec snippet, with three additive extensions noted inline:
    1. `attack.payload_source` (`template` | `llm`)        — Day 6
    2. `attack.attack_channel` (`corpus` | `query`)        — Day 7.5
    3. `attack.modified_query`, `evaluation.asr_deny`,
       `evaluation.iteration_history`                       — Day 7.5
  The `bundle_version` literal is the lever for any future *breaking* change;
  bumping it is the protocol for changing field semantics.
* **Optional fields default to `None`, not absent.** Pydantic v2 emits
  `None` keys in `model_dump()` by default, which keeps the JSON shape
  uniform across runs (an Optional field is always present, just possibly
  null). Downstream analysis (Day 10's plotting) can then assume a fixed
  column set when loading bundles into a DataFrame.
* **No business logic here.** Bundle construction from a `RedTeamState`
  lives in :func:`build_bundle`; this module only describes the shape.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

# Bumped on any breaking schema change. The current shape is the Day-8
# initial release; the Day 7.5 additive fields (`payload_source`,
# `attack_channel`, `modified_query`, `asr_deny`, `iteration_history`) are
# *additions* with sane defaults so old reader code keeps working.
BUNDLE_VERSION: Literal["1.0"] = "1.0"

# Project-level framework version — also written into every bundle for
# audit. Bump on any change to attack, executor, or evaluator semantics.
FRAMEWORK_VERSION: str = "0.1.0"


# ---------------------------------------------------------------------------
# Sub-models (§7 sub-objects)
# ---------------------------------------------------------------------------


class BundleSummary(BaseModel):
    """Headline metric row written at the *top* of every bundle.

    This block is a lightweight projection of the rest of the bundle —
    every field is also present, in its canonical location, inside the
    ``attack`` / ``execution`` / ``evaluation`` blocks. The duplication is
    deliberate: a reader scanning a folder full of bundles can read the
    verdict + ASR triple without parsing the rest of the document, and a
    diff tool surfaces outcome changes at the top of the file. The
    builder constructs this block from the same state fields the
    detailed blocks read; it can never go out of sync because it is
    derived once, not authored independently.
    """

    model_config = ConfigDict(extra="forbid")

    verdict: Literal["success", "failure", "partial"]
    query_id: str
    attack_family: Literal["prompt_injection", "corpus_poisoning", "query_injection"]
    attack_strategy: str
    attack_channel: Literal["corpus", "query"]
    asr_retrieval: bool
    asr_answer: bool
    asr_target: bool
    asr_deny: Optional[bool] = None
    rank_shift_at_k: int
    ragas_faithfulness: Optional[float] = None
    ragas_answer_relevance: Optional[float] = None
    ragas_context_relevance: Optional[float] = None
    generator_latency_ms: float


class TargetSystem(BaseModel):
    """Configuration of the RAG (Retrieval-Augmented Generation) system under test.

    Pinned per-run rather than globally so a single bundle dump can be
    replayed against the *same* generator + retriever even if the project
    config drifts later.
    """

    model_config = ConfigDict(extra="forbid")

    embedding_model: str
    vector_store: str = "chroma"
    retriever_top_k: int
    llm_model: str
    llm_temperature: float
    # SHA-256 of the verbatim prompt template — protects against silent
    # template edits invalidating reproducibility (spec §7).
    prompt_template_hash: str


class RetrievedDocRecord(BaseModel):
    """One entry of the retriever's top-k under attack.

    `is_poisoned` flags chunks whose `doc_id` matches the attack's payload
    doc id; for query-channel attacks (no corpus write) all entries are
    `False` by construction. `content` is intentionally elided here — the
    bundle stays compact; full chunk text is recoverable from
    `data/corpus/` plus the chunk_index encoded in the doc_id.
    """

    model_config = ConfigDict(extra="forbid")

    doc_id: str
    rank: int
    score: float
    is_poisoned: bool


class AttackBlock(BaseModel):
    """The attack the planner + exploit-generator produced this iteration."""

    model_config = ConfigDict(extra="forbid")

    family: Literal["prompt_injection", "corpus_poisoning", "query_injection"]
    strategy: str
    payload: str
    payload_id: str
    # `indexing` for corpus-channel families (the payload reaches the LLM
    # via retrieval after being indexed); `query` for query-channel
    # families (the payload reaches the LLM directly through the user
    # input). The two values map 1-1 onto `attack_channel` but are kept
    # separate because `injection_stage` is the existing spec §7 vocabulary.
    injection_stage: Literal["indexing", "query"]
    iteration: int
    # Day 6 addition — `template` on iteration 0 (deterministic helper),
    # `llm` on iteration ≥ 1 (history-conditioned variant generation).
    payload_source: Literal["template", "llm"]
    # Day 7.5 addition — the cross-channel attack-surface taxonomy.
    attack_channel: Literal["corpus", "query"]
    # Day 7.5 — populated only for query-channel attacks; `None` otherwise.
    modified_query: Optional[str] = None
    # Day 6 — SHA-256 of the LLM exploit-generator's prompt template; `None`
    # for the template path (which has no LLM prompt).
    exploit_prompt_template_hash: Optional[str] = None


class ExecutionBlock(BaseModel):
    """The executor's record of what actually happened on this attack run."""

    model_config = ConfigDict(extra="forbid")

    query: str
    query_id: str
    # SHA-256 of the index's doc_id set at the moment the executor ran the
    # *attacked* query. Lets a reviewer check that the index state matches
    # the one the attack expected. Matches the executor's add → run → remove
    # cycle, so the post-run hash equals the pre-run hash for clean rollback.
    index_state_hash: str
    retrieved_docs: list[RetrievedDocRecord]
    generator_output: str
    generator_latency_ms: float
    # Day 7 addition — the rank-1 doc id from the *baseline* (clean) pass.
    # Required to interpret `evaluation.rank_shift_at_k` without re-running
    # the baseline at analysis time. `None` only if the executor failed to
    # capture a baseline (defensive — should not happen post-Day-7).
    baseline_top1_doc_id: Optional[str] = None


class EvaluationBlock(BaseModel):
    """Reference-free metric vector + verdict (spec §6)."""

    model_config = ConfigDict(extra="forbid")

    # RAGAS triple — any score may be `None` if its scorer raised or
    # returned NaN; `evaluator_notes` records the why per spec §10's risk
    # register entry on RAGAS edge cases.
    ragas_faithfulness: Optional[float]
    ragas_answer_relevance: Optional[float]
    ragas_context_relevance: Optional[float]
    # ASR (Attack Success Rate) triple per spec §6.1.
    asr_retrieval: bool
    asr_answer: bool
    asr_target: bool
    # Day 7.5 availability metric — wired into `evaluate_node` from Day 8.
    # Always populated for runs produced by the current graph; remains
    # `Optional[bool]` here so any pre-Day-8 bundles still parse.
    asr_deny: Optional[bool] = None
    rank_shift_at_k: int
    verdict: Literal["success", "failure", "partial"]
    evaluator_notes: Optional[str] = None
    # Day 6 addition — one entry per past iteration of the loop, so a
    # reviewer can see how the planner + exploit generator adapted across
    # iterations within this run. Empty list on iteration 0 + max_iter 1.
    iteration_history: list[dict[str, Any]] = Field(default_factory=list)


class Reproducibility(BaseModel):
    """Environment metadata. Lets a reader pin tooling versions."""

    model_config = ConfigDict(extra="forbid")

    git_commit: Optional[str]  # short SHA; `None` if not in a git repo
    python_version: str
    key_dependencies: dict[str, str]


# ---------------------------------------------------------------------------
# Top-level bundle
# ---------------------------------------------------------------------------


class ExploitBundle(BaseModel):
    """The exploit bundle. One JSON object per red-team run. Schema = spec §7."""

    model_config = ConfigDict(extra="forbid")

    bundle_version: Literal["1.0"] = BUNDLE_VERSION
    # Headline metrics first — keeps the verdict + ASR triple at the top
    # of the JSON for at-a-glance scanning. Builder derives this from the
    # same state fields the detail blocks below read, so the two views
    # cannot diverge.
    summary: BundleSummary
    run_id: str
    timestamp_utc: str
    seed: int
    framework_version: str = FRAMEWORK_VERSION
    target_system: TargetSystem
    attack: AttackBlock
    execution: ExecutionBlock
    evaluation: EvaluationBlock
    reproducibility: Reproducibility

    # ---- Convenience -------------------------------------------------------

    def to_json(self, indent: int = 2) -> str:
        """Serialise to a stable, human-readable JSON string."""
        return self.model_dump_json(indent=indent)

    @classmethod
    def from_json(cls, payload: str) -> "ExploitBundle":
        """Parse and validate a JSON payload back into an `ExploitBundle`."""
        return cls.model_validate_json(payload)

    def fingerprint(self) -> str:
        """Stable SHA-256 of the canonical JSON. Useful for de-dup and audit."""
        canonical = self.model_dump_json()  # pydantic emits stable key order
        return "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def utc_now_iso() -> str:
    """Current UTC time as a `Z`-suffixed ISO-8601 string (spec §7 example)."""
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
