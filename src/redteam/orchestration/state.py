"""LangGraph state schema for the red-team workflow.

`RedTeamState` is the single dict that flows through all four nodes
(plan -> generate -> execute -> evaluate -> loop). Field shape mirrors
PROJECT_SPEC.md S5 verbatim, with one practical addition:

- `payload_doc_id` — the executor needs this to flag retrieved chunks as
  poisoned and to remove the payload during cleanup. Keeping it in state
  (rather than re-deriving from `payload_metadata`) makes the executor
  trivially stateless.

RAGAS (Retrieval-Augmented Generation Assessment) fields stay `None` until
Day 7's metrics module fills them in. `rank_shift_at_k` likewise stays 0
until Day 7.
"""

from __future__ import annotations

from typing import Any, Literal, Optional, TypedDict

# Attack family is a closed set per spec S2 ("Two attack families only").
AttackFamily = Literal["prompt_injection", "corpus_poisoning"]

# Verdict alphabet matches the bundle JSON `evaluation.verdict` field.
Verdict = Literal["success", "failure", "partial"]

# Payload provenance — which generator path produced the payload. Day 5 was
# template-only; Day 6 adds the "llm" path triggered on iteration >= 1.
PayloadSource = Literal["template", "llm"]


class RedTeamState(TypedDict, total=False):
    """State carried through one query's iteration loop in the LangGraph.

    `total=False` so individual nodes can write only the fields they own —
    the planner does not need to set executor fields, etc. The graph entry
    point is responsible for seeding identifiers + iteration counters.
    """

    # --- identifiers --------------------------------------------------------
    run_id: str
    seed: int
    query: str
    query_id: str

    # --- planner output -----------------------------------------------------
    attack_family: AttackFamily
    attack_strategy: str  # e.g. "instruction_override", "answer_replacement"
    iteration: int
    max_iterations: int

    # --- exploit-generator output ------------------------------------------
    payload: str  # the adversarial document body (page_content)
    payload_doc_id: str  # added: lets the executor track + remove cleanly
    payload_metadata: dict[str, Any]
    # Day 6: provenance flag — "template" on iteration 0, "llm" on retries.
    # Bundle JSON (Day 8) lifts this directly so reviewers can see which
    # generator path produced each recorded exploit.
    payload_source: PayloadSource

    # --- executor output ----------------------------------------------------
    index_state_hash: str
    retrieved_docs: list[dict[str, Any]]  # [{doc_id, rank, score, content, is_poisoned}]
    generator_output: str
    generator_latency_ms: float
    # Day 7: clean-pass results captured before the payload is inserted.
    # Powers `rank_shift_at_k` (compares baseline rank-1 vs attacked rank).
    # Cached once per query inside the executor's closure so iterating a
    # query multiple times only runs one baseline pass.
    baseline_retrieved_docs: list[dict[str, Any]]
    baseline_generator_output: str

    # --- evaluator output ---------------------------------------------------
    # RAGAS triple — stays None until Day 7.
    ragas_faithfulness: Optional[float]
    ragas_answer_relevance: Optional[float]
    ragas_context_relevance: Optional[float]
    # ASR (Attack Success Rate) triple — computed by `redteam.metrics.asr`.
    asr_retrieval: bool
    asr_answer: bool
    asr_target: bool  # Day 7: was implicit (asr_r and asr_a); now stored.
    rank_shift_at_k: int  # Day 7: computed by `redteam.metrics.rank_shift`.
    # Day 7: RAGAS notes — captures NaN/exception reasons for traceability.
    ragas_notes: Optional[str]
    verdict: Verdict

    # --- bookkeeping --------------------------------------------------------
    # One small dict per past iteration. Powers the loop decision and the
    # eventual exploit-bundle history field.
    history: list[dict[str, Any]]
