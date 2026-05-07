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

    # --- executor output ----------------------------------------------------
    index_state_hash: str
    retrieved_docs: list[dict[str, Any]]  # [{doc_id, rank, score, content, is_poisoned}]
    generator_output: str
    generator_latency_ms: float

    # --- evaluator output ---------------------------------------------------
    # RAGAS triple — stays None until Day 7.
    ragas_faithfulness: Optional[float]
    ragas_answer_relevance: Optional[float]
    ragas_context_relevance: Optional[float]
    # ASR (Attack Success Rate) triple — Day 5 fills these inline; Day 7
    # moves the logic into `redteam.metrics.asr`.
    asr_retrieval: bool
    asr_answer: bool
    rank_shift_at_k: int
    verdict: Verdict

    # --- bookkeeping --------------------------------------------------------
    # One small dict per past iteration. Powers the loop decision and the
    # eventual exploit-bundle history field.
    history: list[dict[str, Any]]
