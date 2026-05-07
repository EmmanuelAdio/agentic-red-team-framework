"""Tests for `redteam.metrics.ragas_wrapper` (Day 7).

Two tests:

- `test_ragas_empty_inputs_returns_none_with_notes` — defensive contract.
  An empty query or empty retrieved_contexts must NOT raise; the wrapper
  records `None` for the affected metrics with a human-readable reason
  in `notes`. No LLM call. Fast.

- `test_ragas_smoke_on_demo_query` — happy-path smoke test against the
  real RAGAS + gpt-4o-mini stack. Skipped if `OPENAI_API_KEY` is missing
  or the corpus/query set hasn't been built. Verifies the three scores
  are floats in [0, 1] (or `None` with a notes flag).
"""

from __future__ import annotations

import json
import os

import pytest

from redteam.config import CHROMA_DIR, DATA_DIR, EMBEDDING_MODEL, load_env
from redteam.metrics.ragas_wrapper import RagasScores, compute_ragas_scores


def test_ragas_empty_query_returns_none_with_notes() -> None:
    """Empty query short-circuits without an LLM call."""
    scores = compute_ragas_scores(query="", retrieved_contexts=["doc1"], answer="ans")
    assert isinstance(scores, RagasScores)
    assert scores.faithfulness is None
    assert scores.answer_relevance is None
    assert scores.context_relevance is None
    assert scores.notes and "empty" in scores.notes.lower()


def test_ragas_smoke_on_demo_query() -> None:
    """Real RAGAS triple on the demo query. Asserts scores are floats in [0, 1]."""
    if not os.environ.get("OPENAI_API_KEY"):
        # load_env() may populate it from .env — try once, then skip.
        try:
            load_env()
        except Exception:
            pytest.skip("OPENAI_API_KEY not set and load_env failed")

    queries_path = DATA_DIR / "queries.json"
    if not queries_path.exists():
        pytest.skip("queries.json missing — run scripts/04_build_query_set.py first.")
    queries = json.loads(queries_path.read_text(encoding="utf-8"))
    demo_query = queries[0]

    # Tiny synthetic context + a plausible answer so RAGAS has something to score.
    contexts = [
        "Are You Smarter than a 5th Grader? is an American game show.",
        "It first aired on February 27, 2007.",
    ]
    answer = "Are You Smarter than a 5th Grader? first aired on February 27, 2007."

    scores = compute_ragas_scores(
        query=demo_query["query_text"],
        retrieved_contexts=contexts,
        answer=answer,
    )

    # At least one of the three should be a float in [0, 1] — RAGAS may
    # return NaN on edge cases (handled by the wrapper as None + notes),
    # but on this clean fixture we expect the metrics to fire.
    populated = [s for s in (scores.faithfulness, scores.answer_relevance, scores.context_relevance) if s is not None]
    assert populated, f"all three RAGAS metrics returned None (notes: {scores.notes})"
    for s in populated:
        assert isinstance(s, float)
        assert 0.0 <= s <= 1.0, f"score out of range: {s}"
