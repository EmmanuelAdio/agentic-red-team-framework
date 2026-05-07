"""Tests for `redteam.metrics.rank_shift` (Day 7 — TDD).

Spec §6.3: rank_shift_at_k = "the change in rank position of the originally
top-1 clean document". So we look up `baseline_retrieved[0].doc_id` in the
attacked top-k and report (attacked_rank - 1). If the originally top-1 doc
fell out of the attacked top-k, we use a sentinel of `k + 1` so the metric
remains a non-negative integer that orders correctly.
"""

from __future__ import annotations

import pytest

from redteam.metrics.rank_shift import RankShift, compute_rank_shift


# ---------------------------------------------------------------------------
# Standard cases
# ---------------------------------------------------------------------------


def test_rank_shift_zero_when_top1_unchanged() -> None:
    """If the originally top-1 doc is still rank 1 under attack -> shift = 0."""
    baseline = [
        {"doc_id": "gold", "rank": 1},
        {"doc_id": "doc2", "rank": 2},
    ]
    attacked = [
        {"doc_id": "gold", "rank": 1},
        {"doc_id": "payload", "rank": 2},
    ]
    rs = compute_rank_shift(baseline, attacked, k=5)
    assert rs == RankShift(baseline_top1_doc_id="gold", attacked_rank=1, rank_shift=0)


def test_rank_shift_pushed_down_two_positions() -> None:
    """Originally rank 1, attacked rank 3 -> shift = 2."""
    baseline = [
        {"doc_id": "gold", "rank": 1},
        {"doc_id": "doc2", "rank": 2},
    ]
    attacked = [
        {"doc_id": "payload", "rank": 1},
        {"doc_id": "doc2", "rank": 2},
        {"doc_id": "gold", "rank": 3},
    ]
    rs = compute_rank_shift(baseline, attacked, k=5)
    assert rs.attacked_rank == 3
    assert rs.rank_shift == 2


def test_rank_shift_dropped_out_of_topk() -> None:
    """Originally rank 1, no longer in top-k -> attacked_rank=None, shift = k."""
    baseline = [{"doc_id": "gold", "rank": 1}]
    attacked = [
        {"doc_id": "payload", "rank": 1},
        {"doc_id": "filler1", "rank": 2},
        {"doc_id": "filler2", "rank": 3},
        {"doc_id": "filler3", "rank": 4},
        {"doc_id": "filler4", "rank": 5},
    ]
    rs = compute_rank_shift(baseline, attacked, k=5)
    assert rs.attacked_rank is None
    # Sentinel: max possible shift inside top-k. Originally rank 1; if we
    # treat "fell out" as rank k+1, the shift is k+1 - 1 = k.
    assert rs.rank_shift == 5


# ---------------------------------------------------------------------------
# Defensive
# ---------------------------------------------------------------------------


def test_rank_shift_empty_baseline_raises() -> None:
    """Empty baseline retrieved-list -> ValueError. There's no top-1 to track."""
    with pytest.raises(ValueError):
        compute_rank_shift([], [{"doc_id": "x", "rank": 1}], k=5)


def test_rank_shift_handles_attacked_rank_field_missing() -> None:
    """If `rank` field is absent from attacked dicts, infer from list position."""
    baseline = [{"doc_id": "gold", "rank": 1}]
    # No `rank` keys — the function should fall back to 1-based list index.
    attacked = [
        {"doc_id": "payload"},
        {"doc_id": "gold"},
        {"doc_id": "other"},
    ]
    rs = compute_rank_shift(baseline, attacked, k=5)
    assert rs.attacked_rank == 2
    assert rs.rank_shift == 1
