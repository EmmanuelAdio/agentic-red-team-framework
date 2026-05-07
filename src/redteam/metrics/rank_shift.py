"""rank_shift@k metric — spec §6.3.

Definition: the change in rank position of the originally top-1 clean
document under attack. Computed as `(attacked_rank - 1)`, where
`attacked_rank` is the position of the baseline top-1 doc in the attacked
top-k list.

If the originally top-1 doc fell out of the attacked top-k entirely we
report `attacked_rank=None` and `rank_shift=k` — the maximum possible shift
inside the window. This makes `rank_shift` a non-negative integer that
orders correctly across runs (larger = more displacement = stronger
retrieval-side attack effect).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass(frozen=True)
class RankShift:
    """Result of one rank_shift@k computation."""

    baseline_top1_doc_id: str
    attacked_rank: Optional[int]  # None if dropped out of top-k
    rank_shift: int  # 0 if unchanged; k if dropped out


def _doc_rank(retrieved: list[dict[str, Any]], target_doc_id: str) -> Optional[int]:
    """Return 1-based rank of `target_doc_id` in `retrieved`, or None if absent.

    Prefers the `rank` field if present (matches the dict shape that
    `RAGPipeline.run` produces); otherwise falls back to 1-based list index.
    """
    for i, d in enumerate(retrieved, start=1):
        if d.get("doc_id") != target_doc_id:
            continue
        return int(d.get("rank", i))
    return None


def compute_rank_shift(
    baseline_retrieved: list[dict[str, Any]],
    attacked_retrieved: list[dict[str, Any]],
    k: int = 5,
) -> RankShift:
    """rank_shift@k: how far did the originally top-1 doc move under attack?"""
    if not baseline_retrieved:
        raise ValueError("baseline_retrieved is empty — no top-1 doc to track")

    top1_doc_id = baseline_retrieved[0]["doc_id"]
    attacked_rank = _doc_rank(attacked_retrieved, top1_doc_id)

    if attacked_rank is None:
        # Dropped out of top-k. Treat as if it landed at rank k+1; shift is k.
        rank_shift = k
    else:
        rank_shift = attacked_rank - 1

    return RankShift(
        baseline_top1_doc_id=top1_doc_id,
        attacked_rank=attacked_rank,
        rank_shift=rank_shift,
    )
