"""Unit tests for the ε-greedy planner (Day 6).

Pure-Python: no LLM, no Chroma, no I/O. Deterministic via seeded RNG.
"""

from __future__ import annotations

import pytest

from redteam.agents.planner import ATTACK_FAMILIES, Planner


def test_planner_greedy_picks_winner() -> None:
    """With ε=0 (no exploration) and one family scoring higher, planner picks it."""
    planner = Planner(epsilon=0.0, seed=42)

    # IPI succeeded twice; corpus_poisoning failed twice.
    planner.update("q", "prompt_injection", asr_t=True)
    planner.update("q", "prompt_injection", asr_t=True)
    planner.update("q", "corpus_poisoning", asr_t=False)
    planner.update("q", "corpus_poisoning", asr_t=False)

    # Greedy: every selection picks IPI.
    for _ in range(10):
        assert planner.select("q") == "prompt_injection"


def test_planner_explore_with_full_epsilon() -> None:
    """With ε=1 (always explore) the choice is RNG-driven, not argmax."""
    planner = Planner(epsilon=1.0, seed=42)

    # Even though IPI has all the success, the RNG should produce both
    # families across many calls.
    planner.update("q", "prompt_injection", asr_t=True)
    planner.update("q", "prompt_injection", asr_t=True)

    seen = {planner.select("q") for _ in range(50)}
    assert seen == set(ATTACK_FAMILIES), f"explore failed to cover both families, got {seen}"


def test_planner_update_increments_counts() -> None:
    """`update` modifies attempts/successes; success_rate reflects the ratio."""
    planner = Planner(seed=42)

    planner.update("q", "prompt_injection", asr_t=True)
    planner.update("q", "prompt_injection", asr_t=False)
    planner.update("q", "prompt_injection", asr_t=True)

    assert planner.attempts["prompt_injection"] == 3
    assert planner.successes["prompt_injection"] == 2
    assert planner.success_rate("prompt_injection") == pytest.approx(2 / 3)
    # Untried family stays at 0.
    assert planner.success_rate("corpus_poisoning") == 0.0


def test_planner_snapshot_is_json_friendly() -> None:
    """`snapshot()` returns plain dicts/floats so the bundle JSON serializer eats it."""
    import json

    planner = Planner(seed=42)
    planner.update("q", "prompt_injection", asr_t=True)
    snap = planner.snapshot()

    # Round-trips through json.dumps without raising.
    serialized = json.dumps(snap)
    loaded = json.loads(serialized)
    assert loaded["attempts"]["prompt_injection"] == 1
    assert loaded["successes"]["prompt_injection"] == 1
    assert loaded["epsilon"] == planner.epsilon


def test_planner_seeded_rng_is_deterministic() -> None:
    """Two planners with the same seed produce identical exploration sequences."""
    p1 = Planner(epsilon=1.0, seed=123)
    p2 = Planner(epsilon=1.0, seed=123)
    seq1 = [p1.select("q") for _ in range(20)]
    seq2 = [p2.select("q") for _ in range(20)]
    assert seq1 == seq2
