"""ε-greedy planner with global success-rate memory.

Spec §4.2 calls for a planner that maintains a memory of
`(attack_family, success_rate)` and selects via ε-greedy with ε=0.3 to
encourage exploration. Day 5's deterministic round-robin is replaced here
with this adaptive variant; it is the "agentic" property relevant to RQ2
(does adaptation improve attack success?).

Bucketing decision (Day 6): a single global memory across all queries,
rather than per-query-type buckets. With 50 queries and 6–8 plausible buckets
(`who`/`when`/`where`/...) each bucket would carry ~7 samples — too thin for
ε-greedy to converge meaningfully. Per-bucket memory is logged in
`FUTURE_WORKS.md` §6 as a refinement once larger query sets are run.

Determinism: the RNG is seeded at construction and explicit; re-running the
same experiment matrix produces identical exploration choices.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from redteam.orchestration.state import AttackFamily

# The attack families this project supports. Originally a 2-element closed
# set (spec §2 — IPI + corpus poisoning); Day 7.5 added `query_injection`
# as the third family (input-channel attack pulled in from FUTURE_WORKS §2.1
# during the Day-7 buffer). The ε-greedy planner is now a 3-armed bandit;
# uniform-exploration probability per family becomes 1/3 instead of 1/2.
ATTACK_FAMILIES: tuple[AttackFamily, ...] = (
    "prompt_injection",
    "corpus_poisoning",
    "query_injection",
)


@dataclass
class Planner:
    """ε-greedy planner with a single global success-rate memory.

    `success_rate(family) = successes / max(attempts, 1)`. With probability
    `1 - epsilon` the planner picks the family with the highest success rate
    (ties broken by RNG); with probability `epsilon` it picks uniformly at
    random. Per spec §4.2 the default ε=0.3.
    """

    epsilon: float = 0.3
    seed: int = 42
    # Counters keyed by attack family. A separate `successes` count + total
    # `attempts` count avoids float-state drift across many updates.
    successes: dict[AttackFamily, int] = field(default_factory=dict)
    attempts: dict[AttackFamily, int] = field(default_factory=dict)
    # `_rng` is private — Python's stdlib Random is not pickleable in some
    # nested contexts; but for our purposes it is. Seeded at __post_init__.
    _rng: random.Random = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)
        # Pre-populate the memory dicts so success_rate() never KeyErrors.
        for fam in ATTACK_FAMILIES:
            self.successes.setdefault(fam, 0)
            self.attempts.setdefault(fam, 0)

    def success_rate(self, family: AttackFamily) -> float:
        """Empirical success rate for `family`. Returns 0.0 if untried."""
        a = self.attempts.get(family, 0)
        if a == 0:
            return 0.0
        return self.successes.get(family, 0) / a

    def select(self, query_text: str) -> AttackFamily:
        """ε-greedy family selection.

        `query_text` is accepted (and ignored under the global-memory choice)
        so the call signature is stable when per-bucket memory is added later.
        """
        del query_text  # unused under global-memory bucketing
        if self._rng.random() < self.epsilon:
            # Explore: uniform random over the attack families.
            return self._rng.choice(list(ATTACK_FAMILIES))
        # Exploit: argmax of success-rate, RNG-shuffled to break ties fairly.
        families = list(ATTACK_FAMILIES)
        self._rng.shuffle(families)
        return max(families, key=self.success_rate)

    def update(
        self,
        query_text: str,
        family: AttackFamily,
        asr_t: bool,
    ) -> None:
        """Record the verdict of one (query, family) attempt."""
        del query_text  # unused under global-memory bucketing
        self.attempts[family] = self.attempts.get(family, 0) + 1
        if asr_t:
            self.successes[family] = self.successes.get(family, 0) + 1

    def snapshot(self) -> dict[str, dict[AttackFamily, int | float]]:
        """JSON-friendly view of the planner's state — for bundle history."""
        return {
            "attempts": dict(self.attempts),
            "successes": dict(self.successes),
            "success_rate": {f: self.success_rate(f) for f in ATTACK_FAMILIES},
            "epsilon": self.epsilon,
        }
