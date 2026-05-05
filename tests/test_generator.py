"""Unit tests for LLMGenerator — prompt template hash + cache behaviour.

The hash test is offline. The cache test makes a real LLM call but asserts
on the second call's latency, which should hit the SQLite cache and be fast.
"""

from __future__ import annotations

from redteam.config import load_env
from redteam.target.generator import PROMPT_TEMPLATE, PROMPT_TEMPLATE_HASH, LLMGenerator
from redteam.target.retriever import RetrievedDoc


def test_prompt_template_hash_matches_template() -> None:
    """Hash is SHA-256 of the literal template string. Stays put unless §4.1 changes."""
    import hashlib

    expected = "sha256:" + hashlib.sha256(PROMPT_TEMPLATE.encode("utf-8")).hexdigest()
    assert PROMPT_TEMPLATE_HASH == expected


def test_cache_hit_is_fast() -> None:
    """Second call with identical inputs should hit SQLiteCache (latency ~ 0)."""
    load_env()

    docs = [
        RetrievedDoc(
            doc_id="dummy",
            content="The capital of France is Paris.",
            score=0.99,
            rank=1,
        )
    ]
    gen = LLMGenerator()

    first = gen.generate("What is the capital of France?", docs)
    second = gen.generate("What is the capital of France?", docs)

    assert first.text == second.text
    # Cache hit should be <100 ms; first call may be much slower (real API).
    assert second.latency_ms < 100.0, f"second call latency = {second.latency_ms:.1f} ms (cache miss?)"
    assert first.prompt_template_hash == second.prompt_template_hash == PROMPT_TEMPLATE_HASH
