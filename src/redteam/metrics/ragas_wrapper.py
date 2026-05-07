"""RAGAS (Retrieval-Augmented Generation Assessment) wrapper — spec §6.2.

Computes the reference-free integrity triple:

- **Faithfulness** ∈ [0, 1] — fraction of generated claims supported by
  the retrieved context.
- **Answer Relevance** ∈ [0, 1] — embedding-similarity between the original
  query and a question reverse-engineered from the answer.
- **Context Relevance** ∈ [0, 1] — relevance of retrieved context to the
  query.

Each metric is computed by calling RAGAS's per-metric scorer with our
`gpt-4o-mini` (temperature 0). Per spec §10's risk register, every call
is wrapped in `try/except`; failures and NaN results land as `None` in
the returned dataclass with a human-readable reason in `notes`. This
preserves the why so reviewers can see whether a missing score is a
transient API blip, an empty-context edge case, or a model refusal.

Caching: RAGAS's underlying calls go through `langchain`'s LLM stack which
already has the global `SQLiteCache` set in `redteam.target.generator`.
Re-runs of the same (query, context, answer) triple hit the cache.

Cost: each scoring call is 1–2 LLM round-trips; the full triple is ~5
calls. For the Day-9 ~300-run matrix that's ~1500 RAGAS calls → ≪ $1
on `gpt-4o-mini`, well under the spec §2 cap.
"""

from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass
from typing import Any, Coroutine, Optional

from redteam.config import LLM_MODEL


def _run_async(coro: Coroutine[Any, Any, Any]) -> Any:
    """Run a coroutine synchronously, tolerating a live outer event loop.

    RAGAS 0.4's `BaseMetric.score()` *refuses* to run when an event loop is
    already running (it raises rather than calling `asyncio.run()`), so we
    bypass `score()` and drive `ascore()` ourselves. `nest_asyncio.apply()`
    (called once at scorer-build time) makes `asyncio.run()` reentrant in
    Jupyter; outside Jupyter there is no running loop and the call is the
    standard sync->async hop.
    """
    return asyncio.run(coro)


@dataclass(frozen=True)
class RagasScores:
    """RAGAS triple. Any score may be None if its scorer raised or returned NaN.

    The `notes` field carries a short human-readable reason for any None;
    multiple reasons are joined with `'; '`.
    """

    faithfulness: Optional[float]
    answer_relevance: Optional[float]
    context_relevance: Optional[float]
    notes: Optional[str] = None


def _to_float(value: Any) -> Optional[float]:
    """Coerce a RAGAS MetricResult.value to a clean float, or None on NaN."""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(f):
        return None
    return f


def _build_default_scorers(model: str = LLM_MODEL) -> dict[str, Any]:
    """Construct RAGAS scorers wired to the project's gpt-4o-mini.

    Imported lazily so the metrics module remains importable even if RAGAS
    is not installed or its API changes — the wrapper just records `None`
    + a notes flag and the bundle still serializes.
    """
    # RAGAS's per-metric .score() calls `asyncio.run(self.ascore(...))`,
    # which raises RuntimeError when called from a context that already
    # owns a running event loop (e.g. Jupyter / IPython kernel). Applying
    # `nest_asyncio` re-entrantly patches asyncio so the nested run() is
    # tolerated. No-op when nest_asyncio is unavailable; idempotent.
    try:
        import nest_asyncio
        nest_asyncio.apply()
    except ImportError:
        pass

    from openai import AsyncOpenAI
    from ragas.embeddings import OpenAIEmbeddings
    from ragas.llms import llm_factory
    from ragas.metrics.collections import (
        AnswerRelevancy,
        ContextRelevance,
        Faithfulness,
    )

    # RAGAS's async metric paths (`ascore` -> `llm.agenerate` /
    # `embeddings.aembed_text`) require *async* clients. The sync OpenAI
    # client refuses both `agenerate()` and `aembed_text()` with TypeError.
    # We use AsyncOpenAI for both LLM and embeddings.
    async_client = AsyncOpenAI()
    llm = llm_factory(model, provider="openai", client=async_client)
    # AnswerRelevancy needs an embedding model (it embeds the original
    # query and the LLM-reverse-engineered question and compares them).
    # Defaults to OpenAI's `text-embedding-3-small` — the RAGAS default.
    # We keep this separate from the project's bge-small retrieval
    # embedder; mixing the two would conflate retrieval geometry with
    # evaluation geometry. Recorded in DIAGRAMS.md §6.4.
    embeddings = OpenAIEmbeddings(client=async_client)

    return {
        "faithfulness": Faithfulness(llm=llm),
        "answer_relevance": AnswerRelevancy(llm=llm, embeddings=embeddings),
        "context_relevance": ContextRelevance(llm=llm),
    }


# Lazy module-level scorer cache: build once per process.
_SCORERS: Optional[dict[str, Any]] = None


def _get_scorers() -> dict[str, Any]:
    global _SCORERS
    if _SCORERS is None:
        _SCORERS = _build_default_scorers()
    return _SCORERS


def compute_ragas_scores(
    query: str,
    retrieved_contexts: list[str],
    answer: str,
) -> RagasScores:
    """Compute the RAGAS triple. Returns `None` per metric on failure with `notes`.

    `retrieved_contexts` is the list of `page_content` strings from the
    attacked retrieval pass — these are what the generator was conditioned
    on, so they are what RAGAS scores against.
    """
    if not query or answer is None:
        return RagasScores(
            faithfulness=None,
            answer_relevance=None,
            context_relevance=None,
            notes="empty query or answer",
        )

    notes_parts: list[str] = []
    f_score: Optional[float] = None
    ar_score: Optional[float] = None
    cr_score: Optional[float] = None

    # Build (or reuse) the scorers. If the RAGAS import fails, we record
    # the reason in notes and return all-None.
    try:
        scorers = _get_scorers()
    except Exception as exc:  # pragma: no cover - import-time path
        return RagasScores(
            faithfulness=None,
            answer_relevance=None,
            context_relevance=None,
            notes=f"ragas-init-failed: {exc.__class__.__name__}: {exc}",
        )

    # ---- Faithfulness (needs query + answer + contexts) -------------------
    if not retrieved_contexts:
        notes_parts.append("faithfulness skipped: empty retrieved_contexts")
    else:
        try:
            res = _run_async(
                scorers["faithfulness"].ascore(
                    user_input=query,
                    response=answer,
                    retrieved_contexts=retrieved_contexts,
                )
            )
            f_score = _to_float(getattr(res, "value", res))
            if f_score is None:
                notes_parts.append("faithfulness: NaN")
        except Exception as exc:
            notes_parts.append(f"faithfulness-error: {exc.__class__.__name__}")

    # ---- Answer relevance (needs query + answer) --------------------------
    try:
        res = _run_async(
            scorers["answer_relevance"].ascore(user_input=query, response=answer)
        )
        ar_score = _to_float(getattr(res, "value", res))
        if ar_score is None:
            notes_parts.append("answer_relevance: NaN")
    except Exception as exc:
        notes_parts.append(f"answer_relevance-error: {exc.__class__.__name__}")

    # ---- Context relevance (needs query + contexts) -----------------------
    if not retrieved_contexts:
        notes_parts.append("context_relevance skipped: empty retrieved_contexts")
    else:
        try:
            res = _run_async(
                scorers["context_relevance"].ascore(
                    user_input=query, retrieved_contexts=retrieved_contexts
                )
            )
            cr_score = _to_float(getattr(res, "value", res))
            if cr_score is None:
                notes_parts.append("context_relevance: NaN")
        except Exception as exc:
            notes_parts.append(f"context_relevance-error: {exc.__class__.__name__}")

    notes = "; ".join(notes_parts) if notes_parts else None
    return RagasScores(
        faithfulness=f_score,
        answer_relevance=ar_score,
        context_relevance=cr_score,
        notes=notes,
    )
