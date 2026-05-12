"""Day 10 clean-condition baseline - persisted RAGAS triple + ASR-r per query.

The Day-9 attack matrix produced per-cell Faithfulness / Answer-Relevance /
Context-Relevance distributions for the *attacked* condition (one row per
(seed, cell, query) under ``results/runs/.../runs[]``). To support the
Chapter-6 clean-vs-attacked comparison the spec calls for in section 6.4
("Faithfulness distributions clean vs attacked (violin or histogram)") we
need the *unattacked* counterpart on disk. The Day-1 baseline script
(:mod:`scripts.02_run_baseline`) prints to stdout but doesn't compute
RAGAS and doesn't persist - it was a retrieval-only sanity script. This
script is the persistence + RAGAS counterpart, run once on Day 10 before
the plotting pass.

What it produces
----------------

Two artefacts under ``results/baseline/``:

::

    results/baseline/
      baseline_<ts>.json          # per-query rows + summary block
      baseline_latest.json        # symlink-style copy of the most recent run

Per-query row shape (matches the attacked-side ``runs[]`` row in
``results/runs/batch_*/batch_*_summary.json`` so the Day-10 plotter
treats both sides the same way)::

    {
      "query_id": "test1195",
      "query_text": "...",
      "gold_doc_ids": ["doc42525"],
      "retrieved_doc_ids": ["doc42525", "doc99988", ...],
      "retrieved_top_scores": [0.92, 0.81, ...],
      "asr_retrieval_clean": true,                # gold-in-top-k
      "top1_is_gold": true,
      "generator_output": "The show first aired in 2007.",
      "generator_latency_ms": 842.3,
      "ragas_faithfulness": 0.95,
      "ragas_answer_relevance": 0.88,
      "ragas_context_relevance": 0.74,
      "ragas_notes": null
    }

The summary block carries the same aggregates the per-cell summaries
carry, so plotting code can swap a clean batch for an attacked batch
without branching on schema.

Why one seed is enough for the clean baseline
---------------------------------------------

The attacked side runs n=3 seeds because the attacks are stochastic
(planner ε-greedy, exploit-generator LLM at temperature 0 but with an
internal RNG for retry, jamming uses a sampled distractor pool). The
clean pipeline is fully deterministic at temperature 0 and a fixed
embedding model: rerunning produces the same RAGAS scores up to LLM
non-determinism in the RAGAS scorer's own LLM calls (temperature 0 there
too, with the global SQLite cache absorbing duplicates). Running the
clean baseline at three seeds would produce three near-identical traces
and waste the API budget. Documented as a methodology decision in the
Day-10 lab notebook entry.

CLI
---

::

    # Full clean baseline over the 50-query test set
    python scripts/07_run_clean_baseline.py

    # Quick smoke (e.g. dev-time)
    python scripts/07_run_clean_baseline.py --limit 3 --no-ragas
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

# Make `redteam` importable when running the script directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from redteam.config import (
    CHROMA_DIR,
    DATA_DIR,
    EMBEDDING_MODEL,
    RESULTS_DIR,
    load_env,
)
from redteam.metrics.ragas_wrapper import compute_ragas_scores
from redteam.target.generator import LLMGenerator
from redteam.target.pipeline import RAGPipeline
from redteam.target.retriever import Retriever

QUERIES_PATH = DATA_DIR / "queries.json"
BASELINE_DIR = RESULTS_DIR / "baseline"


def _safe_mean(values: list[float | None]) -> float | None:
    """Mean of non-None values; None if all entries are None.

    Mirrors :func:`scripts.06_run_experiments._summarise_cell` so the
    clean-side aggregates are computed identically to the attacked-side
    aggregates.
    """
    kept = [v for v in values if v is not None]
    return (sum(kept) / len(kept)) if kept else None


def _run_clean_baseline(
    queries: list[dict[str, Any]],
    pipeline: RAGPipeline,
    with_ragas: bool,
) -> list[dict[str, Any]]:
    """Run each query through the unattacked pipeline + RAGAS triple.

    Per-query failures (RAGAS exceptions, generator errors) are caught and
    recorded as ``None`` scores so one bad query doesn't sink the run -
    the same defensive pattern as the experiment driver's per-cell loop.
    """
    rows: list[dict[str, Any]] = []
    n_q = len(queries)

    for i, q in enumerate(queries, start=1):
        gold = set(q["gold_doc_ids"])
        try:
            result = pipeline.run(q["query_text"])
        except Exception as exc:  # one bad query shouldn't sink the baseline
            print(
                f"  [{i:02d}/{n_q}] q={q['query_id']}: "
                f"PIPELINE-ERROR {type(exc).__name__}: {exc}"
            )
            continue

        retrieved_ids = [d["doc_id"] for d in result["retrieved_docs"]]
        retrieved_scores = [float(d["score"]) for d in result["retrieved_docs"]]
        in_topk = bool(set(retrieved_ids) & gold)
        top1_is_gold = bool(retrieved_ids[:1] and retrieved_ids[0] in gold)

        # RAGAS triple over the clean retrieved context + clean answer.
        if with_ragas:
            contexts = [d["content"] for d in result["retrieved_docs"]]
            scores = compute_ragas_scores(
                query=q["query_text"],
                retrieved_contexts=contexts,
                answer=result["generator_output"],
            )
            f, ar, cr, notes = (
                scores.faithfulness,
                scores.answer_relevance,
                scores.context_relevance,
                scores.notes,
            )
        else:
            f, ar, cr, notes = None, None, None, "ragas-disabled"

        row: dict[str, Any] = {
            "query_id": q["query_id"],
            "query_text": q["query_text"],
            "gold_doc_ids": list(q["gold_doc_ids"]),
            "retrieved_doc_ids": retrieved_ids,
            "retrieved_top_scores": retrieved_scores,
            "asr_retrieval_clean": in_topk,
            "top1_is_gold": top1_is_gold,
            "generator_output": result["generator_output"],
            "generator_latency_ms": float(result["generator_latency_ms"]),
            "ragas_faithfulness": f,
            "ragas_answer_relevance": ar,
            "ragas_context_relevance": cr,
            "ragas_notes": notes,
        }
        rows.append(row)

        print(
            f"  [{i:02d}/{n_q}] q={q['query_id']}: "
            f"top5_has_gold={int(in_topk)} top1_gold={int(top1_is_gold)} "
            f"F={('%.2f' % f) if f is not None else '--'} "
            f"AR={('%.2f' % ar) if ar is not None else '--'} "
            f"CR={('%.2f' % cr) if cr is not None else '--'} "
            f"lat={result['generator_latency_ms']:.0f}ms"
        )

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only run the first N queries (default: all).",
    )
    ragas_group = parser.add_mutually_exclusive_group()
    ragas_group.add_argument(
        "--with-ragas",
        dest="with_ragas",
        action="store_true",
        help="Compute the RAGAS triple per query (default for Day-10 baseline).",
    )
    ragas_group.add_argument(
        "--no-ragas",
        dest="with_ragas",
        action="store_false",
        help="Skip RAGAS - retrieval-only baseline, much faster.",
    )
    parser.set_defaults(with_ragas=True)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=BASELINE_DIR,
        help="Destination directory (default: results/baseline/).",
    )
    args = parser.parse_args()

    load_env()

    if not QUERIES_PATH.exists():
        raise SystemExit(
            f"{QUERIES_PATH} not found. Run `python scripts/04_build_query_set.py` first."
        )
    queries = json.loads(QUERIES_PATH.read_text(encoding="utf-8"))
    if args.limit is not None:
        queries = queries[: args.limit]
    if not queries:
        raise SystemExit("queries.json is empty.")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    retriever = Retriever(persist_dir=CHROMA_DIR, embedding_model_name=EMBEDDING_MODEL)
    if retriever._count() == 0:
        raise SystemExit("Chroma is empty. Run `python scripts/01_build_corpus.py` first.")

    pipeline = RAGPipeline(retriever=retriever, generator=LLMGenerator())

    batch_ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    print(
        f"Day-10 clean baseline: {len(queries)} queries, "
        f"ragas={'on' if args.with_ragas else 'off'}, ts={batch_ts}\n"
    )
    t_start = time.perf_counter()
    rows = _run_clean_baseline(queries, pipeline, with_ragas=args.with_ragas)
    wall_s = time.perf_counter() - t_start

    n = len(rows)
    summary: dict[str, Any] = {
        "schema_version": "1.0",
        "kind": "clean_baseline",
        "batch_ts": batch_ts,
        "n_queries_input": len(queries),
        "n_queries_completed": n,
        "with_ragas": args.with_ragas,
        "wall_seconds": round(wall_s, 2),
        "asr_retrieval_clean_total": sum(int(r["asr_retrieval_clean"]) for r in rows),
        "top1_is_gold_total": sum(int(r["top1_is_gold"]) for r in rows),
        "mean_latency_ms": (
            sum(r["generator_latency_ms"] for r in rows) / n if n else None
        ),
        "mean_ragas_faithfulness": _safe_mean([r["ragas_faithfulness"] for r in rows]),
        "mean_ragas_answer_relevance": _safe_mean([r["ragas_answer_relevance"] for r in rows]),
        "mean_ragas_context_relevance": _safe_mean([r["ragas_context_relevance"] for r in rows]),
    }
    payload = {**summary, "rows": rows}

    out_path = args.out_dir / f"baseline_{batch_ts}.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # Stable pointer to the most recent run for the plotting code.
    # We write a real copy rather than a symlink because Windows symlinks
    # require admin privileges by default and the project's reproducibility
    # primitives forbid platform-specific behaviour (METHODOLOGY.md section 4.6).
    latest_path = args.out_dir / "baseline_latest.json"
    latest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print()
    print("=" * 72)
    print(
        f"DONE: clean baseline over {n}/{len(queries)} queries "
        f"in {wall_s:.1f}s ({wall_s / 60:.1f} min)"
    )
    print(
        f"  ASR-r (gold in top-5):     "
        f"{summary['asr_retrieval_clean_total']}/{n}  "
        f"({summary['asr_retrieval_clean_total'] / n:.1%})"
    )
    print(
        f"  top1 == gold:               "
        f"{summary['top1_is_gold_total']}/{n}  "
        f"({summary['top1_is_gold_total'] / n:.1%})"
    )
    if args.with_ragas:
        f_mean = summary["mean_ragas_faithfulness"]
        ar_mean = summary["mean_ragas_answer_relevance"]
        cr_mean = summary["mean_ragas_context_relevance"]
        print(
            f"  RAGAS mean F / AR / CR:    "
            f"{('%.3f' % f_mean) if f_mean is not None else '--'} / "
            f"{('%.3f' % ar_mean) if ar_mean is not None else '--'} / "
            f"{('%.3f' % cr_mean) if cr_mean is not None else '--'}"
        )
    print(f"  Wrote: {out_path}")
    print(f"  Latest pointer: {latest_path}")


if __name__ == "__main__":
    main()
