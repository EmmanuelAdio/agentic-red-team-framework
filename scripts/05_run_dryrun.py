"""Day 8 dry run — first 50 exploit bundles → ``data/runs/``.

Wires the four-node LangGraph end-to-end against the 50-query NQ slice and
materialises one :class:`ExploitBundle` per query, written through
:class:`BundleStore` to ``data/runs/``. This is the spec §9 Day-8
deliverable ("First 50 bundles written to disk") and the precursor to the
Day-9 full experiment matrix (50 queries x 2 families x 3 seeds = ~300
bundles); the dry run gives us:

* an end-to-end check that the graph + bundle layer compose cleanly,
* a measured per-run cost / latency baseline before the matrix multiplies it,
* a JSON corpus the Day-10 plotting code can already start parsing.

Run from repo root::

    python scripts/05_run_dryrun.py                # 50 queries, no RAGAS
    python scripts/05_run_dryrun.py --limit 5      # quick smoke
    python scripts/05_run_dryrun.py --with-ragas   # full evaluator (LLM-cost)
    python scripts/05_run_dryrun.py --seed 17      # alternate RNG seed

Defaults to **RAGAS off** because the dry run's purpose is the *pipeline*
contract, not the integrity scores; flipping `--with-ragas` brings RAGAS
back in for the final pre-Day-9 sanity sweep. Day 9's experiment driver
will leave RAGAS on by default.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

# Make `redteam` importable when running the script directly without `pip install -e .`.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from redteam.agents.exploit_generator import LLMExploitGenerator
from redteam.agents.planner import Planner
from redteam.bundles import BundleStore, build_bundle
from redteam.config import (
    CHROMA_DIR,
    DATA_DIR,
    EMBEDDING_MODEL,
    RUNS_DIR,
    load_env,
)
from redteam.orchestration.graph import build_graph
from redteam.orchestration.state import RedTeamState
from redteam.target.generator import LLMGenerator
from redteam.target.pipeline import RAGPipeline
from redteam.target.retriever import Retriever

QUERIES_PATH = DATA_DIR / "queries.json"


def _make_run_id(seed: int, query_id: str, batch_ts: str) -> str:
    """Deterministic, filesystem-safe run_id.

    Pattern: ``run_<batch_ts>_seed<seed>_<query_id>``. The batch timestamp
    fixes a single dry-run sweep; multiple sweeps in the same seed land in
    distinct run_ids without colliding. Matches `_SAFE_RUN_ID` in
    `redteam.bundles.store`.
    """
    return f"run_{batch_ts}_seed{seed}_{query_id}"


def _summarise(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-run records into a batch-level summary.

    Reports: counts of verdicts, ASR triple totals (+ ASR-deny totals),
    family distribution, mean generator latency, mean rank-shift, bundles
    written. Useful as the at-a-glance signal that the batch finished in
    the expected shape *before* manually opening any of the JSON files.

    The batch summary is *script-level metadata* — it spans multiple runs
    and therefore does not live inside any single run's folder. Per-run
    summaries (the headline-metric row for one run) are written by
    :class:`BundleStore` into each run's own folder.
    """
    n = len(records)
    if n == 0:
        return {"n_runs": 0}
    family_counts = Counter(r["attack_family"] for r in records)
    verdicts = Counter(r["verdict"] for r in records)
    return {
        "n_runs": n,
        "verdict_counts": dict(verdicts),
        "asr_retrieval_total": sum(int(r["asr_retrieval"]) for r in records),
        "asr_answer_total": sum(int(r["asr_answer"]) for r in records),
        "asr_target_total": sum(int(r["asr_target"]) for r in records),
        "asr_deny_total": sum(int(r["asr_deny"]) for r in records),
        "family_counts": dict(family_counts),
        "mean_latency_ms": sum(r["generator_latency_ms"] for r in records) / n,
        "mean_rank_shift_at_k": sum(r["rank_shift_at_k"] for r in records) / n,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--limit", type=int, default=50, help="Number of queries (default: 50).")
    parser.add_argument("--seed", type=int, default=42, help="Planner + run seed.")
    parser.add_argument(
        "--max-iter",
        type=int,
        default=1,
        help="Max graph iterations per query (default: 1 — Day-8 dry run).",
    )
    parser.add_argument(
        "--with-ragas",
        action="store_true",
        help="Enable the RAGAS triple in evaluate_node. Off by default to "
        "keep the dry run cheap; Day 9 leaves it on.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=RUNS_DIR,
        help="Destination directory for the bundle JSON files (default: data/runs/).",
    )
    args = parser.parse_args()

    load_env()

    if not QUERIES_PATH.exists():
        raise SystemExit(
            f"{QUERIES_PATH} not found. Run `python scripts/04_build_query_set.py` first."
        )
    queries = json.loads(QUERIES_PATH.read_text(encoding="utf-8"))[: args.limit]
    if not queries:
        raise SystemExit("queries.json is empty.")

    retriever = Retriever(persist_dir=CHROMA_DIR, embedding_model_name=EMBEDDING_MODEL)
    if retriever._count() == 0:
        raise SystemExit("Chroma is empty. Run `python scripts/01_build_corpus.py` first.")

    pipeline = RAGPipeline(retriever=retriever, generator=LLMGenerator())
    planner = Planner(epsilon=0.3, seed=args.seed)
    exploit_gen = LLMExploitGenerator()
    app = build_graph(
        pipeline,
        planner=planner,
        exploit_gen=exploit_gen,
        run_ragas=args.with_ragas,
    )

    # One batch folder per script invocation. The batch_id is a UTC
    # timestamp, used as the directory name (`batch_<batch_id>/`) and as
    # the trailing token of every bundle filename inside.
    batch_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    store = BundleStore(args.out_dir, batch_id=batch_id)

    pre_count = retriever._count()
    pre_hash = retriever.get_state_hash()

    records: list[dict[str, Any]] = []
    print(
        f"Dry run: {len(queries)} queries × max_iter={args.max_iter}, "
        f"seed={args.seed}, ragas={'on' if args.with_ragas else 'off'}"
    )
    print(f"Writing bundles to: {args.out_dir.resolve()}\n")

    t_start = time.perf_counter()
    for i, q in enumerate(queries, start=1):
        run_id = _make_run_id(args.seed, q["query_id"], batch_id)

        initial: RedTeamState = {
            "run_id": run_id,
            "seed": args.seed,
            "query": q["query_text"],
            "query_id": q["query_id"],
            "iteration": 0,
            "max_iterations": args.max_iter,
            "history": [],
        }
        try:
            final = app.invoke(initial)
        except Exception as exc:  # don't let one bad query nuke the batch
            print(f"  [{i:02d}/{len(queries)}] {q['query_id']}: GRAPH-ERROR {exc}")
            continue

        bundle = build_bundle(final)
        path = store.write(bundle)
        records.append({
            "run_id": run_id,
            "query_id": q["query_id"],
            "attack_family": final["attack_family"],
            "verdict": final["verdict"],
            "asr_retrieval": final["asr_retrieval"],
            "asr_answer": final["asr_answer"],
            "asr_target": final["asr_target"],
            "asr_deny": final["asr_deny"],
            "rank_shift_at_k": final["rank_shift_at_k"],
            "generator_latency_ms": final["generator_latency_ms"],
        })
        print(
            f"  [{i:02d}/{len(queries)}] {q['query_id']}: "
            f"{final['attack_family']:<17} -> {final['verdict']:<7} "
            f"(ASR-t={int(final['asr_target'])}, rs@k={final['rank_shift_at_k']}) "
            f"-> {path.name}"
        )

    wall_s = time.perf_counter() - t_start

    # Index rollback is the contract: corpus-channel attacks add+remove the
    # payload inside the executor; query-channel attacks don't write at all.
    # If this assert fires, an attack leaked state and Day-9 results would
    # be cross-contaminated between runs.
    post_count = retriever._count()
    post_hash = retriever.get_state_hash()
    rollback_ok = post_count == pre_count and post_hash == pre_hash

    summary = {
        "batch_id": batch_id,
        "args": {
            "limit": args.limit,
            "seed": args.seed,
            "max_iter": args.max_iter,
            "with_ragas": args.with_ragas,
        },
        "wall_seconds": round(wall_s, 2),
        "rollback_ok": rollback_ok,
        "pre_index_state_hash": pre_hash,
        "post_index_state_hash": post_hash,
        "planner_snapshot": planner.snapshot(),
        "runs": records,
        **_summarise(records),
    }
    # Batch-level summary lives inside the same batch folder as its
    # bundles (`batch_<batch_id>/batch_<batch_id>_summary.json`) — every
    # artefact for one batch is co-located.
    summary_path = store.write_batch_summary(summary)

    print("\n" + "=" * 72)
    print(f"Wrote {len(records)} bundles in {wall_s:.1f}s.")
    print(f"Batch folder: {store.batch_dir}")
    print(f"Rollback ok: {rollback_ok}")
    print(f"Verdict counts: {summary.get('verdict_counts')}")
    print(f"ASR-t total: {summary.get('asr_target_total')}/{len(records)}")
    print(f"ASR-deny total: {summary.get('asr_deny_total')}/{len(records)}")
    print(f"Batch summary: {summary_path}")

    if not rollback_ok:
        raise SystemExit(
            "Index state drifted across the dry run — an attack leaked. "
            "Investigate the executor's add/remove cycle before Day 9."
        )


if __name__ == "__main__":
    main()