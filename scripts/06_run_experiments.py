"""Day 9 experiment driver - 600-run forced-Cartesian matrix to ``results/runs/``.

The deliverable for Day 9 (PROJECT_SPEC.md row Day 9) is the headline
experiment matrix: 50 queries x 4 attack cells x 3 seeds = 600 reproducible
exploit bundles, written under ``results/runs/`` (per spec section 13
def-of-done line 421). The four cells cover every attack capability the
framework implements - both delivery channels (corpus / query) crossed with
both objectives (integrity / availability):

==== ===================== ====================== ======== ============= =================
cell  family                strategy               channel  objective    success metric
==== ===================== ====================== ======== ============= =================
ipi   prompt_injection      instruction_override   corpus   integrity    ASR-t
poi-a corpus_poisoning      answer_replacement     corpus   integrity    ASR-t
poi-j corpus_poisoning      jamming                corpus   availability ASR-deny
qry-i query_injection       prefix_injection       query    integrity    ASR-a (asr_r=True)
==== ===================== ====================== ======== ============= =================

Cells 1+2 are the spec's two-family pair (PROJECT_SPEC.md section 2 line 22).
Cells 3+4 are additive coverage - the dissertation's Results chapter
reports the four cells under a 2-channel x 2-objective taxonomy.

Sweep strategy: forced Cartesian. The epsilon-greedy planner picks ONE
family per query stochastically, which cannot produce a per-cell
comparison with statistical power; we drive the matrix with a
:class:`ForcedCellPlanner` instead. The real epsilon-greedy planner is
preserved as a *sidecar log* per seed - run against the same query stream
with the actual ASR-t verdicts fed back via :func:`Planner.update`. Its
selection sequence is recorded in each batch summary for the RQ2
(planner-adaptivity) discussion in Chapter 7. No bundles are produced by
the sidecar; it is a pure behavioural log.

On-disk layout
--------------

The project's :class:`BundleStore` writes bundles as
``run_<query_id>_<batch_id>_bundle.json`` inside ``batch_<batch_id>/``.
If we used one batch per seed (3 batches x 200 bundles each) the four
cells running against the same query_id would collide on the same
filename. Resolution: **one batch per (seed, cell) = 12 batches total**.
The cell label is encoded in the batch_id, so the existing store stays
frozen (per the Day-9 plan's "do not touch ``src/redteam/bundles/``"
constraint) and Day 10's plotting can recover the (seed, cell) pair from
the batch folder name without parsing each bundle.

::

    results/runs/
      batch_seed42_ipi_<ts>/             # 50 bundles
      batch_seed42_poiA_<ts>/            # 50 bundles
      batch_seed42_poiJ_<ts>/            # 50 bundles
      batch_seed42_qInj_<ts>/            # 50 bundles
      batch_seed123_ipi_<ts>/            # 50 bundles
      ... (12 batches, 600 bundles total)
      experiment_manifest.json           # cross-batch index Day 10 reads

CLI
---

::

    # Full job - all defaults (3 seeds x 4 cells x 50 queries, RAGAS on, max-iter 3)
    python scripts/06_run_experiments.py

    # Pre-flight smoke - 8 bundles in <90 seconds
    python scripts/06_run_experiments.py --smoke

    # Single-seed restart (the seed loop is restartable)
    python scripts/06_run_experiments.py --seeds 42

    # Single-cell investigation
    python scripts/06_run_experiments.py --cells poison_jam

Cost
----

Day-8 dry run benchmarked 50 bundles in 14.7s with RAGAS off and
``max-iter=1``. Day 9 has RAGAS on (~4x LLM calls per run for the three
RAGAS scorers) and ``max-iter=3`` with early-exit. Per the plan's
estimate the wall-clock budget is 60-90 minutes for the full 600-run
job; OpenAI's LangChain ``SQLiteCache`` absorbs duplicate calls (the
baseline-retrieval pass for each query is identical across the four
cells, so 75% of baseline-pass LLM calls hit cache). Spend tripwire is
90 minutes per spec line 368.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

# Make `redteam` importable when running the script directly without `pip install -e .`.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from redteam.agents.exploit_generator import LLMExploitGenerator
from redteam.agents.planner import ATTACK_FAMILIES, Planner
from redteam.bundles import BundleStore, build_bundle
from redteam.config import (
    CHROMA_DIR,
    DATA_DIR,
    EMBEDDING_MODEL,
    EXPERIMENT_RUNS_DIR,
    load_env,
)
from redteam.orchestration.graph import ForcedCellPlanner, build_graph
from redteam.orchestration.state import RedTeamState
from redteam.target.generator import LLMGenerator
from redteam.target.pipeline import RAGPipeline
from redteam.target.retriever import Retriever

QUERIES_PATH = DATA_DIR / "queries.json"

# ---------------------------------------------------------------------------
# Cell registry
# ---------------------------------------------------------------------------

# Each cell is a 2-channel x 2-objective slot in the experiment matrix.
# Tuple shape: (label, family, strategy, channel, objective, success_metric).
# `label` is the short identifier used in batch_ids and manifest keys; the
# remaining columns are descriptive (also lifted into the batch summary
# `cell_meta` block so a bundle reader doesn't need to consult this file).
#
# `label` is constrained to the BundleStore's `_SAFE_ID` regex
# `[A-Za-z0-9_\-:.]+` (no spaces, no slashes). Short labels keep the
# resulting batch_id strings under common filesystem path limits on
# Windows when nested under deep workspace paths.
CELLS: list[tuple[str, str, str, str, str, str]] = [
    ("ipi",  "prompt_injection", "instruction_override", "corpus", "integrity",    "asr_target"),
    ("poiA", "corpus_poisoning", "answer_replacement",   "corpus", "integrity",    "asr_target"),
    ("poiJ", "corpus_poisoning", "jamming",              "corpus", "availability", "asr_deny"),
    ("qInj", "query_injection",  "prefix_injection",     "query",  "integrity",    "asr_answer"),
]

# Default seeds: heterogeneous, RNG-uncorrelated, short-string-friendly for
# run_id encoding. The justification paragraph lives in
# docs/EXPERIMENTATION.md section 3.1.
DEFAULT_SEEDS: tuple[int, ...] = (42, 123, 7)


def _make_run_id(seed: int, cell_label: str, query_id: str, batch_ts: str) -> str:
    """Deterministic, filesystem-safe run_id.

    Pattern: ``run_<batch_ts>_seed<seed>_<cell>_<query_id>``. Encodes all
    four matrix axes (timestamp, seed, cell, query) so a bundle's run_id
    is self-describing without needing the surrounding batch context.
    """
    return f"run_{batch_ts}_seed{seed}_{cell_label}_{query_id}"


# ---------------------------------------------------------------------------
# Per-cell summary
# ---------------------------------------------------------------------------


def _summarise_cell(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-run records into a per-cell rollup.

    Reports: counts of verdicts, ASR triple totals (+ ASR-deny totals),
    mean RAGAS triple, mean generator latency, mean rank-shift, bundles
    written. Mean-of-floats computations skip ``None`` entries (RAGAS
    fields land as ``None`` when the wrapper hits an exception or when
    the LLM refused to answer; counting them as 0.0 would skew the
    distribution).
    """
    n = len(records)
    if n == 0:
        return {"n_runs": 0}

    def _mean(values: list[float | None]) -> float | None:
        kept = [v for v in values if v is not None]
        return (sum(kept) / len(kept)) if kept else None

    verdicts = Counter(r["verdict"] for r in records)
    return {
        "n_runs": n,
        "verdict_counts": dict(verdicts),
        "asr_retrieval_total": sum(int(r["asr_retrieval"]) for r in records),
        "asr_answer_total": sum(int(r["asr_answer"]) for r in records),
        "asr_target_total": sum(int(r["asr_target"]) for r in records),
        "asr_deny_total": sum(int(r["asr_deny"]) for r in records),
        "mean_latency_ms": sum(r["generator_latency_ms"] for r in records) / n,
        "mean_rank_shift_at_k": sum(r["rank_shift_at_k"] for r in records) / n,
        "mean_ragas_faithfulness": _mean([r.get("ragas_faithfulness") for r in records]),
        "mean_ragas_answer_relevance": _mean([r.get("ragas_answer_relevance") for r in records]),
        "mean_ragas_context_relevance": _mean([r.get("ragas_context_relevance") for r in records]),
    }


# ---------------------------------------------------------------------------
# Manifest merge (surgical re-runs)
# ---------------------------------------------------------------------------


# Cell labels that participate in the planner sidecar's family-feedback
# table (see :func:`_run_planner_sidecar`'s ``canonical`` filter). A
# ``--cells`` selection that does not intersect this set produces no
# meaningful sidecar signal — the merge path uses this constant to
# decide whether the original sidecar files should be preserved.
_CANONICAL_SIDECAR_CELLS = frozenset({"ipi", "poiA", "qInj"})


def _merge_manifest_in_place(
    existing: dict[str, Any],
    new_seeds: list[dict[str, Any]],
    selected_labels: list[str],
    merge_ts: str,
) -> dict[str, Any]:
    """Merge a partial re-run's seed/cell entries into the prior full manifest.

    Behaviour:

    * For each seed in ``new_seeds``, locate the matching seed entry in
      ``existing["seeds"]`` (by ``seed`` value). Within that entry, each
      cell record in ``existing["seeds"][i]["cells"]`` whose ``label`` is
      in ``selected_labels`` is replaced by the new run's cell record.
      Cells not in ``selected_labels`` are preserved verbatim.

    * Per-seed scalar fields (``wall_seconds``, ``n_runs``,
      ``by_cell_totals``) are recomputed: ``wall_seconds`` is replaced
      with the new partial-run timing for transparency (the original
      seed's wall time is no longer meaningful after replacement);
      ``n_runs`` is recomputed by summing the merged cells' ``n_runs``;
      ``by_cell_totals`` for replaced labels is overwritten from the new
      records (recovered from the new ``cells`` entries' totals).

    * The top-level ``manifest_version`` and ``cell_registry`` blocks are
      preserved from the existing manifest unchanged. ``batch_ts`` is
      preserved as the *original* run's timestamp (so the manifest still
      identifies the headline experiment); ``merged_at`` is added to
      record the partial-rerun's timestamp.

    * If a seed in ``new_seeds`` has no matching entry in ``existing`` (a
      seed that wasn't part of the original run), the new entry is
      appended verbatim.

    Pure function — caller is responsible for the file I/O. Kept here
    rather than inline in ``main()`` so the merge logic is unit-testable
    in isolation.
    """
    merged = dict(existing)
    merged_seeds: list[dict[str, Any]] = list(existing.get("seeds", []))

    by_seed_idx: dict[int, int] = {
        s["seed"]: i for i, s in enumerate(merged_seeds) if "seed" in s
    }

    for new_entry in new_seeds:
        seed = new_entry["seed"]
        if seed not in by_seed_idx:
            merged_seeds.append(new_entry)
            continue
        idx = by_seed_idx[seed]
        old = dict(merged_seeds[idx])
        old_cells = list(old.get("cells", []))
        new_cells_by_label = {c["label"]: c for c in new_entry.get("cells", [])}
        replaced_cells: list[dict[str, Any]] = []
        for c in old_cells:
            if c.get("label") in selected_labels and c["label"] in new_cells_by_label:
                replaced_cells.append(new_cells_by_label[c["label"]])
            else:
                replaced_cells.append(c)
        # Append cells in `new_entry` that didn't exist in the old manifest
        # (defensive — shouldn't happen in normal flow but covers a
        # never-before-run cell label being added in a re-run).
        existing_labels = {c.get("label") for c in old_cells}
        for label, c in new_cells_by_label.items():
            if label not in existing_labels:
                replaced_cells.append(c)
        old["cells"] = replaced_cells
        old["n_runs"] = sum(c.get("n_runs", 0) for c in replaced_cells)
        # Replace the per-cell totals for the cells we re-ran; keep the
        # others untouched.
        old_by_cell = dict(old.get("by_cell_totals", {}))
        new_by_cell = dict(new_entry.get("by_cell_totals", {}))
        for label in selected_labels:
            if label in new_by_cell:
                old_by_cell[label] = new_by_cell[label]
        old["by_cell_totals"] = old_by_cell
        old["wall_seconds"] = new_entry.get("wall_seconds", old.get("wall_seconds"))
        # Sidecar file may or may not have been regenerated. If
        # `new_entry` lacks `sidecar_file`, keep the old one.
        if new_entry.get("sidecar_file"):
            old["sidecar_file"] = new_entry["sidecar_file"]
        merged_seeds[idx] = old

    merged["seeds"] = merged_seeds
    merged["n_bundles_total"] = sum(s.get("n_runs", 0) for s in merged_seeds)
    merged["merged_at"] = merge_ts
    merged.setdefault("merge_history", []).append(
        {
            "ts": merge_ts,
            "cells": list(selected_labels),
            "seeds": [s["seed"] for s in new_seeds],
        }
    )
    return merged


# ---------------------------------------------------------------------------
# Single-cell run
# ---------------------------------------------------------------------------


def _run_one_cell(
    cell: tuple[str, str, str, str, str, str],
    queries: list[dict[str, Any]],
    seed: int,
    max_iter: int,
    with_ragas: bool,
    out_dir: Path,
    batch_ts: str,
    pipeline: RAGPipeline,
    exploit_gen: LLMExploitGenerator,
) -> tuple[Path, dict[str, Any], list[dict[str, Any]]]:
    """Run one cell (seed, family, strategy) over the 50 queries.

    Returns ``(batch_dir, summary, records)``. ``records`` is the
    per-run flat list - the calling seed-loop concatenates these across
    cells to feed the ASR-t lookup the planner sidecar consumes.
    """
    label, family, strategy, channel, objective, success_metric = cell

    batch_id = f"seed{seed}_{label}_{batch_ts}"
    store = BundleStore(out_dir, batch_id=batch_id)

    forced_planner = ForcedCellPlanner(  # type: ignore[arg-type]
        family=family,
        strategy=strategy,
        success_metric=success_metric,
    )
    app = build_graph(
        pipeline,
        planner=forced_planner,
        exploit_gen=exploit_gen,
        run_ragas=with_ragas,
    )

    pre_count = pipeline.retriever._count()
    pre_hash = pipeline.retriever.get_state_hash()

    records: list[dict[str, Any]] = []
    n_q = len(queries)
    t_cell_start = time.perf_counter()

    for i, q in enumerate(queries, start=1):
        run_id = _make_run_id(seed, label, q["query_id"], batch_ts)
        initial: RedTeamState = {
            "run_id": run_id,
            "seed": seed,
            "query": q["query_text"],
            "query_id": q["query_id"],
            "iteration": 0,
            "max_iterations": max_iter,
            "history": [],
        }
        try:
            final = app.invoke(initial)
        except Exception as exc:  # one bad query/cell shouldn't sink the batch
            print(
                f"  [seed={seed} cell={label} {i:02d}/{n_q}] "
                f"q={q['query_id']}: GRAPH-ERROR {type(exc).__name__}: {exc}"
            )
            continue

        bundle = build_bundle(final)
        path = store.write(bundle)

        record = {
            "run_id": run_id,
            "query_id": q["query_id"],
            "cell": label,
            "attack_family": final["attack_family"],
            "attack_strategy": final["attack_strategy"],
            "verdict": final["verdict"],
            "asr_retrieval": bool(final["asr_retrieval"]),
            "asr_answer": bool(final["asr_answer"]),
            "asr_target": bool(final["asr_target"]),
            "asr_deny": bool(final["asr_deny"]),
            "rank_shift_at_k": int(final["rank_shift_at_k"]),
            "generator_latency_ms": float(final["generator_latency_ms"]),
            "ragas_faithfulness": final.get("ragas_faithfulness"),
            "ragas_answer_relevance": final.get("ragas_answer_relevance"),
            "ragas_context_relevance": final.get("ragas_context_relevance"),
            "iterations_used": int(final.get("iteration", 0)),
        }
        records.append(record)

        # Headline success per the cell's own metric, not just ASR-t.
        # For the jamming cell ASR-deny is the success signal; for the
        # query-injection cell ASR-a is (since ASR-r is trivially True).
        headline = bool(record[success_metric])
        print(
            f"  [seed={seed} cell={label} {i:02d}/{n_q}] q={q['query_id']}: "
            f"{final['verdict']:<7} "
            f"(success_metric={success_metric}={int(headline)}, "
            f"ASR-t={int(record['asr_target'])}, "
            f"ASR-deny={int(record['asr_deny'])}, "
            f"rs@k={record['rank_shift_at_k']}, iters={record['iterations_used']}) "
            f"-> {path.name}"
        )

    cell_wall_s = time.perf_counter() - t_cell_start

    post_count = pipeline.retriever._count()
    post_hash = pipeline.retriever.get_state_hash()
    rollback_ok = post_count == pre_count and post_hash == pre_hash

    summary: dict[str, Any] = {
        "batch_id": batch_id,
        "cell_meta": {
            "label": label,
            "family": family,
            "strategy": strategy,
            "channel": channel,
            "objective": objective,
            "success_metric": success_metric,
        },
        "args": {
            "seed": seed,
            "n_queries": n_q,
            "max_iter": max_iter,
            "with_ragas": with_ragas,
        },
        "wall_seconds": round(cell_wall_s, 2),
        "rollback_ok": rollback_ok,
        "pre_index_state_hash": pre_hash,
        "post_index_state_hash": post_hash,
        "runs": records,
        **_summarise_cell(records),
    }
    summary_path = store.write_batch_summary(summary)

    print(
        f"  -> cell={label} done in {cell_wall_s:.1f}s, "
        f"rollback_ok={rollback_ok}, summary={summary_path.name}"
    )
    if not rollback_ok:
        raise SystemExit(
            f"Index state drifted during cell={label} seed={seed} - "
            f"an attack leaked corpus state. Investigate before continuing."
        )

    return store.batch_dir, summary, records


# ---------------------------------------------------------------------------
# Planner sidecar
# ---------------------------------------------------------------------------


def _run_planner_sidecar(
    seed: int,
    queries: list[dict[str, Any]],
    cell_records: list[dict[str, Any]],
) -> dict[str, Any]:
    """Re-run the epsilon-greedy planner over the queries for this seed.

    No bundles produced - this is a pure behavioural log. The planner is
    fed the *actual* ASR-t verdicts from the forced-Cartesian runs via
    :func:`Planner.update` so its success-rate memory reflects ground
    truth as it makes each subsequent ``select()`` call.

    Mapping caveat: :data:`ATTACK_FAMILIES` has three entries
    (``prompt_injection``, ``corpus_poisoning``, ``query_injection``) but
    the matrix has four cells - corpus_poisoning has two strategies. When
    the planner picks ``corpus_poisoning``, we feed it the
    ``answer_replacement`` cell's verdict (the family's default per
    ``_DEFAULT_STRATEGY``); the jamming cell is reported separately via
    its own per-cell summary. This is documented as a design choice in
    docs/EXPERIMENTATION.md section 3.5.
    """
    # Verdict lookup keyed by (query_id, family). Prefer the default
    # strategy for each family when multiple cells exist for it.
    asr_t_by_family: dict[tuple[str, str], bool] = {}
    for r in cell_records:
        family = r["attack_family"]
        # Map the family's "canonical" cell:
        #   prompt_injection -> ipi
        #   corpus_poisoning -> poiA (answer_replacement is the family default)
        #   query_injection  -> qInj
        canonical = {"ipi": True, "poiA": True, "qInj": True}.get(r["cell"], False)
        if not canonical:
            continue
        asr_t_by_family[(r["query_id"], family)] = bool(r["asr_target"])

    planner = Planner(epsilon=0.3, seed=seed)
    selections: list[dict[str, Any]] = []
    for q in queries:
        chosen = planner.select(q["query_text"])
        actual_asr_t = asr_t_by_family.get((q["query_id"], chosen), False)
        planner.update(q["query_text"], chosen, actual_asr_t)
        selections.append({
            "query_id": q["query_id"],
            "chosen_family": chosen,
            "fed_back_asr_t": actual_asr_t,
        })

    return {
        "selections": selections,
        "final_snapshot": planner.snapshot(),
        "n_corpus_poisoning": sum(1 for s in selections if s["chosen_family"] == "corpus_poisoning"),
        "n_prompt_injection": sum(1 for s in selections if s["chosen_family"] == "prompt_injection"),
        "n_query_injection": sum(1 for s in selections if s["chosen_family"] == "query_injection"),
        "convergent_to_family": max(
            ATTACK_FAMILIES, key=lambda f: planner.success_rate(f)
        ),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(DEFAULT_SEEDS),
        help=f"Seed values for the n=3 seed sweep (default: {' '.join(map(str, DEFAULT_SEEDS))}).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=50,
        help="Number of queries per cell (default: 50 - the full query set).",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=3,
        help="Max graph iterations per (seed, cell, query); early-exit on success (default: 3).",
    )
    ragas_group = parser.add_mutually_exclusive_group()
    ragas_group.add_argument(
        "--with-ragas",
        dest="with_ragas",
        action="store_true",
        help="Enable the RAGAS triple in evaluate_node (default for Day-9 full runs).",
    )
    ragas_group.add_argument(
        "--no-ragas",
        dest="with_ragas",
        action="store_false",
        help="Disable RAGAS - cheaper, but loses Faithfulness drop evidence.",
    )
    parser.set_defaults(with_ragas=True)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=EXPERIMENT_RUNS_DIR,
        help="Destination directory for bundles (default: results/runs/).",
    )
    parser.add_argument(
        "--cells",
        type=str,
        default="all",
        help="Comma-separated cell labels to run (default: all). "
        "Available: " + ",".join(c[0] for c in CELLS),
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Pre-flight: --limit 2 --max-iter 1 --no-ragas --seeds <first only>. "
        "Produces 8 bundles in <90s; verifies the matrix wiring before the full job.",
    )
    parser.add_argument(
        "--merge-manifest",
        action="store_true",
        help="When re-running a subset of cells, merge the new batch entries "
        "into the existing experiment_manifest.json instead of overwriting it. "
        "Also suppresses sidecar regeneration when the re-run's --cells do not "
        "include the canonical cells (ipi, poiA, qInj), so existing sidecar "
        "files from the original full run are preserved untouched.",
    )
    args = parser.parse_args()

    if args.smoke:
        args.limit = 2
        args.max_iter = 1
        args.with_ragas = False
        args.seeds = args.seeds[:1]
        if args.out_dir == EXPERIMENT_RUNS_DIR:
            args.out_dir = EXPERIMENT_RUNS_DIR.parent / "runs_smoke"

    selected_labels = (
        [c[0] for c in CELLS]
        if args.cells == "all"
        else [s.strip() for s in args.cells.split(",")]
    )
    selected_cells = [c for c in CELLS if c[0] in selected_labels]
    if not selected_cells:
        raise SystemExit(
            f"No cells matched --cells={args.cells!r}. "
            f"Available: {','.join(c[0] for c in CELLS)}"
        )

    load_env()

    if not QUERIES_PATH.exists():
        raise SystemExit(
            f"{QUERIES_PATH} not found. Run `python scripts/04_build_query_set.py` first."
        )
    queries = json.loads(QUERIES_PATH.read_text(encoding="utf-8"))[: args.limit]
    if not queries:
        raise SystemExit("queries.json is empty.")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    retriever = Retriever(persist_dir=CHROMA_DIR, embedding_model_name=EMBEDDING_MODEL)
    if retriever._count() == 0:
        raise SystemExit("Chroma is empty. Run `python scripts/01_build_corpus.py` first.")

    pipeline = RAGPipeline(retriever=retriever, generator=LLMGenerator())
    # The exploit generator is shared across all (seed, cell) graphs - its
    # internal SQLite cache amortises duplicate prompt calls (e.g. the
    # template path on iteration 0 is deterministic per (query, family,
    # seed) so any retry after a graph-level error hits cache).
    exploit_gen = LLMExploitGenerator()

    batch_ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())

    print(
        f"Day-9 experiment matrix: {len(args.seeds)} seeds x {len(selected_cells)} cells "
        f"x {len(queries)} queries = {len(args.seeds) * len(selected_cells) * len(queries)} "
        f"bundles total"
    )
    print(
        f"  seeds={args.seeds} max_iter={args.max_iter} ragas={'on' if args.with_ragas else 'off'}"
    )
    print(f"  out_dir={args.out_dir.resolve()}\n")

    t_total_start = time.perf_counter()
    manifest_seeds: list[dict[str, Any]] = []

    for seed in args.seeds:
        print(f"\n{'=' * 72}\n=== seed={seed} ({len(selected_cells)} cells)\n{'=' * 72}")
        seed_records: list[dict[str, Any]] = []
        cells_in_seed: list[dict[str, Any]] = []
        t_seed_start = time.perf_counter()

        for cell in selected_cells:
            label = cell[0]
            print(f"\n--- seed={seed} cell={label} ({cell[1]}/{cell[2]}) ---")
            batch_dir, cell_summary, cell_records = _run_one_cell(
                cell=cell,
                queries=queries,
                seed=seed,
                max_iter=args.max_iter,
                with_ragas=args.with_ragas,
                out_dir=args.out_dir,
                batch_ts=batch_ts,
                pipeline=pipeline,
                exploit_gen=exploit_gen,
            )
            seed_records.extend(cell_records)
            cells_in_seed.append({
                "label": label,
                "batch_dir": str(batch_dir.relative_to(args.out_dir)),
                "n_runs": cell_summary["n_runs"],
                "asr_target_total": cell_summary.get("asr_target_total", 0),
                "asr_deny_total": cell_summary.get("asr_deny_total", 0),
                "wall_seconds": cell_summary["wall_seconds"],
                "rollback_ok": cell_summary["rollback_ok"],
            })

        # Sidecar runs after all cells for this seed have produced their
        # ASR-t verdicts so the planner.update() calls reflect ground truth.
        # Skip when --merge-manifest is set AND the re-run doesn't touch
        # any canonical sidecar cell: a poiJ-only re-run produces no
        # signal for the ε-greedy sidecar (the family-feedback table maps
        # only ipi/poiA/qInj — `_run_planner_sidecar`'s canonical filter),
        # so the existing sidecar files from the original full run are
        # the authoritative ones to keep.
        selected_label_set = {c[0] for c in selected_cells}
        skip_sidecar = (
            args.merge_manifest
            and not (selected_label_set & _CANONICAL_SIDECAR_CELLS)
        )
        if skip_sidecar:
            print(
                f"\n--- seed={seed} planner sidecar SKIPPED "
                f"(--merge-manifest with no canonical cells in --cells={','.join(sorted(selected_label_set))}) ---"
            )
            sidecar_filename: str | None = None
        else:
            print(f"\n--- seed={seed} planner sidecar ---")
            sidecar = _run_planner_sidecar(seed=seed, queries=queries, cell_records=seed_records)
            # Sidecar gets its own tiny on-disk artefact alongside the cell
            # batches so Day 10 doesn't need to re-derive it.
            sidecar_path = args.out_dir / f"sidecar_seed{seed}_{batch_ts}.json"
            sidecar_path.write_text(
                json.dumps(
                    {"seed": seed, "batch_ts": batch_ts, **sidecar},
                    indent=2,
                ),
                encoding="utf-8",
            )
            print(
                f"  -> planner converged toward family={sidecar['convergent_to_family']}, "
                f"sidecar={sidecar_path.name}"
            )
            sidecar_filename = sidecar_path.name

        seed_wall_s = time.perf_counter() - t_seed_start
        # Per-seed roll-up across the cells.
        by_cell_totals: dict[str, dict[str, int]] = defaultdict(
            lambda: {"asr_t": 0, "asr_deny": 0, "n": 0}
        )
        for r in seed_records:
            by_cell_totals[r["cell"]]["asr_t"] += int(r["asr_target"])
            by_cell_totals[r["cell"]]["asr_deny"] += int(r["asr_deny"])
            by_cell_totals[r["cell"]]["n"] += 1
        seed_entry: dict[str, Any] = {
            "seed": seed,
            "wall_seconds": round(seed_wall_s, 2),
            "n_runs": len(seed_records),
            "cells": cells_in_seed,
            "by_cell_totals": dict(by_cell_totals),
        }
        # Only record `sidecar_file` when we actually wrote one; the
        # manifest merge keeps the existing seed entry's `sidecar_file`
        # untouched when this field is absent from the new entry.
        if sidecar_filename is not None:
            seed_entry["sidecar_file"] = sidecar_filename
        manifest_seeds.append(seed_entry)

    t_total_s = time.perf_counter() - t_total_start

    # Cross-seed manifest - the entry point Day 10 plotting uses to find
    # every batch in this experiment. The shape is intentionally flat
    # so a single `json.loads` reveals the full layout.
    manifest = {
        "manifest_version": "1.0",
        "batch_ts": batch_ts,
        "args": {
            "seeds": args.seeds,
            "cells": [c[0] for c in selected_cells],
            "n_queries": len(queries),
            "max_iter": args.max_iter,
            "with_ragas": args.with_ragas,
        },
        "wall_seconds_total": round(t_total_s, 2),
        "n_bundles_total": sum(s["n_runs"] for s in manifest_seeds),
        "seeds": manifest_seeds,
        "cell_registry": [
            {
                "label": c[0], "family": c[1], "strategy": c[2],
                "channel": c[3], "objective": c[4], "success_metric": c[5],
            }
            for c in CELLS
        ],
    }
    manifest_path = args.out_dir / "experiment_manifest.json"
    if args.merge_manifest and manifest_path.exists():
        # Surgical re-run: fold this invocation's seed entries into the
        # prior full manifest rather than clobbering it. Preserves the
        # untouched cells' batch references, the original `manifest_version`
        # and `cell_registry`, and the sidecar pointers for cells we
        # didn't re-run.
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        merged = _merge_manifest_in_place(
            existing=existing,
            new_seeds=manifest_seeds,
            selected_labels=[c[0] for c in selected_cells],
            merge_ts=batch_ts,
        )
        manifest_path.write_text(json.dumps(merged, indent=2), encoding="utf-8")
        # Recompute the print-out totals from the merged document so the
        # final summary line reflects the union, not just this re-run.
        manifest = merged
    else:
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"\n{'=' * 72}")
    print(
        f"DONE: {manifest['n_bundles_total']} bundles across "
        f"{len(args.seeds)} seeds x {len(selected_cells)} cells "
        f"in {t_total_s:.1f}s ({t_total_s / 60:.1f} min)."
    )
    print(f"Manifest: {manifest_path}")
    for s in manifest_seeds:
        per_cell = ", ".join(
            f"{lbl}=ASR-t {v['asr_t']}/{v['n']} ASR-deny {v['asr_deny']}/{v['n']}"
            for lbl, v in s["by_cell_totals"].items()
        )
        print(f"  seed={s['seed']}: {per_cell}")


if __name__ == "__main__":
    main()
