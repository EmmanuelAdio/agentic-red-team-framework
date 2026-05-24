"""Day-10 headless plotter - regenerate every Chapter-6 table and figure.

Per spec section 13 line 418 ("python scripts/04_make_plots.py
regenerates every figure in the dissertation"), this script is the
single command-line entry point a stranger can run on a freshly-cloned
repo to rebuild the entire ``results/figures/`` + ``results/tables/``
output from the persisted Day-9 / Day-10 artefacts. The notebook
:mod:`notebooks.03_results_analysis` is the demo surface; this script is
the headless reproducibility surface.

Workflow
--------

1. Load the 600-run matrix via :func:`load_experiment` (reads
   ``experiment_manifest.json`` + 12 batch summaries + 3 sidecars).
2. Load the clean baseline via :func:`load_clean_baseline` and refuse
   to proceed if it is short.
3. Build the four Chapter-6 tables via :func:`build_summary_tables`
   (per-cell, per-cell RAGAS, paired-vs-IPI, baseline-summary).
4. Write the tables as CSV + a flat ``results/summary.json``.
5. Lazily read the per-rank bundle DataFrame for F3 then call
   :func:`make_all_plots` to produce 8 PDFs + 8 PNGs (F1-F8; F8 is the
   Day-10 poiJ outcome decomposition added alongside the early-exit fix).

CLI
---

::

    python scripts/08_make_plots.py                    # all defaults
    python scripts/08_make_plots.py --runs-dir results/runs --baseline-path results/baseline/baseline_latest.json
    python scripts/08_make_plots.py --n-resamples 2000 --bootstrap-seed 7

The script is byte-deterministic at fixed ``--bootstrap-seed`` /
``--n-resamples``; two runs produce byte-identical CSVs and visually
identical PDFs - a property the dissertation's def-of-done relies on.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Force headless matplotlib backend before anything imports pyplot. The
# notebook overrides this back to inline via %matplotlib inline.
import matplotlib
matplotlib.use("Agg")

# Make `redteam` importable when running the script directly without `pip install -e .`.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from redteam.analysis import (
    build_summary_tables,
    load_bundles_for_k_curve,
    load_clean_baseline,
    load_experiment,
    make_all_plots,
    validate_clean_baseline,
    write_summary_json,
    write_tables,
)
from redteam.config import EXPERIMENT_RUNS_DIR, RESULTS_DIR


DEFAULT_BASELINE_PATH = RESULTS_DIR / "baseline" / "baseline_latest.json"
DEFAULT_FIGURES_DIR   = RESULTS_DIR / "figures"
DEFAULT_TABLES_DIR    = RESULTS_DIR / "tables"
DEFAULT_SUMMARY_JSON  = RESULTS_DIR / "summary.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--runs-dir", type=Path, default=EXPERIMENT_RUNS_DIR,
        help="Directory containing experiment_manifest.json (default: results/runs/).",
    )
    parser.add_argument(
        "--baseline-path", type=Path, default=DEFAULT_BASELINE_PATH,
        help="Path to baseline_latest.json (default: results/baseline/baseline_latest.json).",
    )
    parser.add_argument(
        "--figures-dir", type=Path, default=DEFAULT_FIGURES_DIR,
        help="Output directory for PDF + PNG figures (default: results/figures/).",
    )
    parser.add_argument(
        "--tables-dir", type=Path, default=DEFAULT_TABLES_DIR,
        help="Output directory for CSV tables (default: results/tables/).",
    )
    parser.add_argument(
        "--summary-path", type=Path, default=DEFAULT_SUMMARY_JSON,
        help="Path for the compact machine-readable summary.json (default: results/summary.json).",
    )
    parser.add_argument(
        "--bootstrap-seed", type=int, default=12345,
        help="RNG seed for the bootstrap CIs - pin for byte-reproducible CSVs.",
    )
    parser.add_argument(
        "--n-resamples", type=int, default=1000,
        help="Bootstrap resample count (default: 1000).",
    )
    parser.add_argument(
        "--expected-clean-n", type=int, default=50,
        help="Minimum number of clean-baseline queries required to proceed.",
    )
    args = parser.parse_args()

    t_start = time.perf_counter()
    print(f"[plotter] runs_dir={args.runs_dir}")
    print(f"[plotter] baseline_path={args.baseline_path}")
    print()

    # 1. Load attacked-condition results.
    print("[plotter] loading 600-run experiment matrix ...")
    data = load_experiment(args.runs_dir)
    print(f"          loaded {len(data.runs)} run rows, "
          f"{len(data.batch_summaries)} batch summaries, "
          f"{len(data.sidecars)} sidecar selections.")

    # 2. Load + validate clean baseline.
    print("[plotter] loading clean baseline ...")
    baseline = load_clean_baseline(args.baseline_path)
    validate_clean_baseline(baseline, expected_n=args.expected_clean_n)
    print(f"          baseline rows = {len(baseline)} "
          f"(expected >= {args.expected_clean_n}).")

    # 3. Build the Chapter-6 tables.
    print("[plotter] building summary tables (bootstrap CIs) ...")
    tables = build_summary_tables(
        data, baseline,
        n_resamples=args.n_resamples,
        bootstrap_seed=args.bootstrap_seed,
    )

    # 4. Persist tables + flat summary JSON.
    print(f"[plotter] writing CSV tables to {args.tables_dir} ...")
    write_tables(tables, args.tables_dir)
    print(f"[plotter] writing summary JSON to {args.summary_path} ...")
    write_summary_json(tables, args.summary_path)

    # 5. Per-rank bundle reader for F3, then all 8 figures (F1-F8).
    print("[plotter] reading per-rank bundle rows for F3 (ASR-r vs k) ...")
    bundles = load_bundles_for_k_curve(args.runs_dir)
    print(f"          loaded {len(bundles)} per-rank rows across "
          f"{bundles['cell'].nunique() if not bundles.empty else 0} corpus-channel cells.")

    print(f"[plotter] rendering 8 figures to {args.figures_dir} ...")
    paths = make_all_plots(
        data, baseline,
        summary_table=tables["summary_by_cell"],
        out_dir=args.figures_dir,
        n_resamples=args.n_resamples,
        bootstrap_seed=args.bootstrap_seed,
        bundles=bundles,
    )

    wall = time.perf_counter() - t_start
    print()
    print("=" * 72)
    print(f"[plotter] DONE in {wall:.1f}s")
    print(f"  tables: {args.tables_dir} ({len(tables)} CSVs)")
    print(f"  summary: {args.summary_path}")
    print(f"  figures: {args.figures_dir} ({len(paths)} PDFs + {len(paths)} PNGs)")
    for p in paths:
        print(f"    {p.name}")


if __name__ == "__main__":
    main()
