"""Load Day-9 / Day-10 result artefacts into analysis-ready DataFrames.

The Chapter-6 figure stack needs four artefact families on disk:

1. ``results/runs/experiment_manifest.json`` plus 12 batch summaries
   (the headline 600-run matrix) - read by :func:`load_experiment` into
   one flat per-run DataFrame.
2. ``results/runs/sidecar_seed*.json`` - the epsilon-greedy planner
   sidecar logs; read alongside the runs and enriched with the cumulative
   success-rate-per-family that Figure F7 needs (the raw file only
   records per-step ``chosen_family`` + ``fed_back_asr_t``).
3. ``results/baseline/baseline_latest.json`` - the Day-10 clean baseline
   (50 queries with RAGAS triple), read by :func:`load_clean_baseline`.
4. The 600 individual bundle JSON files under ``results/runs/batch_*/``.
   These are not needed for the per-cell tables (which read from batch
   summaries' ``runs[]`` block) but they *are* needed for Figure F3
   (ASR-r vs k) which needs ``execution.retrieved_docs[].rank`` and
   ``.is_poisoned`` to recompute ASR-r at k=1..5. Read lazily by
   :func:`load_bundles_for_k_curve` so the cheap loader path stays cheap.

Design choices
--------------

- **Flat DataFrames everywhere.** The Chapter-6 figures all aggregate
  per-cell or per-(cell, seed); a tidy long-form DataFrame is the right
  shape for ``groupby`` + ``seaborn``/matplotlib. Nested dicts make the
  plot code branch.
- **Frozen container** (:class:`ExperimentData`) instead of returning a
  tuple - so a caller can pass ``data`` around without rebuilding it and
  IDEs can navigate the fields.
- **Fail fast on rollback violations.** A non-rollback batch would mean
  attack state leaked into the corpus; downstream metrics are no longer
  comparable. :func:`load_experiment` raises if any batch summary has
  ``rollback_ok=False`` rather than silently producing skewed tables.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from redteam.config import EXPERIMENT_RUNS_DIR, RESULTS_DIR

DEFAULT_BASELINE_PATH: Path = RESULTS_DIR / "baseline" / "baseline_latest.json"

# Number of queries the clean baseline must cover for the chapter's
# clean-vs-attacked figures to be paired correctly. Comes from the
# Day-9 experiment matrix's per-cell n_queries.
EXPECTED_CLEAN_QUERIES: int = 50

# Cells whose attack delivery channel is the corpus - the only cells for
# which an ASR-r-vs-k dose-response curve (Figure F3) is meaningful. The
# query-injection cell does not poison the corpus so its ASR-r is True
# by construction at every k.
CORPUS_CHANNEL_CELLS: tuple[str, ...] = ("ipi", "poiA", "poiJ")


@dataclass(frozen=True)
class ExperimentData:
    """Container for the attacked-condition artefacts.

    Attributes
    ----------
    results_dir:
        Root directory containing ``experiment_manifest.json``.
    manifest:
        The manifest dict, parsed verbatim.
    runs:
        One row per (seed, cell, query) - 600 rows for the full matrix.
        Columns: every field in the per-run record (verdict, asr_*,
        rank_shift_at_k, ragas_*, iterations_used, latency_ms) joined
        with cell metadata (family, strategy, channel, objective,
        success_metric) and a derived ``headline_success`` boolean using
        the cell's own ``success_metric``.
    batch_summaries:
        One row per (seed, cell) batch - 12 rows for the full matrix.
        Carries n_runs, rollback_ok, wall_seconds, and total counts.
    sidecars:
        One row per (seed, query_index) planner selection - 150 rows for
        50 queries x 3 seeds. Enriched with a ``running_success_rate``
        column per chosen family (cumulative mean of fed_back_asr_t for
        each family across selection_order). This is the column Figure
        F7's left panel reads.
    cell_registry:
        Cell-metadata table lifted from the manifest's ``cell_registry``
        block.
    """

    results_dir: Path
    manifest: dict[str, Any]
    runs: pd.DataFrame
    batch_summaries: pd.DataFrame
    sidecars: pd.DataFrame
    cell_registry: pd.DataFrame
    # Lazily-populated cache for the F3 bundle reader. Kept here so that a
    # caller that loads the bundles once doesn't pay for them twice.
    _bundle_cache: dict[str, pd.DataFrame] = field(default_factory=dict, repr=False)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _read_json(path: Path) -> Any:
    """Read + json-parse a file; raise FileNotFoundError with the path."""
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _summary_path(batch_dir: Path) -> Path:
    """Path to the per-batch summary file inside a batch folder."""
    return batch_dir / f"{batch_dir.name}_summary.json"


def _registry(manifest: dict[str, Any]) -> pd.DataFrame:
    """Cell-metadata table; raises if the manifest is missing the block."""
    registry = pd.DataFrame(manifest.get("cell_registry", []))
    if registry.empty:
        raise ValueError("experiment_manifest.json has no cell_registry entries")
    return registry


def _enrich_run_row(
    row: dict[str, Any],
    *,
    seed: int,
    cell: str,
    batch_id: str | None,
    batch_dir: str,
    cell_meta: dict[str, Any],
) -> dict[str, Any]:
    """Join one per-run record with seed + cell metadata + headline_success."""
    success_metric = str(cell_meta.get("success_metric", "asr_target"))
    enriched = dict(row)
    enriched.update({
        "seed":            seed,
        "cell":            cell,
        "batch_id":        batch_id,
        "batch_dir":       batch_dir,
        "family":          cell_meta.get("family"),
        "strategy":        cell_meta.get("strategy"),
        "channel":         cell_meta.get("channel"),
        "objective":       cell_meta.get("objective"),
        "success_metric":  success_metric,
        # The cell's own headline-success signal. For integrity cells this
        # is ASR-t; for the jamming cell it is ASR-deny; for query
        # injection it is ASR-a (since ASR-r is trivially True for a
        # query-channel attack).
        "headline_success": bool(row.get(success_metric, False)),
    })
    return enriched


def _read_sidecar(path: Path, *, seed: int) -> list[dict[str, Any]]:
    """Read one sidecar JSON; flatten its ``selections[]`` into rows."""
    payload = _read_json(path)
    rows: list[dict[str, Any]] = []
    for i, selection in enumerate(payload.get("selections", []), start=1):
        rows.append({
            "seed":                  seed,
            "batch_ts":              payload.get("batch_ts"),
            "selection_order":       i,
            "query_id":              selection.get("query_id"),
            "chosen_family":         selection.get("chosen_family"),
            "fed_back_asr_t":        bool(selection.get("fed_back_asr_t")),
            "convergent_to_family":  payload.get("convergent_to_family"),
        })
    return rows


def _add_running_success_rate(sidecars: pd.DataFrame) -> pd.DataFrame:
    """Compute the per-family cumulative success rate vs selection order.

    For each (seed, chosen_family) trajectory through the 50-query stream,
    compute the running mean of ``fed_back_asr_t`` at every step where
    that family was selected. We then forward-fill within each seed so
    every (seed, selection_order) row carries the *most recent* running
    rate for every family - the shape Figure F7's left panel needs.

    Why forward-fill within seed: between two consecutive selections of
    the same family the planner has not observed any new data for that
    family, so its memory of that family's success-rate is unchanged. The
    plot should reflect that flat segment rather than dropping to NaN.
    """
    if sidecars.empty:
        return sidecars.assign(
            running_success_rate_chosen=pd.Series(dtype=float),
        )

    sidecars = sidecars.sort_values(["seed", "selection_order"]).copy()

    # Running success rate for the family that was *just* chosen on this
    # step. Computed per (seed, chosen_family) group as the expanding mean.
    sidecars["running_success_rate_chosen"] = (
        sidecars.groupby(["seed", "chosen_family"])["fed_back_asr_t"]
        .expanding()
        .mean()
        .reset_index(level=[0, 1], drop=True)
        .astype(float)
    )

    # Wide-form per-family running rates, forward-filled within seed so
    # every step carries the most-recent estimate for every family.
    family_rates = (
        sidecars.pivot_table(
            index=["seed", "selection_order"],
            columns="chosen_family",
            values="running_success_rate_chosen",
            aggfunc="last",
        )
        .groupby(level="seed")
        .ffill()
    )
    family_rates.columns = [f"running_rate_{c}" for c in family_rates.columns]
    sidecars = sidecars.merge(
        family_rates.reset_index(),
        on=["seed", "selection_order"],
        how="left",
    )
    return sidecars


# ---------------------------------------------------------------------------
# Public loaders
# ---------------------------------------------------------------------------


def load_experiment(results_dir: Path = EXPERIMENT_RUNS_DIR) -> ExperimentData:
    """Load the Day-9 manifest, batch summaries, run rows, and sidecars.

    Parameters
    ----------
    results_dir:
        Directory containing ``experiment_manifest.json`` and the
        ``batch_*/`` folders. Defaults to the project's
        ``results/runs/``.

    Returns
    -------
    :class:`ExperimentData` with four DataFrames populated. The bundle
    cache is empty - :func:`load_bundles_for_k_curve` populates it on
    demand.

    Raises
    ------
    FileNotFoundError
        If the manifest or any expected batch summary is missing.
    ValueError
        If the manifest's claimed total ``n_bundles_total`` does not match
        the number of run rows actually loaded, or if any batch reports
        ``rollback_ok=False`` (the corpus index drifted - downstream
        metrics would be incomparable).
    """
    results_dir = Path(results_dir)
    manifest = _read_json(results_dir / "experiment_manifest.json")
    registry = _registry(manifest)
    registry_by_label = registry.set_index("label").to_dict("index")

    run_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    sidecar_rows: list[dict[str, Any]] = []

    for seed_entry in manifest.get("seeds", []):
        seed = int(seed_entry["seed"])
        for cell_entry in seed_entry.get("cells", []):
            label = cell_entry["label"]
            batch_dir = results_dir / cell_entry["batch_dir"]
            summary = _read_json(_summary_path(batch_dir))
            cell_meta = summary.get("cell_meta", registry_by_label.get(label, {}))

            summary_rows.append({
                "seed":              seed,
                "cell":              label,
                "batch_id":          summary.get("batch_id"),
                "batch_dir":         str(batch_dir),
                "n_runs":            int(summary.get("n_runs", 0)),
                "rollback_ok":       bool(summary.get("rollback_ok")),
                "wall_seconds":      summary.get("wall_seconds"),
                "asr_target_total":  summary.get("asr_target_total"),
                "asr_deny_total":    summary.get("asr_deny_total"),
                **{f"cell_{k}": v for k, v in cell_meta.items()},
            })

            for row in summary.get("runs", []):
                run_rows.append(
                    _enrich_run_row(
                        row,
                        seed=seed,
                        cell=label,
                        batch_id=summary.get("batch_id"),
                        batch_dir=str(batch_dir),
                        cell_meta=cell_meta,
                    )
                )

        # Sidecar file is named in the manifest; tolerate it missing in
        # synthetic fixtures used by tests (the manifest carries the
        # ``sidecar_file`` key only on real runs).
        sidecar_file = seed_entry.get("sidecar_file")
        if sidecar_file:
            sidecar_rows.extend(
                _read_sidecar(results_dir / sidecar_file, seed=seed)
            )

    runs = pd.DataFrame(run_rows)
    summaries = pd.DataFrame(summary_rows)
    sidecars = _add_running_success_rate(pd.DataFrame(sidecar_rows))

    expected = int(manifest.get("n_bundles_total", 0))
    if expected and len(runs) != expected:
        raise ValueError(
            f"Manifest claims n_bundles_total={expected} but loaded {len(runs)} runs"
        )
    if not summaries.empty and not summaries["rollback_ok"].all():
        bad = summaries.loc[~summaries["rollback_ok"], ["seed", "cell"]].to_dict("records")
        raise ValueError(
            f"Index rollback failed in batch summaries: {bad}. "
            f"An attack leaked corpus state - downstream metrics are incomparable."
        )

    return ExperimentData(
        results_dir=results_dir,
        manifest=manifest,
        runs=runs,
        batch_summaries=summaries,
        sidecars=sidecars,
        cell_registry=registry,
    )


def load_clean_baseline(path: Path = DEFAULT_BASELINE_PATH) -> pd.DataFrame:
    """Load ``baseline_latest.json`` rows into a DataFrame.

    The full JSON payload is stashed in ``df.attrs["payload"]`` so the
    aggregate fields (``mean_ragas_*``, ``asr_retrieval_clean_total``,
    etc.) stay accessible to downstream code without a second JSON load.
    """
    payload = _read_json(Path(path))
    rows = pd.DataFrame(payload.get("rows", []))
    rows.attrs["payload"] = payload
    rows.attrs["path"] = str(path)
    return rows


def validate_clean_baseline(
    baseline: pd.DataFrame,
    expected_n: int = EXPECTED_CLEAN_QUERIES,
) -> None:
    """Fail fast unless the clean baseline is complete enough for Chapter 6.

    The clean-vs-attacked Faithfulness figure (F5) and the per-cell
    Faithfulness-drop table require one paired clean row per attacked
    row. If the baseline is short (e.g. someone ran ``--limit 3`` while
    testing) downstream stats silently degrade to whatever queries were
    covered. We refuse to proceed in that case rather than emitting
    quietly-skewed CIs.
    """
    payload = baseline.attrs.get("payload", {})
    completed = int(payload.get("n_queries_completed", len(baseline)))
    if completed < expected_n or len(baseline) < expected_n:
        raise ValueError(
            f"Clean baseline is incomplete: completed={completed}/{expected_n}, "
            f"rows_loaded={len(baseline)}. "
            "Run `python scripts/07_run_clean_baseline.py` first."
        )
    if "ragas_faithfulness" in baseline and baseline["ragas_faithfulness"].isna().all():
        raise ValueError(
            "Clean baseline has no RAGAS faithfulness scores. "
            "Re-run with --with-ragas."
        )


def load_bundles_for_k_curve(
    results_dir: Path = EXPERIMENT_RUNS_DIR,
    cells: tuple[str, ...] = CORPUS_CHANNEL_CELLS,
) -> pd.DataFrame:
    """Read individual bundles and extract per-rank retrieval rows for F3.

    For each bundle in each (seed, cell) batch matching ``cells``, this
    walks ``execution.retrieved_docs[]`` and emits one row per retrieved
    document. The returned DataFrame is the input for
    :func:`redteam.analysis.stats.asr_r_at_k`, which evaluates ASR-r at
    each k by checking whether any ``is_poisoned=True`` row has
    ``rank <= k``.

    Why this is its own loader rather than part of :func:`load_experiment`:
    the per-cell tables and 6/7 of the figures need only the per-run
    summary rows; pulling 600 bundles off disk to compute means would be
    wasted I/O. F3 alone needs the per-rank detail, so we read it on
    demand.

    Returns
    -------
    DataFrame with columns:
        ``seed``, ``cell``, ``query_id``, ``run_id``, ``doc_id``,
        ``rank``, ``score``, ``is_poisoned``.
    """
    results_dir = Path(results_dir)
    manifest = _read_json(results_dir / "experiment_manifest.json")

    rows: list[dict[str, Any]] = []
    for seed_entry in manifest.get("seeds", []):
        seed = int(seed_entry["seed"])
        for cell_entry in seed_entry.get("cells", []):
            cell = cell_entry["label"]
            if cell not in cells:
                continue
            batch_dir = results_dir / cell_entry["batch_dir"]
            for bundle_path in sorted(batch_dir.glob("run_*_bundle.json")):
                bundle = _read_json(bundle_path)
                execution = bundle.get("execution", {})
                run_id = bundle.get("run_id")
                query_id = execution.get("query_id")
                for d in execution.get("retrieved_docs", []):
                    rows.append({
                        "seed":         seed,
                        "cell":         cell,
                        "query_id":     query_id,
                        "run_id":       run_id,
                        "doc_id":       d.get("doc_id"),
                        "rank":         int(d.get("rank", -1)),
                        "score":        float(d.get("score", 0.0)),
                        "is_poisoned":  bool(d.get("is_poisoned", False)),
                    })

    return pd.DataFrame(rows)
