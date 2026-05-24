"""Data layer — load exploit-bundle JSONs into a flat DataFrame.

The dashboard reads from the two run roots the framework actually writes
to:

* ``data/runs/batch_<id>/`` — Day-8 dry-run bundles
* ``results/runs/batch_<id>/`` — Day-9 full-experiment bundles, plus
  per-batch ``*_summary.json`` rollups and an ``experiment_manifest.json``.

Both roots share the same nested layout enforced by
``redteam.bundles.store.BundleStore``. We discover bundles with a
recursive glob, skip the ``*_summary.json`` rollups (different shape) and
any ``*.tmp`` write-in-flight sidecars, and project each bundle into a
flat row. Manifest-aware aggregation (cell registry, sidecar info) is
out of scope here — Build-B Aggregate calls
``redteam.analysis.loaders.load_experiment`` for that.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

try:  # streamlit is an optional import path so unit tests can run without it
    import streamlit as st  # type: ignore
    _cache_data = st.cache_data
except ImportError:  # pragma: no cover
    def _cache_data(*args, **kwargs):  # type: ignore
        def deco(fn):
            return fn
        return deco

try:
    from redteam.config import EXPERIMENT_RUNS_DIR, RUNS_DIR
except ImportError:  # tolerate running the module standalone
    EXPERIMENT_RUNS_DIR = Path("results/runs")
    RUNS_DIR = Path("data/runs")


# ---------------------------------------------------------------------------
# Cell registry — per-objective attribution mapping
# ---------------------------------------------------------------------------
#
# Maps each (attack_family, attack_strategy) pair to its dissertation cell
# label, channel, adversarial objective, and the success metric the cell
# is judged by. Lifted from `scripts/06_run_experiments.py`'s CELLS
# constant so the dashboard does not import from a script. Keep these
# two definitions in lock-step: when a new cell is added to the experiment
# matrix, add the matching row here.
#
# Why this lives on the analysis side: ASR-t (integrity) and ASR-deny
# (availability) are conceptually orthogonal metrics — pooling them
# across cells with different objectives produces a number that does
# not represent any single attack design. The KPI helpers and the
# per-cell summary table both consult this registry to attribute runs
# to the right objective bucket.
CELL_REGISTRY: dict[tuple[str, str], dict[str, str]] = {
    ("prompt_injection", "instruction_override"):
        {"label": "ipi",  "channel": "corpus", "objective": "integrity",    "success_metric": "asr_t"},
    ("corpus_poisoning", "answer_replacement"):
        {"label": "poiA", "channel": "corpus", "objective": "integrity",    "success_metric": "asr_t"},
    ("corpus_poisoning", "jamming"):
        {"label": "poiJ", "channel": "corpus", "objective": "availability", "success_metric": "asr_deny"},
    ("query_injection",  "prefix_injection"):
        {"label": "qInj", "channel": "query",  "objective": "integrity",    "success_metric": "asr_a"},
}
# Note: `success_metric` values reference the dashboard's flat-row
# column names (`asr_t`, `asr_a`, `asr_deny`) — not the bundle JSON's
# longer names (`asr_target`, `asr_answer`). The mapping convention is
# documented in `_project()` above.


def _classify(df: pd.DataFrame) -> pd.DataFrame:
    """Attach cell_label, objective, success_metric columns to every row.

    Rows whose (attack_family, attack_strategy) pair is not in
    ``CELL_REGISTRY`` are left with NaN/None for the three new columns
    so they are excluded from per-objective KPIs rather than silently
    mis-counted. This guards against future cells being added to the
    experiment driver before the dashboard registry is updated.

    Returns a copy — the input frame is not mutated.
    """
    out = df.copy()
    out["cell_label"]     = pd.NA
    out["objective"]      = pd.NA
    out["success_metric"] = pd.NA
    for (family, strategy), meta in CELL_REGISTRY.items():
        mask = (out["attack_family"] == family) & (out["attack_strategy"] == strategy)
        out.loc[mask, "cell_label"]     = meta["label"]
        out.loc[mask, "objective"]      = meta["objective"]
        out.loc[mask, "success_metric"] = meta["success_metric"]
    return out


# ---------------------------------------------------------------------------
# Path discovery
# ---------------------------------------------------------------------------


def _iter_bundle_paths(roots: Iterable[Path]) -> list[Path]:
    """Recursively find every ``run_*_bundle.json`` under each root.

    Skips ``*_summary.json`` rollups (different schema) and any ``*.tmp``
    sidecars left behind by interrupted writes.
    """
    paths: list[Path] = []
    for root in roots:
        root = Path(root)
        if not root.exists():
            continue
        for p in root.glob("**/run_*_bundle.json"):
            if p.name.endswith(".tmp"):
                continue
            paths.append(p)
    return sorted(paths)


# ---------------------------------------------------------------------------
# Flat-row projection
# ---------------------------------------------------------------------------


def _project(b: dict[str, Any], path: Path) -> dict[str, Any]:
    """Project one bundle dict to the flat row used by the Overview table."""
    attack = b["attack"]
    execution = b["execution"]
    evaluation = b["evaluation"]
    target = b["target_system"]
    return {
        "run_id":          b["run_id"],
        "timestamp":       b["timestamp_utc"],
        "seed":            b["seed"],
        "batch_id":        path.parent.name.removeprefix("batch_"),
        "query":           execution["query"],
        "query_id":        execution["query_id"],
        "attack_family":   attack["family"],
        "attack_strategy": attack["strategy"],
        "attack_channel":  attack.get("attack_channel", "corpus"),
        "payload_source":  attack.get("payload_source", "template"),
        "iteration":       attack.get("iteration", 0),
        "embedding_model": target["embedding_model"],
        "llm_model":       target["llm_model"],
        "asr_r":           bool(evaluation["asr_retrieval"]),
        "asr_a":           bool(evaluation["asr_answer"]),
        "asr_t":           bool(evaluation["asr_target"]),
        "asr_deny":        evaluation.get("asr_deny"),
        "faithfulness":    evaluation.get("ragas_faithfulness"),
        "answer_rel":      evaluation.get("ragas_answer_relevance"),
        "context_rel":     evaluation.get("ragas_context_relevance"),
        "rank_shift":      evaluation["rank_shift_at_k"],
        "verdict":         evaluation["verdict"],
        "latency_ms":      execution["generator_latency_ms"],
        "_path":           str(path),
    }


# ---------------------------------------------------------------------------
# Public loaders
# ---------------------------------------------------------------------------


@_cache_data(ttl=300)
def load_bundles(*roots: Path) -> pd.DataFrame:
    """Load every bundle under ``roots`` into a flat DataFrame.

    Defaults to ``(EXPERIMENT_RUNS_DIR, RUNS_DIR)`` when called with no
    args — Day-9 results take precedence; the Day-8 dry-run tree is the
    fallback.

    Backend dispatch: when ``REDTEAM_DASHBOARD_DUCKDB=1`` is set in the
    environment *and* no custom roots are supplied, the call delegates
    to ``duck.load_bundles_via_duck``. The DuckDB import lives inside
    that branch so a fresh ``git clone`` without DuckDB installed still
    works on the default glob path.
    """
    if not roots:
        try:
            from . import duck  # local import keeps cold-path cheap

            if duck.is_enabled():
                return duck.load_bundles_via_duck()
        except (ImportError, RuntimeError):  # pragma: no cover
            pass  # fall through to the glob path
    roots_t = roots or (EXPERIMENT_RUNS_DIR, RUNS_DIR)
    rows: list[dict[str, Any]] = []
    for path in _iter_bundle_paths(roots_t):
        try:
            with open(path, encoding="utf-8") as fh:
                b = json.load(fh)
            rows.append(_project(b, path))
        except (json.JSONDecodeError, KeyError) as exc:  # pragma: no cover
            # A malformed bundle should not crash the dashboard.
            print(f"[dashboard.data] skipping {path}: {exc}")
            continue
    df = pd.DataFrame(rows)
    if not df.empty:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        # Sort primary on timestamp (newest first); break ties by run_id ascending
        # so the recent-runs table is deterministic across reloads — Day-9
        # bundles share the same minute-granularity timestamp_utc so a single-key
        # sort returned an unstable order between Streamlit reruns.
        df = df.sort_values(
            ["timestamp", "run_id"],
            ascending=[False, True],
        ).reset_index(drop=True)
    return df


@_cache_data(ttl=300)
def load_one_bundle(path: str) -> dict[str, Any]:
    """Load and return one bundle JSON verbatim."""
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def bootstrap_ci(
    values: np.ndarray | list[float] | pd.Series,
    n_resamples: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Bootstrap mean and ``ci``-CI bounds. Returns ``(mean, lo, hi)``.

    Empty input returns ``(nan, nan, nan)``; a single-element input
    returns ``(x, x, x)`` (no whiskers).
    """
    arr = np.asarray(list(values), dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return float("nan"), float("nan"), float("nan")
    if arr.size == 1:
        return float(arr[0]), float(arr[0]), float(arr[0])
    rng = np.random.default_rng(seed)
    means = np.empty(n_resamples)
    n = arr.size
    for i in range(n_resamples):
        means[i] = arr[rng.integers(0, n, size=n)].mean()
    lo, hi = np.quantile(means, [(1 - ci) / 2, 1 - (1 - ci) / 2])
    return float(arr.mean()), float(lo), float(hi)


# ---------------------------------------------------------------------------
# Per-objective KPI helpers (drive the header tiles on the Overview page)
# ---------------------------------------------------------------------------
#
# Each helper filters the input frame to runs whose cell objective matches
# the metric being reported, then computes a bootstrap-CI mean. The
# headline ASR-t and ASR-deny numbers on the Overview page therefore
# never mix objectives — for example a jamming-cell run whose ASR-t
# happens to be True (which can occur in the pre-fix legacy data) is
# never counted toward the integrity headline.


def kpi_asr_target_integrity(df: pd.DataFrame) -> tuple[float, float]:
    """ASR-t headline over integrity-objective runs only.

    Returns ``(mean, ci_half_width)``. Empty input or no integrity rows
    returns ``(nan, 0.0)``.
    """
    if df.empty:
        return float("nan"), 0.0
    classified = _classify(df)
    integrity = classified[classified["objective"] == "integrity"]
    if integrity.empty:
        return float("nan"), 0.0
    vals = integrity["asr_t"].dropna().astype(float).values
    if vals.size == 0:
        return float("nan"), 0.0
    mean, lo, hi = bootstrap_ci(vals)
    half = (hi - lo) / 2.0 if pd.notna(hi) else 0.0
    return mean, half


def kpi_asr_deny_availability(df: pd.DataFrame) -> tuple[float, float]:
    """ASR-deny headline over availability-objective runs only.

    Returns ``(mean, ci_half_width)``. Empty input or no availability
    rows returns ``(nan, 0.0)``.
    """
    if df.empty:
        return float("nan"), 0.0
    classified = _classify(df)
    avail = classified[classified["objective"] == "availability"]
    if avail.empty:
        return float("nan"), 0.0
    vals = avail["asr_deny"].dropna().astype(float).values
    if vals.size == 0:
        return float("nan"), 0.0
    mean, lo, hi = bootstrap_ci(vals)
    half = (hi - lo) / 2.0 if pd.notna(hi) else 0.0
    return mean, half


# ---------------------------------------------------------------------------
# Per-cell summary (replaces the pooled family×channel rollup)
# ---------------------------------------------------------------------------


def summary_by_cell(
    df: pd.DataFrame,
    *,
    n_resamples: int = 500,
    seed: int = 12345,
) -> pd.DataFrame:
    """Per-cell summary keyed on (attack_family, attack_channel, attack_strategy).

    Replaces the prior (attack_family, attack_channel)-only grouping
    which conflated poiA and poiJ into one ``corpus_poisoning × corpus``
    row. Pooling the two cells dragged the family ASR-t headline from
    80% (poiA alone) to 57% (poiA + poiJ pooled) because jamming runs
    structurally have ASR-t = False (their successful output is a
    refusal that contains no marker substring). The per-cell view keeps
    the conceptually-different attacks visibly separate.

    Each row adds three attribution columns sourced from ``CELL_REGISTRY``:

    * ``cell_label`` — short dissertation label (`ipi`, `poiA`, `poiJ`, `qInj`).
    * ``objective`` — `integrity` or `availability`.
    * ``success_metric`` — the column name whose mean is the cell's headline
      success rate (`asr_t`, `asr_a`, or `asr_deny`).

    And one derived metric column:

    * ``headline_success_rate`` — the mean of the cell's own success-metric
      column. The dashboard's bar chart reads this column so each cell's
      bar represents its own success criterion rather than a single
      universal ASR-t.

    The integrity-degraded rate is defined here as
    ``mean(faithfulness < 0.65)`` (the absolute threshold used on the
    Overview metric tile). The chapter's analysis module uses a relative
    drop of ≥ 0.20 vs the clean baseline — different metric, deliberately
    left out of the dashboard's grouping because the dashboard does not
    join against the baseline at this level.
    """
    if df.empty:
        return pd.DataFrame()

    classified = _classify(df)

    def _rate(series: pd.Series) -> tuple[float, float]:
        """Return ``(mean, half_width)`` of a boolean series."""
        vals = series.dropna().astype(float).values
        if vals.size == 0:
            return float("nan"), 0.0
        mean, lo, hi = bootstrap_ci(vals, n_resamples=n_resamples, seed=seed)
        return mean, (hi - lo) / 2.0 if pd.notna(hi) else 0.0

    rows: list[dict[str, Any]] = []
    # Group on the three axes that distinguish cells. Bundles with a
    # missing strategy column should never occur in the Day-9 matrix,
    # but `dropna=False` keeps them visible rather than silently dropped.
    grouped = classified.groupby(
        ["attack_family", "attack_channel", "attack_strategy"],
        dropna=False,
    )
    for (family, channel, strategy), grp in grouped:
        asr_r_mean, asr_r_hw = _rate(grp["asr_r"])
        asr_a_mean, asr_a_hw = _rate(grp["asr_a"])
        asr_t_mean, asr_t_hw = _rate(grp["asr_t"])
        deny_mean,  deny_hw  = _rate(grp["asr_deny"]) if "asr_deny" in grp else (float("nan"), 0.0)

        faith_series = grp["faithfulness"].dropna() if "faithfulness" in grp else pd.Series(dtype=float)
        integ_rate = float((faith_series < 0.65).mean()) if len(faith_series) else float("nan")
        faith_mean = float(faith_series.mean()) if len(faith_series) else float("nan")

        rank_series = grp["rank_shift"].dropna() if "rank_shift" in grp else pd.Series(dtype=float)
        rank_mean = float(rank_series.mean()) if len(rank_series) else float("nan")

        # Cell-level attribution: pick the (family, strategy) registry
        # entry if it exists; cells without an entry render as NaN
        # objective/success_metric and are visible-but-excluded from the
        # per-objective tiles.
        meta = CELL_REGISTRY.get((family, strategy), {})
        cell_label     = meta.get("label")
        objective      = meta.get("objective")
        success_metric = meta.get("success_metric")
        # Compute the cell's own headline success rate by reading its
        # nominated success metric column. Cells without a registered
        # success metric carry NaN here.
        if success_metric and success_metric in grp.columns:
            headline_mean, headline_hw = _rate(grp[success_metric])
        else:
            headline_mean, headline_hw = float("nan"), 0.0

        rows.append({
            "attack_family":          family,
            "attack_channel":         channel,
            "attack_strategy":        strategy,
            "cell_label":             cell_label,
            "objective":              objective,
            "success_metric":         success_metric,
            "n":                      int(len(grp)),
            "asr_r":                  asr_r_mean,
            "asr_r_ci_hw":            asr_r_hw,
            "asr_a":                  asr_a_mean,
            "asr_a_ci_hw":            asr_a_hw,
            "asr_t":                  asr_t_mean,
            "asr_t_ci_hw":            asr_t_hw,
            "asr_deny":               deny_mean,
            "asr_deny_ci_hw":         deny_hw,
            "headline_success_rate":  headline_mean,
            "headline_ci_hw":         headline_hw,
            "faithfulness_mean":      faith_mean,
            "integrity_degraded":     integ_rate,
            "rank_shift_mean":        rank_mean,
        })
    out = pd.DataFrame(rows)
    if not out.empty:
        # Stable display order: prefer the dissertation's cell order
        # (ipi, poiA, poiJ, qInj) for registered cells; unknown cells
        # sink to the bottom alphabetically. Achieved by sorting on a
        # numeric proxy that maps known labels to their registry index.
        order = {meta["label"]: i for i, (_, meta) in enumerate(CELL_REGISTRY.items())}
        out["_sort"] = out["cell_label"].map(lambda x: order.get(x, len(order)))
        out = out.sort_values(
            ["_sort", "attack_family", "attack_channel", "attack_strategy"]
        ).drop(columns="_sort").reset_index(drop=True)
    return out


def summary_by_family_channel(df: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
    """Deprecated alias — forwards to :func:`summary_by_cell`.

    The Day-10 attribution refactor split the prior (family, channel)
    rollup into per-cell rows. The returned frame's column set is a
    strict superset of the pre-refactor frame, so any caller that read
    the old columns (e.g. ``asr_t``, ``asr_deny``, ``faithfulness_mean``)
    keeps working; callers gain access to ``cell_label``, ``objective``,
    ``success_metric``, and ``headline_success_rate`` for the new
    per-objective view. Kept rather than removed so existing imports
    (smoke tests, user scripts) don't break in this changeset.
    """
    return summary_by_cell(df, **kwargs)


__all__ = [
    "load_bundles",
    "load_one_bundle",
    "bootstrap_ci",
    "kpi_asr_target_integrity",
    "kpi_asr_deny_availability",
    "summary_by_cell",
    "summary_by_family_channel",
    "CELL_REGISTRY",
]


if __name__ == "__main__":  # pragma: no cover
    df = load_bundles()
    print("shape:", df.shape)
    print("columns:", list(df.columns))
    print(df.head())
