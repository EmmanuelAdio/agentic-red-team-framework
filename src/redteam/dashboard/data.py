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


def summary_by_family_channel(
    df: pd.DataFrame,
    *,
    n_resamples: int = 500,
    seed: int = 12345,
) -> pd.DataFrame:
    """Per ``(attack_family, attack_channel)`` summary for the Overview table.

    Mirrors the analysis module's ``summary_by_cell`` shape but reads the
    dashboard's flat-row column names directly so we don't have to project
    twice. Each row carries the mean ASR triple + ASR-deny + Faithfulness
    + integrity-degraded rate, plus the 95% bootstrap-CI half-width for
    each rate.

    The integrity-degraded rate is defined here as
    ``mean(faithfulness < 0.65)`` (the absolute threshold used on the
    Overview metric tile). The chapter's analysis module uses a relative
    drop of ≥ 0.20 vs the clean baseline — different metric, deliberately
    left out of the dashboard's grouping because the dashboard does not
    join against the baseline at this level.
    """
    if df.empty:
        return pd.DataFrame()

    def _rate(series: pd.Series) -> tuple[float, float]:
        """Return ``(mean, half_width)`` of a boolean series."""
        vals = series.dropna().astype(float).values
        if vals.size == 0:
            return float("nan"), 0.0
        mean, lo, hi = bootstrap_ci(vals, n_resamples=n_resamples, seed=seed)
        return mean, (hi - lo) / 2.0 if pd.notna(hi) else 0.0

    rows: list[dict[str, Any]] = []
    grouped = df.groupby(["attack_family", "attack_channel"], dropna=False)
    for (family, channel), grp in grouped:
        asr_r_mean, asr_r_hw = _rate(grp["asr_r"])
        asr_a_mean, asr_a_hw = _rate(grp["asr_a"])
        asr_t_mean, asr_t_hw = _rate(grp["asr_t"])
        deny_mean, deny_hw = _rate(grp["asr_deny"]) if "asr_deny" in grp else (float("nan"), 0.0)

        faith_series = grp["faithfulness"].dropna() if "faithfulness" in grp else pd.Series(dtype=float)
        integ_rate = float((faith_series < 0.65).mean()) if len(faith_series) else float("nan")
        faith_mean = float(faith_series.mean()) if len(faith_series) else float("nan")

        rank_series = grp["rank_shift"].dropna() if "rank_shift" in grp else pd.Series(dtype=float)
        rank_mean = float(rank_series.mean()) if len(rank_series) else float("nan")

        rows.append({
            "attack_family":         family,
            "attack_channel":        channel,
            "n":                     int(len(grp)),
            "asr_r":                 asr_r_mean,
            "asr_r_ci_hw":           asr_r_hw,
            "asr_a":                 asr_a_mean,
            "asr_a_ci_hw":           asr_a_hw,
            "asr_t":                 asr_t_mean,
            "asr_t_ci_hw":           asr_t_hw,
            "asr_deny":              deny_mean,
            "asr_deny_ci_hw":        deny_hw,
            "faithfulness_mean":     faith_mean,
            "integrity_degraded":    integ_rate,
            "rank_shift_mean":       rank_mean,
        })
    out = pd.DataFrame(rows)
    # Stable display order: families alphabetical, channels alphabetical.
    if not out.empty:
        out = out.sort_values(
            ["attack_family", "attack_channel"]
        ).reset_index(drop=True)
    return out


__all__ = [
    "load_bundles",
    "load_one_bundle",
    "bootstrap_ci",
    "summary_by_family_channel",
]


if __name__ == "__main__":  # pragma: no cover
    df = load_bundles()
    print("shape:", df.shape)
    print("columns:", list(df.columns))
    print(df.head())
