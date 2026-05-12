"""Unit tests for the Day-10 analysis pipeline.

Every test runs against synthetic JSON fixtures written into a ``tmp_path``;
no Chroma, no API, no real bundle reads. The aim is to pin the *contracts*
of every public function in :mod:`redteam.analysis`:

- :func:`load_experiment` parses manifest + summaries + sidecars into the
  right DataFrame shape, and refuses to proceed on rollback violations.
- :func:`load_bundles_for_k_curve` selectively reads the corpus-channel
  cells and produces one row per (run, retrieved-doc).
- :func:`bootstrap_mean_ci` is seed-deterministic and bounded.
- :func:`asr_r_at_k` is monotonically non-decreasing in k.
- :func:`cohen_h` has the correct sign convention.
- :func:`paired_differences_vs_ipi` omits the IPI-vs-IPI row.
- :func:`make_all_plots` writes 7 PDFs + 7 PNGs to disk, with non-zero
  file sizes, under matplotlib's headless ``Agg`` backend.
- :func:`validate_clean_baseline` rejects a too-short baseline.

The fixtures live in tiny helper functions so each test can build the
minimal payload it needs without sharing global state.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib
import pandas as pd
import pytest

# Force a headless backend before any plotting code imports matplotlib.pyplot.
matplotlib.use("Agg")

from redteam.analysis import (
    asr_r_at_k,
    bootstrap_mean_ci,
    bootstrap_proportion_ci,
    build_summary_tables,
    cohen_h,
    load_bundles_for_k_curve,
    load_clean_baseline,
    load_experiment,
    make_all_plots,
    paired_differences_vs_ipi,
    summary_by_cell,
    validate_clean_baseline,
)


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _write_json(path: Path, payload: Any) -> None:
    """Pretty-write a JSON payload, creating parent dirs as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _make_run(
    query_id: str,
    *,
    cell: str,
    asr_retrieval: bool,
    asr_answer: bool,
    asr_target: bool,
    asr_deny: bool = False,
    rank_shift: int = 0,
    faithfulness: float = 0.5,
    answer_relevance: float = 0.5,
    context_relevance: float = 0.5,
    iterations_used: int = 1,
    latency_ms: float = 10.0,
) -> dict[str, Any]:
    """Build one minimal per-run record matching the Day-9 summary schema."""
    return {
        "run_id":                    f"run_{cell}_{query_id}",
        "query_id":                  query_id,
        "cell":                      cell,
        "attack_family":             {"ipi": "prompt_injection",
                                       "poiA": "corpus_poisoning",
                                       "poiJ": "corpus_poisoning",
                                       "qInj": "query_injection"}[cell],
        "attack_strategy":           "synthetic",
        "verdict":                   "success" if asr_target or asr_deny else "failure",
        "asr_retrieval":             asr_retrieval,
        "asr_answer":                asr_answer,
        "asr_target":                asr_target,
        "asr_deny":                  asr_deny,
        "rank_shift_at_k":           rank_shift,
        "generator_latency_ms":      latency_ms,
        "ragas_faithfulness":        faithfulness,
        "ragas_answer_relevance":    answer_relevance,
        "ragas_context_relevance":   context_relevance,
        "iterations_used":           iterations_used,
    }


# Cell metadata used by both the summaries and the manifest's cell_registry.
_CELLS = [
    {"label": "ipi",  "family": "prompt_injection", "strategy": "instruction_override",
     "channel": "corpus", "objective": "integrity",    "success_metric": "asr_target"},
    {"label": "poiA", "family": "corpus_poisoning", "strategy": "answer_replacement",
     "channel": "corpus", "objective": "integrity",    "success_metric": "asr_target"},
    {"label": "poiJ", "family": "corpus_poisoning", "strategy": "jamming",
     "channel": "corpus", "objective": "availability","success_metric": "asr_deny"},
    {"label": "qInj", "family": "query_injection",  "strategy": "prefix_injection",
     "channel": "query",  "objective": "integrity",    "success_metric": "asr_answer"},
]


def _make_summary(
    *,
    seed: int,
    cell_meta: dict[str, str],
    runs: list[dict[str, Any]],
    rollback_ok: bool = True,
) -> dict[str, Any]:
    """Build a batch summary JSON payload (matches the Day-9 contract)."""
    label = cell_meta["label"]
    return {
        "batch_id":                f"seed{seed}_{label}_synth",
        "cell_meta":               cell_meta,
        "args":                    {"seed": seed, "n_queries": len(runs)},
        "wall_seconds":            1.0,
        "rollback_ok":             rollback_ok,
        "pre_index_state_hash":    "sha256:pre",
        "post_index_state_hash":   "sha256:post",
        "runs":                    runs,
        "n_runs":                  len(runs),
        "asr_target_total":        sum(1 for r in runs if r["asr_target"]),
        "asr_deny_total":          sum(1 for r in runs if r["asr_deny"]),
    }


def _make_bundle(
    *,
    seed: int,
    cell: str,
    query_id: str,
    retrieved_docs: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build a bundle JSON payload with only the fields F3 reads."""
    return {
        "bundle_version": "1.0",
        "run_id":         f"run_{cell}_{seed}_{query_id}",
        "seed":           seed,
        "execution": {
            "query":           "synthetic question",
            "query_id":        query_id,
            "retrieved_docs":  retrieved_docs,
            "generator_output": "synthetic answer",
            "generator_latency_ms": 1.0,
        },
    }


def _write_experiment(
    root: Path,
    *,
    seeds: tuple[int, ...] = (42,),
    n_queries: int = 3,
    rollback_ok: bool = True,
    include_bundles: bool = False,
) -> Path:
    """Lay out a synthetic experiment matrix on disk under ``root``."""
    root.mkdir(parents=True, exist_ok=True)
    manifest_seeds: list[dict[str, Any]] = []

    for seed in seeds:
        cells_in_seed: list[dict[str, Any]] = []
        for cell_meta in _CELLS:
            label = cell_meta["label"]
            batch_dir_name = f"batch_seed{seed}_{label}_synth"
            batch_dir = root / batch_dir_name
            batch_dir.mkdir(parents=True, exist_ok=True)

            # Build deterministic synthetic runs - ipi succeeds always,
            # poiA succeeds 2/3, poiJ never on integrity but always on
            # deny, qInj succeeds 2/3 on answer-injection.
            runs: list[dict[str, Any]] = []
            for i in range(n_queries):
                q = f"q{i}"
                if label == "ipi":
                    r = _make_run(q, cell="ipi",  asr_retrieval=True,  asr_answer=True,  asr_target=True,
                                  rank_shift=0, faithfulness=0.4 + 0.01 * i)
                elif label == "poiA":
                    succ = i != 0
                    r = _make_run(q, cell="poiA", asr_retrieval=True,  asr_answer=succ,  asr_target=succ,
                                  rank_shift=1 if succ else 0, faithfulness=0.6 + 0.02 * i)
                elif label == "poiJ":
                    deny = True
                    r = _make_run(q, cell="poiJ", asr_retrieval=True,  asr_answer=False, asr_target=False,
                                  asr_deny=deny, rank_shift=2, faithfulness=0.3)
                else:  # qInj
                    succ = i != 0
                    r = _make_run(q, cell="qInj", asr_retrieval=True,  asr_answer=succ,  asr_target=succ,
                                  rank_shift=3 if succ else 1, faithfulness=0.2 + 0.01 * i)
                runs.append(r)

                if include_bundles:
                    # F3 needs one poisoned doc somewhere in the top-5;
                    # for corpus-channel cells we put it at rank 2 on
                    # successful runs, rank 4 on failures (so ASR-r@1
                    # is < ASR-r@2 - the monotonicity test relies on
                    # this).
                    poisoned_rank = 2 if r.get("asr_target") or r.get("asr_deny") else 4
                    retrieved = []
                    for rank in range(1, 6):
                        retrieved.append({
                            "doc_id":      f"doc_{label}_{q}_{rank}",
                            "rank":        rank,
                            "score":       1.0 - 0.1 * rank,
                            "is_poisoned": (rank == poisoned_rank) and label != "qInj",
                            "content":     "synthetic context",
                        })
                    bundle = _make_bundle(
                        seed=seed, cell=label, query_id=q,
                        retrieved_docs=retrieved,
                    )
                    _write_json(
                        batch_dir / f"run_{q}_seed{seed}_{label}_synth_bundle.json",
                        bundle,
                    )

            summary = _make_summary(
                seed=seed, cell_meta=cell_meta, runs=runs, rollback_ok=rollback_ok,
            )
            _write_json(batch_dir / f"{batch_dir_name}_summary.json", summary)
            cells_in_seed.append({
                "label":            label,
                "batch_dir":        batch_dir_name,
                "n_runs":           len(runs),
                "asr_target_total": summary["asr_target_total"],
                "asr_deny_total":   summary["asr_deny_total"],
                "wall_seconds":     1.0,
                "rollback_ok":      rollback_ok,
            })

        # Sidecar: 50-query-style selection log with simple round-robin.
        sidecar_path = root / f"sidecar_seed{seed}_synth.json"
        families = ["prompt_injection", "corpus_poisoning", "query_injection"]
        selections = [
            {"query_id": f"q{i}", "chosen_family": families[i % 3],
             "fed_back_asr_t": (i % 2 == 0)}
            for i in range(n_queries)
        ]
        _write_json(sidecar_path, {
            "seed":                 seed,
            "batch_ts":             "synth",
            "selections":           selections,
            "final_snapshot":       {},
            "n_prompt_injection":   sum(1 for s in selections if s["chosen_family"] == "prompt_injection"),
            "n_corpus_poisoning":   sum(1 for s in selections if s["chosen_family"] == "corpus_poisoning"),
            "n_query_injection":    sum(1 for s in selections if s["chosen_family"] == "query_injection"),
            "convergent_to_family": "prompt_injection",
        })
        manifest_seeds.append({
            "seed":         seed,
            "wall_seconds": 1.0,
            "n_runs":       n_queries * len(_CELLS),
            "cells":        cells_in_seed,
            "sidecar_file": sidecar_path.name,
        })

    manifest = {
        "manifest_version":  "1.0",
        "batch_ts":          "synth",
        "args":              {"seeds": list(seeds), "cells": [c["label"] for c in _CELLS],
                              "n_queries": n_queries, "max_iter": 1, "with_ragas": True},
        "wall_seconds_total": 1.0,
        "n_bundles_total":    n_queries * len(_CELLS) * len(seeds),
        "seeds":              manifest_seeds,
        "cell_registry":      _CELLS,
    }
    _write_json(root / "experiment_manifest.json", manifest)
    return root


def _write_baseline(path: Path, *, n: int = 50) -> Path:
    """Write a complete clean-baseline JSON with n rows."""
    rows = [
        {
            "query_id":               f"q{i}",
            "query_text":             "synthetic",
            "gold_doc_ids":           [f"doc_q{i}_1"],
            "retrieved_doc_ids":      [f"doc_q{i}_{j}" for j in range(1, 6)],
            "retrieved_top_scores":   [0.9, 0.8, 0.7, 0.6, 0.5],
            "asr_retrieval_clean":    True,
            "top1_is_gold":           True,
            "generator_output":       "synthetic",
            "generator_latency_ms":   10.0,
            "ragas_faithfulness":     0.95,
            "ragas_answer_relevance": 0.85,
            "ragas_context_relevance":0.80,
            "ragas_notes":            None,
        }
        for i in range(n)
    ]
    payload = {
        "schema_version":              "1.0",
        "kind":                        "clean_baseline",
        "batch_ts":                    "synth",
        "n_queries_input":             n,
        "n_queries_completed":         n,
        "with_ragas":                  True,
        "wall_seconds":                1.0,
        "asr_retrieval_clean_total":   n,
        "top1_is_gold_total":          n,
        "mean_latency_ms":             10.0,
        "mean_ragas_faithfulness":     0.95,
        "mean_ragas_answer_relevance": 0.85,
        "mean_ragas_context_relevance":0.80,
        "rows":                        rows,
    }
    _write_json(path, payload)
    return path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_load_experiment_reads_manifest_summaries_and_sidecars(tmp_path: Path) -> None:
    root = _write_experiment(tmp_path / "runs", seeds=(42,), n_queries=3)
    data = load_experiment(root)

    # 4 cells x 1 seed x 3 queries = 12 runs.
    assert len(data.runs) == 12
    assert set(data.runs["cell"]) == {"ipi", "poiA", "poiJ", "qInj"}
    # Cell metadata should be joined onto every run row.
    assert set(data.runs["channel"]) == {"corpus", "query"}
    # The headline_success column should follow the cell's success_metric.
    poij = data.runs[data.runs["cell"] == "poiJ"]
    # poiJ's success_metric is asr_deny - our fixture sets asr_deny=True
    # for every poiJ run, so headline_success should be all True.
    assert poij["headline_success"].all()
    # Sidecar rows: 1 seed x 3 selections = 3 rows.
    assert len(data.sidecars) == 3
    # Running rate columns are populated.
    assert any(c.startswith("running_rate_") for c in data.sidecars.columns)


def test_load_experiment_rejects_failed_rollback(tmp_path: Path) -> None:
    root = _write_experiment(tmp_path / "runs", seeds=(42,), n_queries=2,
                              rollback_ok=False)
    with pytest.raises(ValueError, match="rollback"):
        load_experiment(root)


def test_load_bundles_for_k_curve_reads_only_corpus_cells(tmp_path: Path) -> None:
    root = _write_experiment(tmp_path / "runs", seeds=(42,), n_queries=2,
                              include_bundles=True)
    bundles = load_bundles_for_k_curve(root)
    assert not bundles.empty
    # Corpus-channel cells only: ipi, poiA, poiJ.
    assert set(bundles["cell"]) == {"ipi", "poiA", "poiJ"}
    # Each (cell, query) has 5 ranks recorded.
    by_run = bundles.groupby(["seed", "cell", "query_id"]).size()
    assert (by_run == 5).all()


def test_bootstrap_mean_ci_is_seed_deterministic() -> None:
    out_a = bootstrap_mean_ci([0, 1, 1, 1], n_resamples=200, seed=7)
    out_b = bootstrap_mean_ci([0, 1, 1, 1], n_resamples=200, seed=7)
    assert out_a == out_b
    assert out_a["n"] == 4
    assert out_a["mean"] == pytest.approx(0.75)


def test_bootstrap_proportion_ci_lies_within_zero_one() -> None:
    # 8 successes out of 10.
    successes = [True] * 8 + [False] * 2
    ci = bootstrap_proportion_ci(successes, n_resamples=500, seed=5)
    assert 0.0 <= ci["ci_low"] <= ci["mean"] <= ci["ci_high"] <= 1.0
    assert ci["mean"] == pytest.approx(0.8)


def test_asr_r_at_k_monotone_non_decreasing(tmp_path: Path) -> None:
    root = _write_experiment(tmp_path / "runs", seeds=(42,), n_queries=4,
                              include_bundles=True)
    bundles = load_bundles_for_k_curve(root)
    asr_per_k = {k: asr_r_at_k(bundles, k) for k in range(1, 6)}
    # For each (seed, cell) the asr_r value must be non-decreasing in k.
    for seed in (42,):
        for cell in ("ipi", "poiA", "poiJ"):
            series = [
                float(asr_per_k[k].loc[
                    (asr_per_k[k]["seed"] == seed) & (asr_per_k[k]["cell"] == cell),
                    "asr_r",
                ].iloc[0])
                for k in range(1, 6)
            ]
            for a, b in zip(series, series[1:]):
                assert a <= b + 1e-9, f"{cell} broke monotonicity: {series}"


def test_cohen_h_signs() -> None:
    # cohen_h(0.9, 0.5) > 0 because p1 > p2.
    assert cohen_h(0.9, 0.5) > 0
    # cohen_h(0.5, 0.9) < 0 by symmetry.
    assert cohen_h(0.5, 0.9) < 0
    # cohen_h(0.5, 0.5) is exactly 0.
    assert cohen_h(0.5, 0.5) == pytest.approx(0.0, abs=1e-9)


def test_paired_differences_vs_ipi_drops_self_comparison(tmp_path: Path) -> None:
    root = _write_experiment(tmp_path / "runs", seeds=(42,), n_queries=3)
    data = load_experiment(root)
    table = paired_differences_vs_ipi(data.runs)
    # Should have exactly the three non-ipi rows.
    assert set(table["cell"]) == {"poiA", "poiJ", "qInj"}
    assert "ipi" not in set(table["cell"])
    # cell_success_rate must be in [0,1].
    assert ((table["cell_success_rate"] >= 0.0) & (table["cell_success_rate"] <= 1.0)).all()


def test_validate_clean_baseline_fails_on_smoke_baseline(tmp_path: Path) -> None:
    short_path = _write_baseline(tmp_path / "baseline_short.json", n=3)
    baseline = load_clean_baseline(short_path)
    with pytest.raises(ValueError, match="incomplete"):
        validate_clean_baseline(baseline, expected_n=50)


def test_make_all_plots_writes_seven_pdfs(tmp_path: Path) -> None:
    root = _write_experiment(
        tmp_path / "runs", seeds=(42, 123, 7), n_queries=6, include_bundles=True,
    )
    baseline_path = _write_baseline(tmp_path / "baseline.json", n=50)
    # The synthetic baseline carries query_ids q0..q49; the synthetic
    # experiment matrix uses q0..q5. Both pair fine on the q0..q5 overlap.

    data = load_experiment(root)
    baseline = load_clean_baseline(baseline_path)
    # Build the summary table from the same data the plots consume.
    summary = summary_by_cell(data.runs, n_resamples=200, seed=12345)

    out_dir = tmp_path / "figures"
    paths = make_all_plots(
        data, baseline, summary, out_dir,
        n_resamples=200, bootstrap_seed=12345,
    )
    assert len(paths) == 7
    for p in paths:
        assert p.exists()
        assert p.stat().st_size > 0
        png = p.with_suffix(".png")
        assert png.exists()
        assert png.stat().st_size > 0


def test_build_summary_tables_on_synthetic_data(tmp_path: Path) -> None:
    root = _write_experiment(tmp_path / "runs", seeds=(42,), n_queries=3)
    baseline_path = _write_baseline(tmp_path / "baseline.json", n=50)
    data = load_experiment(root)
    baseline = load_clean_baseline(baseline_path)
    tables = build_summary_tables(data, baseline, n_resamples=200, bootstrap_seed=12345)

    assert set(tables) == {
        "summary_by_cell", "ragas_by_cell", "paired_differences_vs_ipi",
        "baseline_summary",
    }
    s = tables["summary_by_cell"]
    assert set(s["cell"]) == {"ipi", "poiA", "poiJ", "qInj"}
    # IPI's success rate is 1.0 by construction in the fixture.
    ipi_rate = float(s.loc[s["cell"] == "ipi", "headline_success_rate"].iloc[0])
    assert ipi_rate == pytest.approx(1.0)
    # PoiJ's headline_success_rate uses asr_deny, which is True for every fixture row.
    poij_rate = float(s.loc[s["cell"] == "poiJ", "headline_success_rate"].iloc[0])
    assert poij_rate == pytest.approx(1.0)
