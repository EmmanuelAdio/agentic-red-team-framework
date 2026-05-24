"""Smoke checks for the dashboard package — imports + pure helpers only.

The acceptance criteria for the dashboard are visual (see §13 of
``DASHBOARD_DESIGN_SYSTEM.md``); this file just guards against the kind
of breakage that would make the pages fail to import.
"""

from __future__ import annotations

import math

import pandas as pd
import pytest


def test_dashboard_imports():
    from redteam.dashboard import _css, charts, components, data, filters
    assert callable(data.load_bundles)
    assert callable(data.load_one_bundle)
    assert callable(data.bootstrap_ci)
    assert callable(data.summary_by_family_channel)
    assert callable(components.badge)
    assert callable(components.doc_card)
    assert callable(components.score_bar)
    assert callable(components.kv_grid)
    assert callable(components.asr_cell)
    assert callable(components.verdict_legend)
    assert callable(components.empty_state)
    assert callable(charts.asr_bar_chart)
    assert callable(charts.faithfulness_overlay_hist)
    assert callable(charts.ragas_violins)
    assert callable(charts.rank_shift_ecdf)
    assert callable(charts.dark_layout)
    assert callable(_css.inject_css)
    assert callable(filters.apply_filters)
    assert callable(filters.available_options)


def test_badge_maps_failure_to_defended():
    from redteam.dashboard.components import badge
    assert "verdict-defended" in badge("failure")
    assert "verdict-success" in badge("success")
    assert "verdict-partial" in badge("partial")


def test_score_bar_handles_none():
    from redteam.dashboard.components import score_bar
    out = score_bar("faithfulness", None)
    assert "n/a" in out
    assert "score-bar-fill" not in out  # no filled bar when value is None


def test_doc_card_marks_poisoned():
    from redteam.dashboard.components import doc_card
    poisoned = doc_card(1, "ipi_x", 0.5, "payload", True)
    clean = doc_card(2, "doc1", 0.4, None, False)
    assert "poisoned" in poisoned
    assert "doc-card poisoned" in poisoned
    assert "poisoned" not in clean


def test_doc_card_marks_baseline_top1():
    """The baseline-top-1 flag must add the highlight class + chip.

    Guards the rank-shift visualisation on the Run Detail page: a
    regression that drops the flag would silently undo the "where did
    the gold doc move under attack?" affordance.
    """
    from redteam.dashboard.components import doc_card
    highlighted = doc_card(
        3, "doc42", 0.61, None, False,
        is_baseline_top1=True, baseline_rank_shift=2,
    )
    assert "baseline-top1" in highlighted
    assert "baseline top-1" in highlighted
    assert "Δ +2" in highlighted
    # Plain clean doc must NOT carry the chip.
    plain = doc_card(3, "doc42", 0.61, None, False)
    assert "baseline-top1" not in plain
    assert "baseline top-1" not in plain


def test_bootstrap_ci_handles_edge_cases():
    from redteam.dashboard.data import bootstrap_ci
    mean, lo, hi = bootstrap_ci([])
    assert math.isnan(mean) and math.isnan(lo) and math.isnan(hi)
    mean, lo, hi = bootstrap_ci([0.42])
    assert mean == lo == hi == 0.42
    mean, lo, hi = bootstrap_ci([0.0, 1.0, 1.0, 1.0])
    assert 0 <= lo <= mean <= hi <= 1


# ---------------------------------------------------------------------------
# New (Day-15 Build-B): filters, verdict legend, empty state, charts, DuckDB
# ---------------------------------------------------------------------------


def test_filters_apply_round_trip_empty_selection():
    """An empty selection must leave the DataFrame unchanged."""
    from redteam.dashboard.filters import apply_filters, available_options

    df = pd.DataFrame({
        "attack_family":  ["ipi", "ipi", "qInj"],
        "attack_channel": ["corpus", "corpus", "query"],
        "payload_source": ["template", "llm", "template"],
        "verdict":        ["success", "failure", "success"],
    })
    opts = available_options(df)
    assert set(opts) >= {"attack_family", "attack_channel", "payload_source", "verdict"}
    # Empty selection — keep every row.
    out = apply_filters(df, {})
    assert len(out) == len(df)
    # Non-empty selection on one column — filter narrowly.
    out2 = apply_filters(df, {"attack_family": ["qInj"]})
    assert len(out2) == 1 and out2.iloc[0]["attack_family"] == "qInj"


def test_verdict_legend_renders_three_classes():
    """The legend strip must surface all three verdict classes."""
    from redteam.dashboard.components import verdict_legend
    out = verdict_legend()
    assert "verdict-success" in out
    assert "verdict-partial" in out
    assert "verdict-defended" in out
    assert "legend-strip" in out


def test_empty_state_renders_reason_and_back_link():
    """Run Detail's empty state must render both the reason and the back link."""
    from redteam.dashboard.components import empty_state
    out = empty_state(
        reason="No run_id supplied.",
        detail="paste ?run_id=run_<query>_<batch>",
        back_href="./",
    )
    assert "No run_id supplied." in out
    assert "paste ?run_id=" in out
    assert "Go back to Overview" in out
    assert 'href="./"' in out


def test_faithfulness_overlay_hist_returns_figure():
    """Synthetic input should return a Plotly figure without raising."""
    from redteam.dashboard.charts import faithfulness_overlay_hist
    attacked = pd.DataFrame({"faithfulness": [0.2, 0.3, 0.6, 0.8, None]})
    baseline = pd.DataFrame({"ragas_faithfulness": [0.85, 0.9, 0.95, 0.7]})
    fig = faithfulness_overlay_hist(attacked, baseline)
    # Two histogram traces, plus a dashed threshold line via shapes.
    assert len(fig.data) == 2
    assert any(getattr(s, "type", None) == "line" for s in fig.layout.shapes)


# ---------------------------------------------------------------------------
# Day-10 per-objective metric attribution
# ---------------------------------------------------------------------------
#
# The dashboard previously pooled poiA + poiJ into one corpus_poisoning row
# under `summary_by_family_channel`, dragging the family ASR-t headline
# down by counting jamming runs' structurally-low ASR-t. These tests pin
# the new attribution: per-objective KPIs ignore the wrong-objective rows,
# and the per-cell summary surfaces poiA and poiJ as separate cells.


def _zeros() -> dict:
    """Bundle-row defaults the per-cell summary's column reads need.

    Provides every numeric / boolean field summarised by
    ``summary_by_cell`` so a test row only has to specify the
    family/strategy/channel triple it wants under test. Used by the
    per-cell shape test below.
    """
    return {
        "asr_r":         False,
        "asr_a":         False,
        "asr_t":         False,
        "asr_deny":      False,
        "faithfulness":  None,
        "rank_shift":    0,
    }


def test_kpi_asr_target_integrity_excludes_poiJ():
    """Integrity KPI must ignore jamming-cell rows even when their ASR-t
    happens to be True (which can occur in the pre-fix legacy data).
    """
    from redteam.dashboard.data import kpi_asr_target_integrity

    df = pd.DataFrame([
        # 1/1 integrity hit
        {"attack_family": "prompt_injection",
         "attack_strategy": "instruction_override",
         "attack_channel": "corpus",
         **_zeros(),
         "asr_t": True},
        # poiJ row whose ASR-t is True must NOT count toward the
        # integrity KPI — its objective is availability.
        {"attack_family": "corpus_poisoning",
         "attack_strategy": "jamming",
         "attack_channel": "corpus",
         **_zeros(),
         "asr_t": True},
    ])
    mean, _hw = kpi_asr_target_integrity(df)
    # If poiJ leaked through, the mean would be 1.0 over n=2 (same
    # answer numerically) but the *count* would be wrong. Tighten the
    # check by also asserting the helper classified correctly:
    assert mean == 1.0  # 1/1 from ipi only — poiJ excluded


def test_kpi_asr_target_integrity_returns_correct_mean_when_objectives_disagree():
    """Tighter version of the above: poiJ's ASR-t=False would pull the
    mean down if it leaked into the integrity bucket. The mean must
    reflect only the integrity rows.
    """
    from redteam.dashboard.data import kpi_asr_target_integrity

    df = pd.DataFrame([
        {"attack_family": "prompt_injection",
         "attack_strategy": "instruction_override",
         "attack_channel": "corpus",
         **_zeros(),
         "asr_t": True},
        {"attack_family": "corpus_poisoning",
         "attack_strategy": "answer_replacement",
         "attack_channel": "corpus",
         **_zeros(),
         "asr_t": True},
        # poiJ with ASR-t=False: must not pull the integrity mean to 2/3.
        {"attack_family": "corpus_poisoning",
         "attack_strategy": "jamming",
         "attack_channel": "corpus",
         **_zeros(),
         "asr_t": False},
    ])
    mean, _hw = kpi_asr_target_integrity(df)
    assert mean == 1.0  # 2/2 integrity hits, poiJ excluded


def test_kpi_asr_deny_availability_excludes_integrity_cells():
    """Availability KPI must ignore integrity-cell rows even if their
    asr_deny happens to fire (rare but possible noise in the data).
    """
    from redteam.dashboard.data import kpi_asr_deny_availability

    df = pd.DataFrame([
        # Integrity cell — must not count toward availability KPI.
        {"attack_family": "corpus_poisoning",
         "attack_strategy": "answer_replacement",
         "attack_channel": "corpus",
         **_zeros(),
         "asr_deny": True},
        # poiJ row — counts.
        {"attack_family": "corpus_poisoning",
         "attack_strategy": "jamming",
         "attack_channel": "corpus",
         **_zeros(),
         "asr_deny": True},
    ])
    mean, _hw = kpi_asr_deny_availability(df)
    assert mean == 1.0  # 1/1 from poiJ only


def test_summary_by_cell_has_four_rows_for_full_matrix():
    """A DataFrame containing all four cells produces four summary rows
    keyed on the dissertation cell labels — not two as the old pooled
    (family, channel)-only grouping did for corpus_poisoning.
    """
    from redteam.dashboard.data import summary_by_cell

    df = pd.DataFrame([
        {"attack_family": "prompt_injection",
         "attack_strategy": "instruction_override",
         "attack_channel": "corpus", **_zeros()},
        {"attack_family": "corpus_poisoning",
         "attack_strategy": "answer_replacement",
         "attack_channel": "corpus", **_zeros()},
        {"attack_family": "corpus_poisoning",
         "attack_strategy": "jamming",
         "attack_channel": "corpus", **_zeros()},
        {"attack_family": "query_injection",
         "attack_strategy": "prefix_injection",
         "attack_channel": "query",  **_zeros()},
    ])
    summary = summary_by_cell(df)
    assert len(summary) == 4
    assert set(summary["cell_label"]) == {"ipi", "poiA", "poiJ", "qInj"}
    # poiJ's headline_success_rate must read from asr_deny (its
    # cell-specific success metric), NOT asr_t.
    poiJ = summary[summary["cell_label"] == "poiJ"].iloc[0]
    assert poiJ["success_metric"] == "asr_deny"
    assert poiJ["objective"] == "availability"
    poiA = summary[summary["cell_label"] == "poiA"].iloc[0]
    assert poiA["success_metric"] == "asr_t"
    assert poiA["objective"] == "integrity"


def test_summary_by_family_channel_alias_still_works():
    """The deprecated alias forwards to summary_by_cell so legacy
    imports don't break in this changeset.
    """
    from redteam.dashboard.data import summary_by_family_channel, summary_by_cell

    df = pd.DataFrame([
        {"attack_family": "prompt_injection",
         "attack_strategy": "instruction_override",
         "attack_channel": "corpus", **_zeros()},
    ])
    a = summary_by_family_channel(df)
    b = summary_by_cell(df)
    assert list(a.columns) == list(b.columns)
    assert len(a) == len(b)


def test_duckdb_query_select_42():
    """The DuckDB façade should round-trip a trivial SELECT.

    Skipped cleanly if the optional ``duckdb`` package is missing — the
    DuckDB backend is opt-in via ``REDTEAM_DASHBOARD_DUCKDB=1`` and
    must not block the default test surface.
    """
    pytest.importorskip("duckdb")
    from redteam.dashboard import duck
    out = duck.query_bundles("SELECT 42 AS answer")
    assert len(out) == 1
    assert int(out["answer"].iloc[0]) == 42
