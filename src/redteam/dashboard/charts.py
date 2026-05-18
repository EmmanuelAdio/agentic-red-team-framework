"""Plotly chart factories used by the dashboard.

Build-A scope: ``asr_bar_chart``. Build-B layered on top:
``faithfulness_overlay_hist`` (Overview), ``ragas_violins`` and
``rank_shift_ecdf`` (Aggregate page). Every factory accepts a
*theme-agnostic* layout; ``dark_layout(fig)`` is the single hook that
re-tints a figure for dark mode (called by the pages, not the
factories, so the helpers stay theme-pure).
"""

from __future__ import annotations

import os

import pandas as pd
import plotly.graph_objects as go

from .data import bootstrap_ci


def current_theme() -> str:
    """Read the dashboard theme from the environment.

    The value is ``"dark"`` if ``REDTEAM_DASHBOARD_THEME`` (case
    insensitive) is set to ``dark``; otherwise ``"light"``. Read at the
    top of every page; Streamlit reruns *do not* re-import os.environ,
    so this is a per-server-start setting (documented in
    ``dashboard/README.md``).
    """
    return "dark" if os.environ.get(
        "REDTEAM_DASHBOARD_THEME", "light"
    ).lower() == "dark" else "light"


def dark_layout(fig: go.Figure) -> go.Figure:
    """Re-tint a Plotly figure for the dark theme. No-op if light.

    Surface, gridlines, axis label colours, and the legend font are
    swapped to dark-friendly values. The trace fills (verdict reds,
    palette greens, etc.) are deliberately *not* touched — the
    semantic palette must stay consistent across themes.
    """
    fig.update_layout(
        paper_bgcolor="#161614",
        plot_bgcolor="#1F1F1B",
        xaxis=dict(
            gridcolor="#3A3A35",
            zerolinecolor="#3A3A35",
            tickfont=dict(color="#B4B2A9"),
            title=dict(font=dict(color="#B4B2A9")),
        ),
        yaxis=dict(
            gridcolor="#3A3A35",
            zerolinecolor="#3A3A35",
            tickfont=dict(color="#B4B2A9"),
            title=dict(font=dict(color="#B4B2A9")),
        ),
        legend=dict(font=dict(color="#B4B2A9")),
        font=dict(color="#F1EFE8"),
    )
    return fig


# Verdict-aligned palette (light variant of the alarm reds/ambers used
# in the verdict chips). Keeping these literal instead of CSS-var means
# the chart matches whether or not the inline stylesheet has loaded.
_PALETTE = {
    "asr_t":  "#E24B4A",  # red — end-to-end attack success
    "asr_a":  "#EF9F27",  # amber — answer-only success
    "asr_r":  "#EF9F27",
    "asr_deny": "#0C447C",
}


def asr_bar_chart(
    df: pd.DataFrame,
    metric: str = "asr_t",
    *,
    min_group_size: int = 2,
) -> go.Figure:
    """Horizontal bars of mean ``metric`` by ``(attack_family, attack_channel)``.

    Whiskers come from :func:`bootstrap_ci`. Groups smaller than
    ``min_group_size`` render the mean with no whisker.
    """
    if df.empty or metric not in df.columns:
        fig = go.Figure()
        fig.update_layout(
            margin=dict(l=20, r=20, t=10, b=20),
            height=240,
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            annotations=[dict(
                x=0.5, y=0.5, xref="paper", yref="paper",
                text="No runs available",
                showarrow=False,
                font=dict(family="JetBrains Mono, monospace", size=12, color="#888780"),
            )],
        )
        return fig

    rows: list[dict] = []
    grouped = df.groupby(["attack_family", "attack_channel"], dropna=False)
    for (fam, chan), grp in grouped:
        vals = grp[metric].astype(float).values
        mean, lo, hi = bootstrap_ci(vals)
        rows.append({
            "label": f"{fam} · {chan}",
            "mean": mean,
            "lo": lo,
            "hi": hi,
            "n": len(grp),
            "show_ci": len(grp) >= min_group_size,
        })
    rows.sort(key=lambda r: (r["mean"] if pd.notna(r["mean"]) else -1))

    labels = [r["label"] for r in rows]
    means = [r["mean"] for r in rows]
    err_plus = [(r["hi"] - r["mean"]) if r["show_ci"] else 0 for r in rows]
    err_minus = [(r["mean"] - r["lo"]) if r["show_ci"] else 0 for r in rows]
    n_text = [f"n={r['n']}" for r in rows]

    fig = go.Figure(
        data=go.Bar(
            x=means,
            y=labels,
            orientation="h",
            marker=dict(color=_PALETTE.get(metric, "#E24B4A")),
            error_x=dict(
                type="data",
                symmetric=False,
                array=err_plus,
                arrayminus=err_minus,
                color="#5F5E5A",
                thickness=1,
                width=4,
            ),
            text=n_text,
            textposition="outside",
            textfont=dict(family="JetBrains Mono, monospace", size=10, color="#5F5E5A"),
            hovertemplate=(
                "<b>%{y}</b><br>"
                f"mean {metric}: " + "%{x:.2f}<br>"
                "<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        margin=dict(l=180, r=40, t=10, b=24),
        height=max(180, 28 * len(rows) + 80),
        plot_bgcolor="#FAFAF7",
        paper_bgcolor="#FAFAF7",
        xaxis=dict(
            range=[0, 1.05],
            tickformat=".0%",
            gridcolor="#E5E4DC",
            zerolinecolor="#E5E4DC",
            tickfont=dict(family="JetBrains Mono, monospace", size=10, color="#5F5E5A"),
        ),
        yaxis=dict(
            tickfont=dict(family="JetBrains Mono, monospace", size=11, color="#1F1F1B"),
            automargin=True,
        ),
        showlegend=False,
        bargap=0.4,
    )
    return fig


def faithfulness_overlay_hist(
    attacked_df: pd.DataFrame,
    baseline_df: pd.DataFrame,
    *,
    bins: int = 20,
    integrity_threshold: float = 0.65,
) -> go.Figure:
    """Overlay histogram of clean vs attacked RAGAS Faithfulness.

    The clean distribution comes from ``baseline_df['ragas_faithfulness']``
    (50 NQ queries scored with the same RAGAS scorer as the attacked
    runs); the attacked distribution comes from
    ``attacked_df['faithfulness']`` (the per-bundle projection). A
    vertical dashed line at ``integrity_threshold`` marks the
    "integrity-degraded" cutoff from PROJECT_SPEC §6.2.
    """
    clean_vals: list[float] = []
    if not baseline_df.empty and "ragas_faithfulness" in baseline_df.columns:
        clean_vals = (
            baseline_df["ragas_faithfulness"]
            .dropna()
            .astype(float)
            .tolist()
        )
    attacked_vals: list[float] = []
    if not attacked_df.empty and "faithfulness" in attacked_df.columns:
        attacked_vals = (
            attacked_df["faithfulness"]
            .dropna()
            .astype(float)
            .tolist()
        )

    fig = go.Figure()
    if clean_vals:
        fig.add_trace(go.Histogram(
            x=clean_vals,
            name=f"clean (n={len(clean_vals)})",
            nbinsx=bins,
            marker=dict(color="#97C459"),
            opacity=0.55,
            hovertemplate=(
                "clean<br>faith: %{x:.2f}<br>"
                "count: %{y}<extra></extra>"
            ),
        ))
    if attacked_vals:
        fig.add_trace(go.Histogram(
            x=attacked_vals,
            name=f"attacked (n={len(attacked_vals)})",
            nbinsx=bins,
            marker=dict(color="#E24B4A"),
            opacity=0.55,
            hovertemplate=(
                "attacked<br>faith: %{x:.2f}<br>"
                "count: %{y}<extra></extra>"
            ),
        ))

    fig.update_layout(
        barmode="overlay",
        margin=dict(l=40, r=20, t=10, b=24),
        height=280,
        plot_bgcolor="#FAFAF7",
        paper_bgcolor="#FAFAF7",
        xaxis=dict(
            range=[0, 1.02],
            tickformat=".1f",
            gridcolor="#E5E4DC",
            zerolinecolor="#E5E4DC",
            tickfont=dict(family="JetBrains Mono, monospace", size=10, color="#5F5E5A"),
            title=dict(text="RAGAS Faithfulness", font=dict(size=11, color="#5F5E5A")),
        ),
        yaxis=dict(
            gridcolor="#E5E4DC",
            tickfont=dict(family="JetBrains Mono, monospace", size=10, color="#5F5E5A"),
            title=dict(text="bundle count", font=dict(size=11, color="#5F5E5A")),
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.0,
            xanchor="right",
            x=1.0,
            font=dict(family="JetBrains Mono, monospace", size=10, color="#5F5E5A"),
        ),
        shapes=[dict(
            type="line",
            x0=integrity_threshold,
            x1=integrity_threshold,
            y0=0,
            y1=1,
            yref="paper",
            line=dict(color="#888780", width=1, dash="dash"),
        )],
        annotations=[dict(
            x=integrity_threshold,
            y=1.02,
            xref="x",
            yref="paper",
            text=f"integrity-degraded @ {integrity_threshold:.2f}",
            showarrow=False,
            font=dict(family="JetBrains Mono, monospace", size=9, color="#888780"),
        )],
    )
    return fig


# Cell colours for the Aggregate page — sourced from the analysis
# palette so dashboard violins/ECDFs match matplotlib F1–F7 exactly.
# Local copy keeps the import surface tight (no matplotlib pull-in at
# module top); the analysis module is only loaded when the Aggregate
# page is opened.
_CELL_COLOURS: dict[str, str] = {
    "ipi":   "#E69F00",   # Okabe-Ito orange
    "poiA":  "#56B4E9",   # sky blue
    "poiJ":  "#D55E00",   # vermillion
    "qInj":  "#009E73",   # bluish green
    "clean": "#999999",   # neutral grey
}
_CELL_ORDER: tuple[str, ...] = ("ipi", "poiA", "poiJ", "qInj")


def ragas_violins(merged: pd.DataFrame) -> go.Figure:
    """Three-panel split violins of RAGAS triple, clean vs attacked, per cell.

    ``merged`` must carry ``cell`` + the three attacked-condition
    columns (``ragas_*``) + the three clean-baseline columns
    (``baseline_*``). Build the merge inline in the Aggregate page via
    ``runs.merge(baseline, on='query_id', how='left')``.
    """
    metrics = [
        ("ragas_faithfulness",      "baseline_faithfulness",      "Faithfulness"),
        ("ragas_answer_relevance",  "baseline_answer_relevance",  "Answer Relevance"),
        ("ragas_context_relevance", "baseline_context_relevance", "Context Relevance"),
    ]

    fig = go.Figure()
    if merged.empty:
        return fig

    cells_present = [c for c in _CELL_ORDER if c in merged["cell"].unique().tolist()]
    n_metrics = len(metrics)
    n_cells = len(cells_present)
    # Place violins on an integer x-axis: metric index × cell offset.
    width_per_cell = 0.85 / max(n_cells, 1)
    x_ticks: list[float] = []
    x_labels: list[str] = []

    for mi, (attacked_col, clean_col, label) in enumerate(metrics):
        for ci, cell in enumerate(cells_present):
            g = merged[merged["cell"] == cell]
            attacked_vals = g[attacked_col].dropna().astype(float).tolist()
            clean_vals = g[clean_col].dropna().astype(float).tolist() if clean_col in g.columns else []
            x_centre = mi + (ci - (n_cells - 1) / 2) * width_per_cell
            if clean_vals:
                fig.add_trace(go.Violin(
                    x=[x_centre] * len(clean_vals),
                    y=clean_vals,
                    side="negative",
                    line_color="#888780",
                    fillcolor="#CCCCCC",
                    opacity=0.55,
                    showlegend=(mi == 0 and ci == 0),
                    name="clean",
                    legendgroup="clean",
                    points=False,
                    width=width_per_cell * 0.95,
                    spanmode="hard",
                ))
            if attacked_vals:
                fig.add_trace(go.Violin(
                    x=[x_centre] * len(attacked_vals),
                    y=attacked_vals,
                    side="positive",
                    line_color=_CELL_COLOURS.get(cell, "#444444"),
                    fillcolor=_CELL_COLOURS.get(cell, "#444444"),
                    opacity=0.7,
                    showlegend=(mi == 0),
                    name=cell,
                    legendgroup=cell,
                    points=False,
                    width=width_per_cell * 0.95,
                    spanmode="hard",
                ))
        x_ticks.append(mi)
        x_labels.append(label)

    fig.update_layout(
        violinmode="overlay",
        margin=dict(l=40, r=20, t=30, b=24),
        height=360,
        plot_bgcolor="#FAFAF7",
        paper_bgcolor="#FAFAF7",
        xaxis=dict(
            tickmode="array",
            tickvals=x_ticks,
            ticktext=x_labels,
            tickfont=dict(size=11, color="#1F1F1B"),
            range=[-0.6, n_metrics - 0.4],
            showgrid=False,
        ),
        yaxis=dict(
            range=[-0.02, 1.05],
            tickformat=".1f",
            gridcolor="#E5E4DC",
            zerolinecolor="#E5E4DC",
            tickfont=dict(family="JetBrains Mono, monospace", size=10, color="#5F5E5A"),
            title=dict(text="score", font=dict(size=11, color="#5F5E5A")),
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.0,
            xanchor="right",
            x=1.0,
            font=dict(family="JetBrains Mono, monospace", size=10, color="#5F5E5A"),
        ),
    )
    return fig


def rank_shift_ecdf(runs: pd.DataFrame) -> go.Figure:
    """ECDF of ``rank_shift_at_k`` per cell.

    Mirrors Figure F6 from ``scripts/08_make_plots.py``. X is the
    rank-shift value (integer ∈ [0, k]); Y is the cumulative
    fraction of runs at or below that value.
    """
    fig = go.Figure()
    if runs.empty or "rank_shift_at_k" not in runs.columns:
        return fig

    cells_present = [c for c in _CELL_ORDER if c in runs["cell"].unique().tolist()]
    for cell in cells_present:
        g = runs[runs["cell"] == cell]
        vals = g["rank_shift_at_k"].dropna().astype(float).sort_values().to_numpy()
        if vals.size == 0:
            continue
        # Step-function ECDF: at each unique x, y rises to (#vals ≤ x) / n.
        n = vals.size
        ys = ((vals.searchsorted(vals, side="right")) / n)
        fig.add_trace(go.Scatter(
            x=vals.tolist(),
            y=ys.tolist(),
            mode="lines",
            line=dict(color=_CELL_COLOURS.get(cell, "#444444"), width=2, shape="hv"),
            name=f"{cell} (n={n})",
        ))

    fig.update_layout(
        margin=dict(l=40, r=20, t=10, b=24),
        height=320,
        plot_bgcolor="#FAFAF7",
        paper_bgcolor="#FAFAF7",
        xaxis=dict(
            gridcolor="#E5E4DC",
            zerolinecolor="#E5E4DC",
            tickfont=dict(family="JetBrains Mono, monospace", size=10, color="#5F5E5A"),
            title=dict(text="rank_shift@k", font=dict(size=11, color="#5F5E5A")),
        ),
        yaxis=dict(
            range=[0, 1.02],
            tickformat=".0%",
            gridcolor="#E5E4DC",
            zerolinecolor="#E5E4DC",
            tickfont=dict(family="JetBrains Mono, monospace", size=10, color="#5F5E5A"),
            title=dict(text="cumulative share", font=dict(size=11, color="#5F5E5A")),
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.0,
            xanchor="right",
            x=1.0,
            font=dict(family="JetBrains Mono, monospace", size=10, color="#5F5E5A"),
        ),
    )
    return fig


__all__ = [
    "asr_bar_chart",
    "faithfulness_overlay_hist",
    "ragas_violins",
    "rank_shift_ecdf",
    "current_theme",
    "dark_layout",
]
