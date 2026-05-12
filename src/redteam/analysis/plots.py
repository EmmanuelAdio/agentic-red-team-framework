"""Chapter-6 figure functions (F1-F7).

Each ``plot_*`` function:

- calls :func:`redteam.analysis.palette.apply_default_style` at entry so
  every figure shares the same typography / palette / grid;
- takes the DataFrames it actually needs (not a giant ``data`` blob) so
  the dependency graph is explicit;
- writes a PDF (vector, for the dissertation) and a PNG @ 300 dpi (raster,
  for the notebook preview) into ``out_dir``, and returns the PDF path.

Figure inventory
----------------

================ ============================================ ===============================
ID               filename stem                                source DataFrame
================ ============================================ ===============================
F1               ``asr_triple_by_cell``                       summary_by_cell
F2               ``channel_objective_heatmap``                summary_by_cell
F3               ``asr_r_vs_k``                               asr_r_curve(bundles)
F4               ``asr_deny_by_cell``                         summary_by_cell
F5               ``ragas_triple_clean_vs_attacked``           runs + baseline
F6               ``rank_shift_ecdf``                          runs
F7               ``planner_adaptation``                       sidecars
================ ============================================ ===============================

Conventions across all figures
------------------------------

- 95% bootstrap CIs everywhere; computed by :mod:`redteam.analysis.stats`.
- Cell display labels and colours come from
  :mod:`redteam.analysis.palette` so the legends line up across figures.
- All figures save to ``out_dir`` named with the stem above; both PDF
  and PNG are emitted in one call. PDF is the canonical artefact - PNG
  is for notebook inline display only.
- No chartjunk: no shadows, no 3D bars, no per-bar text annotations
  unless they carry information the axis does not already provide.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from redteam.analysis.loaders import ExperimentData, load_bundles_for_k_curve
from redteam.analysis.palette import (
    ASR_COLOURS,
    CELL_COLOURS,
    CELL_DISPLAY,
    CELL_ORDER,
    OKABE_ITO,
    RAGAS_COLOURS,
    apply_default_style,
)
from redteam.analysis.stats import (
    CELL_ORDER as STATS_CELL_ORDER,
    asr_r_curve,
)

# ---------------------------------------------------------------------------
# Save helper
# ---------------------------------------------------------------------------


def _savefig(fig: plt.Figure, out_dir: Path, stem: str) -> Path:
    """Persist a figure as both PDF (vector) and PNG (300 dpi).

    Returns the PDF path; the PNG sits beside it with the same stem and
    is what the notebook's ``IPython.display.Image`` cells render
    inline.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = out_dir / f"{stem}.pdf"
    png_path = out_dir / f"{stem}.png"
    fig.savefig(pdf_path)
    fig.savefig(png_path)
    plt.close(fig)
    return pdf_path


def _cells_in_summary(summary: pd.DataFrame) -> list[str]:
    """Return cells present in ``summary`` in canonical order."""
    present = set(summary["cell"].astype(str))
    return [c for c in CELL_ORDER if c in present]


# ---------------------------------------------------------------------------
# F1 - ASR-r / ASR-a / ASR-t grouped bar chart
# ---------------------------------------------------------------------------


def plot_asr_triple_by_cell(
    summary: pd.DataFrame,
    out_dir: Path,
    stem: str = "asr_triple_by_cell",
) -> Path:
    """F1: grouped bars of ASR-r / ASR-a / ASR-t per cell with 95% CIs.

    The PoisonedRAG / AgentPoison convention is to separate the three
    ASR variants so the reader can see *where* an attack succeeds or
    fails: high ASR-r with low ASR-a means the poisoned doc reached the
    LLM but the LLM ignored it; high ASR-a with low ASR-r is structurally
    impossible for corpus-channel attacks but can occur for query-
    injection where ASR-r is True by construction.
    """
    apply_default_style()

    cells = _cells_in_summary(summary)
    metrics = [
        ("asr_retrieval_rate", "asr_retrieval_ci_low", "asr_retrieval_ci_high", "ASR-r"),
        ("asr_answer_rate",    "asr_answer_ci_low",    "asr_answer_ci_high",    "ASR-a"),
        ("asr_target_rate",    "asr_target_ci_low",    "asr_target_ci_high",    "ASR-t"),
    ]
    metric_colours = [
        ASR_COLOURS["asr_retrieval"],
        ASR_COLOURS["asr_answer"],
        ASR_COLOURS["asr_target"],
    ]

    fig, ax = plt.subplots(figsize=(8.5, 4.6))

    n_cells = len(cells)
    n_metrics = len(metrics)
    group_width = 0.78
    bar_width = group_width / n_metrics
    x = np.arange(n_cells)

    for i, ((mean_col, lo_col, hi_col, label), colour) in enumerate(zip(metrics, metric_colours)):
        means = [float(summary.loc[summary["cell"] == c, mean_col].iloc[0]) for c in cells]
        lows  = [float(summary.loc[summary["cell"] == c, lo_col].iloc[0])   for c in cells]
        highs = [float(summary.loc[summary["cell"] == c, hi_col].iloc[0])   for c in cells]
        # asymmetric error bars are (mean - low, high - mean).
        yerr = np.array([
            [m - lo for m, lo in zip(means, lows)],
            [hi - m for m, hi in zip(means, highs)],
        ])
        offset = (i - (n_metrics - 1) / 2.0) * bar_width
        ax.bar(
            x + offset, means,
            width=bar_width * 0.92,
            color=colour,
            edgecolor="#333333",
            linewidth=0.5,
            label=label,
        )
        ax.errorbar(
            x + offset, means,
            yerr=yerr,
            fmt="none",
            ecolor="#333333",
            elinewidth=1.0,
            capsize=2.5,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([CELL_DISPLAY.get(c, c) for c in cells])
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Attack-success rate (proportion)")
    ax.set_title("Figure F1 - ASR-r / ASR-a / ASR-t per attack cell (95% bootstrap CI)")
    ax.legend(loc="lower right", ncols=3, title=None)
    ax.set_axisbelow(True)
    fig.tight_layout()
    return _savefig(fig, out_dir, stem)


# ---------------------------------------------------------------------------
# F2 - 2x2 channel x objective heatmap
# ---------------------------------------------------------------------------


def plot_channel_objective_heatmap(
    summary: pd.DataFrame,
    out_dir: Path,
    stem: str = "channel_objective_heatmap",
) -> Path:
    """F2: 2x2 heatmap of headline-success by channel x objective.

    Each tile shows the cell's headline success rate. The legend on the
    right names the metric each tile reports (ASR-t for the three
    integrity cells, ASR-deny for the availability cell) so the reader
    isn't asked to remember that the framework changes metrics between
    objectives.
    """
    apply_default_style()

    channels = ["corpus", "query"]
    objectives = ["integrity", "availability"]
    # Mean values + the label-text for each tile.
    rate_matrix = np.full((len(objectives), len(channels)), np.nan)
    text_matrix = np.full((len(objectives), len(channels)), "", dtype=object)

    for _, row in summary.iterrows():
        ch = str(row["channel"])
        ob = str(row["objective"])
        if ch not in channels or ob not in objectives:
            continue
        i = objectives.index(ob)
        j = channels.index(ch)
        rate = float(row["headline_success_rate"])
        lo = float(row["headline_success_ci_low"])
        hi = float(row["headline_success_ci_high"])
        rate_matrix[i, j] = rate
        text_matrix[i, j] = (
            f"{row['cell']}\n"
            f"{row.get('success_metric', '')}\n"
            f"{rate:.2f}  [{lo:.2f}, {hi:.2f}]"
        )

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    im = ax.imshow(
        rate_matrix,
        cmap="Oranges",
        vmin=0.0,
        vmax=1.0,
        aspect="auto",
    )
    ax.set_xticks(range(len(channels)))
    ax.set_yticks(range(len(objectives)))
    ax.set_xticklabels([c.capitalize() for c in channels])
    ax.set_yticklabels([o.capitalize() for o in objectives])
    ax.set_xlabel("Delivery channel")
    ax.set_ylabel("Adversarial objective")
    ax.set_title("Figure F2 - Headline attack success by channel x objective")

    # Tile annotations: black text on the lighter tiles, white on the darker
    # ones, to keep the labels legible regardless of the underlying rate.
    for i in range(len(objectives)):
        for j in range(len(channels)):
            if np.isnan(rate_matrix[i, j]):
                continue
            text_colour = "white" if rate_matrix[i, j] > 0.6 else "#222222"
            ax.text(
                j, i, text_matrix[i, j],
                ha="center", va="center",
                color=text_colour,
                fontsize=9.0,
                linespacing=1.25,
            )

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.06)
    cbar.set_label("Success rate (proportion)")
    ax.grid(False)
    fig.tight_layout()
    return _savefig(fig, out_dir, stem)


# ---------------------------------------------------------------------------
# F3 - ASR-r vs retrieval depth k
# ---------------------------------------------------------------------------


def plot_asr_r_vs_k(
    bundles: pd.DataFrame,
    out_dir: Path,
    *,
    ks: tuple[int, ...] = (1, 2, 3, 4, 5),
    stem: str = "asr_r_vs_k",
    n_resamples: int = 1000,
    bootstrap_seed: int = 12345,
) -> Path:
    """F3: ASR-r vs retrieval depth k for each corpus-channel cell.

    Dose-response curve mirroring PoisonedRAG (Zou et al. 2024) Fig. 3.
    One line per corpus-channel cell with a shaded 95% bootstrap-CI band
    across seeds. Query-injection (``qInj``) is excluded by construction:
    the attack modifies the query, not the corpus, so its ASR-r is True
    at every k by definition.
    """
    apply_default_style()
    curve = asr_r_curve(
        bundles, ks=ks, n_resamples=n_resamples, seed=bootstrap_seed
    )

    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    for cell in CELL_ORDER:
        c_rows = curve[curve["cell"] == cell].sort_values("k")
        if c_rows.empty:
            continue
        colour = CELL_COLOURS.get(cell, "#444444")
        ax.plot(
            c_rows["k"].to_numpy(), c_rows["mean"].to_numpy(),
            color=colour,
            marker="o",
            label=cell,
        )
        ax.fill_between(
            c_rows["k"].to_numpy(),
            c_rows["ci_low"].to_numpy(),
            c_rows["ci_high"].to_numpy(),
            color=colour,
            alpha=0.18,
            linewidth=0.0,
        )

    ax.set_xlabel("Retrieval depth $k$")
    ax.set_ylabel("ASR-r (retrieval-stage success rate)")
    ax.set_title("Figure F3 - ASR-r vs retrieval depth (95% bootstrap CI over seeds)")
    ax.set_xticks(list(ks))
    ax.set_ylim(0.0, 1.05)
    ax.legend(title="Corpus-channel cells", loc="lower right")
    fig.text(
        0.01, 0.01,
        "Note: query-injection cell (qInj) is excluded - ASR-r is True at every $k$ by construction.",
        fontsize=8.0,
        color="#555555",
    )
    fig.tight_layout(rect=(0.0, 0.04, 1.0, 1.0))
    return _savefig(fig, out_dir, stem)


# ---------------------------------------------------------------------------
# F4 - ASR-deny bar (honest negative-result panel)
# ---------------------------------------------------------------------------


def plot_asr_deny_by_cell(
    summary: pd.DataFrame,
    out_dir: Path,
    stem: str = "asr_deny_by_cell",
) -> Path:
    """F4: ASR-deny per cell with 95% bootstrap CIs.

    The jamming cell (``poiJ``) targets *availability* via ASR-deny; the
    others target integrity and report ASR-deny for completeness. A 0%
    ASR-deny for jamming is a substantive negative finding (the
    framework's jamming strategy does not coerce the LLM into refusing
    or punting on the answer), and the figure shows it without hiding.
    """
    apply_default_style()

    cells = _cells_in_summary(summary)
    means = [float(summary.loc[summary["cell"] == c, "asr_deny_rate"].iloc[0]) for c in cells]
    lows  = [float(summary.loc[summary["cell"] == c, "asr_deny_ci_low"].iloc[0])  for c in cells]
    highs = [float(summary.loc[summary["cell"] == c, "asr_deny_ci_high"].iloc[0]) for c in cells]
    colours = [CELL_COLOURS.get(c, "#666666") for c in cells]

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    x = np.arange(len(cells))
    bars = ax.bar(
        x, means,
        width=0.62,
        color=colours,
        edgecolor="#333333",
        linewidth=0.5,
    )
    yerr = np.array([
        [m - lo for m, lo in zip(means, lows)],
        [hi - m for m, hi in zip(means, highs)],
    ])
    ax.errorbar(x, means, yerr=yerr, fmt="none", ecolor="#333333", elinewidth=1.0, capsize=3.0)

    # Annotate every zero / near-zero bar so a print-out doesn't show an
    # empty bar that's hard to distinguish from a missing data point.
    for xi, m in zip(x, means):
        if m < 0.02:
            ax.text(xi, 0.02, "0.00", ha="center", va="bottom", fontsize=9.0, color="#555555")

    ax.set_xticks(x)
    ax.set_xticklabels([CELL_DISPLAY.get(c, c) for c in cells])
    ax.set_ylim(0.0, max(1.0, max(highs) * 1.10 if highs else 1.0))
    ax.set_ylabel("ASR-deny (refusal / non-answer rate)")
    ax.set_title("Figure F4 - ASR-deny per cell (jamming = availability objective)")
    fig.tight_layout()
    return _savefig(fig, out_dir, stem)


# ---------------------------------------------------------------------------
# F5 - RAGAS triple, clean vs attacked, split violins
# ---------------------------------------------------------------------------


def _violin_bodies(parts: dict, colour: str) -> None:
    """Style a matplotlib violin returned by ``Axes.violinplot``.

    The default styling has a black edge, which makes adjacent violins
    blur together on print. We strip the edge, set a translucent fill,
    and hide the per-distribution lines (median/min/max) that the
    matplotlib default draws - we'll overlay a mean marker + CI bar
    ourselves so the violin and the summary statistic come from the
    same source of truth.
    """
    for body in parts["bodies"]:
        body.set_facecolor(colour)
        body.set_alpha(0.55)
        body.set_edgecolor(colour)
        body.set_linewidth(0.6)
    for key in ("cmins", "cmaxes", "cbars", "cmedians", "cmeans", "cquantiles"):
        if key in parts:
            parts[key].set_visible(False)


def plot_ragas_triple_violins(
    runs: pd.DataFrame,
    baseline: pd.DataFrame,
    out_dir: Path,
    stem: str = "ragas_triple_clean_vs_attacked",
) -> Path:
    """F5: three-panel violins of RAGAS Faithfulness / AR / CR.

    Each panel: clean baseline on the left (grey), then one violin per
    attack cell in the canonical order. A black dot marks the mean and a
    short vertical bar marks the 95% bootstrap CI on the mean - the same
    estimate the per-cell tables report. The point of the figure is to
    show that high ASR can coexist with high Faithfulness (because the
    poisoned context *supports* the false answer), which is the Chapter
    7 hook.
    """
    apply_default_style()

    metrics = [
        ("ragas_faithfulness",      "Faithfulness"),
        ("ragas_answer_relevance",  "Answer Relevance"),
        ("ragas_context_relevance", "Context Relevance"),
    ]
    cells = [c for c in CELL_ORDER if c in set(runs["cell"].astype(str))]
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 4.4), sharey=True)

    for ax, (col, title) in zip(axes, metrics):
        # Series for each violin in plotting order: clean baseline first,
        # then one per cell.
        data_series: list[np.ndarray] = []
        colours: list[str] = []
        labels: list[str] = []

        clean = baseline[col].dropna().to_numpy(dtype=float)
        if clean.size:
            data_series.append(clean)
            colours.append(CELL_COLOURS["clean"])
            labels.append(CELL_DISPLAY["clean"])

        for cell in cells:
            g = runs[runs["cell"] == cell][col].dropna().to_numpy(dtype=float)
            if g.size == 0:
                continue
            data_series.append(g)
            colours.append(CELL_COLOURS.get(cell, "#444444"))
            labels.append(CELL_DISPLAY.get(cell, cell))

        positions = np.arange(1, len(data_series) + 1)
        if data_series:
            parts = ax.violinplot(
                data_series,
                positions=positions,
                showextrema=False,
                widths=0.78,
            )
            # Apply colours one body at a time so each violin has its
            # own fill.
            for body, colour in zip(parts["bodies"], colours):
                body.set_facecolor(colour)
                body.set_alpha(0.55)
                body.set_edgecolor(colour)
                body.set_linewidth(0.7)

            # Mean dot + 95% bootstrap CI bar.
            from redteam.analysis.stats import bootstrap_mean_ci
            for pos, arr in zip(positions, data_series):
                ci = bootstrap_mean_ci(arr.tolist())
                ax.errorbar(
                    pos, ci["mean"],
                    yerr=[[ci["mean"] - ci["ci_low"]], [ci["ci_high"] - ci["mean"]]],
                    fmt="o",
                    color="#222222",
                    ecolor="#222222",
                    elinewidth=1.0,
                    capsize=2.5,
                    markersize=4.5,
                )

        # The five labels won't fit horizontally in a 3-subplot panel
        # without overlapping, so rotate them 35 degrees and right-align.
        # Use the short display tokens (cell label only, family name in
        # parentheses kept tiny via newline + smaller font).
        short_labels = [lab.replace("\n", " ") for lab in labels]
        ax.set_xticks(positions)
        ax.set_xticklabels(
            short_labels,
            rotation=30,
            ha="right",
            fontsize=8.5,
            rotation_mode="anchor",
        )
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(title)
        ax.set_xlabel("")
        if ax is axes[0]:
            ax.set_ylabel("RAGAS score (0-1)")

    fig.suptitle(
        "Figure F5 - RAGAS triple: clean baseline vs each attacked cell",
        y=1.02,
    )
    fig.tight_layout()
    return _savefig(fig, out_dir, stem)


# ---------------------------------------------------------------------------
# F6 - rank_shift@k ECDF
# ---------------------------------------------------------------------------


def plot_rank_shift_ecdf(
    runs: pd.DataFrame,
    out_dir: Path,
    stem: str = "rank_shift_ecdf",
) -> Path:
    """F6: empirical CDF of rank_shift@5 per cell.

    Replaces the previous stacked-bar form. ECDFs are the standard IR /
    adversarial-ranking convention because they preserve the tail (one
    in twenty queries with a big rank-shift is invisible in a mean but
    obvious in a CDF). A reference line at zero marks "no rank change".
    """
    apply_default_style()

    cells = [c for c in CELL_ORDER if c in set(runs["cell"].astype(str))]
    fig, ax = plt.subplots(figsize=(7.4, 4.6))

    for cell in cells:
        rs = runs.loc[runs["cell"] == cell, "rank_shift_at_k"].dropna().astype(int)
        if rs.empty:
            continue
        sorted_rs = np.sort(rs.to_numpy())
        # ECDF y-values: (i+1)/n for the i-th sorted observation.
        y = np.arange(1, len(sorted_rs) + 1) / float(len(sorted_rs))
        ax.step(
            sorted_rs, y,
            where="post",
            color=CELL_COLOURS.get(cell, "#444444"),
            linewidth=1.8,
            label=cell,
        )

    ax.axvline(0.0, linestyle=":", color="#888888", linewidth=1.0)
    # Annotation lives below the x-axis where it does not collide with
    # the step-curves. The Axes are tightened by 4% in fig.tight_layout
    # below to make room.
    ax.annotate(
        "rank-shift = 0  (clean top-1 unchanged)",
        xy=(0.0, 0.0),
        xytext=(0.5, -0.18),
        textcoords="axes fraction",
        fontsize=8.5,
        color="#555555",
        ha="center",
    )
    ax.set_xlabel("rank_shift@5 (positions displaced)")
    ax.set_ylabel("Empirical CDF")
    ax.set_title("Figure F6 - rank_shift@5 distribution per cell (150 runs each)")
    ax.set_ylim(0.0, 1.02)
    ax.legend(title="Cell", loc="lower right")
    fig.tight_layout(rect=(0.0, 0.06, 1.0, 1.0))
    return _savefig(fig, out_dir, stem)


# ---------------------------------------------------------------------------
# F7 - planner adaptation (two-panel)
# ---------------------------------------------------------------------------


def plot_planner_adaptation(
    sidecars: pd.DataFrame,
    out_dir: Path,
    stem: str = "planner_adaptation",
) -> Path:
    """F7: two-panel planner-sidecar adaptation figure.

    Left panel: running success-rate per family vs query index, averaged
    across seeds with a shaded 95% bootstrap-CI band. The lines come from
    the ``running_rate_<family>`` columns that
    :func:`redteam.analysis.loaders._add_running_success_rate` computed
    (forward-filled within each seed so the line is continuous).

    Right panel: arm-pull histogram - how many times each family was
    chosen, totalled across the three seeds. Mirrors the bandit-literature
    convention of pairing a learning-curve with an exploration-profile
    histogram.
    """
    apply_default_style()
    if sidecars.empty:
        raise ValueError("Sidecar DataFrame is empty; cannot draw F7.")

    # Map planner-family names to display colours. The planner has three
    # arms (the four cells collapse into three families because
    # corpus_poisoning covers both poiA and poiJ).
    family_colour: dict[str, str] = {
        "prompt_injection":  CELL_COLOURS["ipi"],
        "corpus_poisoning":  CELL_COLOURS["poiA"],
        "query_injection":   CELL_COLOURS["qInj"],
    }
    family_display: dict[str, str] = {
        "prompt_injection":  "prompt_injection",
        "corpus_poisoning":  "corpus_poisoning",
        "query_injection":   "query_injection",
    }
    rate_columns = [c for c in sidecars.columns if c.startswith("running_rate_")]

    fig, (ax_left, ax_right) = plt.subplots(
        1, 2, figsize=(12.0, 4.6), gridspec_kw={"width_ratios": [2.0, 1.0]}
    )

    # ---- Left panel: per-family running rate ------------------------------
    from redteam.analysis.stats import bootstrap_mean_ci

    for col in rate_columns:
        family = col.replace("running_rate_", "")
        colour = family_colour.get(family, "#555555")
        # Wide form: rows = selection_order, cols = seed, values = running rate.
        wide = (
            sidecars.pivot_table(
                index="selection_order",
                columns="seed",
                values=col,
                aggfunc="last",
            )
            .sort_index()
        )
        if wide.empty:
            continue
        # Bootstrap CI across seeds at each step (small n=3 per step but
        # the band is the right way to show seed-to-seed variability).
        means = wide.mean(axis=1).to_numpy(dtype=float)
        lows: list[float] = []
        highs: list[float] = []
        for _, row in wide.iterrows():
            ci = bootstrap_mean_ci(row.dropna().to_list(), n_resamples=500)
            lows.append(ci["ci_low"])
            highs.append(ci["ci_high"])
        x = wide.index.to_numpy()
        ax_left.plot(x, means, color=colour, linewidth=1.8, label=family_display.get(family, family))
        ax_left.fill_between(x, lows, highs, color=colour, alpha=0.16, linewidth=0.0)

    ax_left.set_xlabel("Selection order (query index 1..50)")
    ax_left.set_ylabel("Running ASR-t success rate")
    ax_left.set_ylim(0.0, 1.05)
    ax_left.set_title("Running success rate per family")
    ax_left.legend(title="Family", loc="lower right", fontsize=8.5)

    # ---- Right panel: arm-pull histogram ----------------------------------
    counts = (
        sidecars.groupby(["seed", "chosen_family"])
        .size()
        .reset_index(name="count")
    )
    families_present: list[str] = sorted(set(counts["chosen_family"]))
    seeds_present: list[int] = sorted(set(counts["seed"]))
    n_fam = len(families_present)
    n_seed = len(seeds_present)
    bar_width = 0.78 / max(n_seed, 1)
    x = np.arange(n_fam)
    seed_colours = [
        OKABE_ITO["blue"],
        OKABE_ITO["reddish_purple"],
        OKABE_ITO["black"],
    ]
    for j, sd in enumerate(seeds_present):
        heights = [
            int(counts.loc[(counts["seed"] == sd) & (counts["chosen_family"] == f), "count"].sum())
            for f in families_present
        ]
        offset = (j - (n_seed - 1) / 2.0) * bar_width
        ax_right.bar(
            x + offset, heights,
            width=bar_width * 0.92,
            color=seed_colours[j % len(seed_colours)],
            edgecolor="#333333",
            linewidth=0.4,
            label=f"seed {sd}",
        )

    ax_right.set_xticks(x)
    ax_right.set_xticklabels(
        [family_display.get(f, f) for f in families_present],
        rotation=14,
        ha="right",
        fontsize=8.5,
    )
    ax_right.set_ylabel("Selection count")
    ax_right.set_title("Arm pulls per family (per seed)")
    ax_right.legend(title=None, loc="upper right", fontsize=8.5)

    fig.suptitle(
        "Figure F7 - Planner adaptation: epsilon-greedy convergence + arm-pull histogram",
        y=1.02,
    )
    fig.tight_layout()
    return _savefig(fig, out_dir, stem)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def make_all_plots(
    data: ExperimentData,
    baseline: pd.DataFrame,
    summary_table: pd.DataFrame,
    out_dir: Path,
    *,
    n_resamples: int = 1000,
    bootstrap_seed: int = 12345,
    bundles: pd.DataFrame | None = None,
) -> list[Path]:
    """Produce all 7 Chapter-6 figures in canonical order.

    The driver takes ``summary_table`` already built (so the figures and
    the CSV stay byte-consistent), and lazily loads the bundle-level
    per-rank data for F3 if the caller hasn't already.
    """
    out_dir = Path(out_dir)

    if bundles is None:
        bundles = load_bundles_for_k_curve(data.results_dir)

    paths: list[Path] = []
    paths.append(plot_asr_triple_by_cell(summary_table, out_dir))
    paths.append(plot_channel_objective_heatmap(summary_table, out_dir))
    paths.append(plot_asr_r_vs_k(
        bundles, out_dir, n_resamples=n_resamples, bootstrap_seed=bootstrap_seed
    ))
    paths.append(plot_asr_deny_by_cell(summary_table, out_dir))
    paths.append(plot_ragas_triple_violins(data.runs, baseline, out_dir))
    paths.append(plot_rank_shift_ecdf(data.runs, out_dir))
    paths.append(plot_planner_adaptation(data.sidecars, out_dir))
    return paths
