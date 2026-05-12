"""Day-10 analysis package - loaders, statistics, and Chapter-6 figures.

Public surface:

- :class:`ExperimentData` and :func:`load_experiment` - the 600-run matrix
- :func:`load_clean_baseline`, :func:`validate_clean_baseline` - clean baseline
- :func:`load_bundles_for_k_curve` - per-rank rows for the F3 dose-response
- :func:`bootstrap_mean_ci`, :func:`bootstrap_proportion_ci`, :func:`cohen_h`
- :func:`asr_r_at_k`, :func:`asr_r_curve` - retrieval-depth analysis
- :func:`summary_by_cell`, :func:`ragas_by_cell`, :func:`paired_differences_vs_ipi`
- :func:`build_summary_tables`, :func:`write_tables`, :func:`write_summary_json`
- :func:`plot_asr_triple_by_cell` (F1), :func:`plot_channel_objective_heatmap` (F2),
  :func:`plot_asr_r_vs_k` (F3), :func:`plot_asr_deny_by_cell` (F4),
  :func:`plot_ragas_triple_violins` (F5), :func:`plot_rank_shift_ecdf` (F6),
  :func:`plot_planner_adaptation` (F7), :func:`make_all_plots`
- :data:`CELL_COLOURS`, :data:`CELL_ORDER`, :func:`apply_default_style`
"""

from redteam.analysis.loaders import (
    CORPUS_CHANNEL_CELLS,
    DEFAULT_BASELINE_PATH,
    EXPECTED_CLEAN_QUERIES,
    ExperimentData,
    load_bundles_for_k_curve,
    load_clean_baseline,
    load_experiment,
    validate_clean_baseline,
)
from redteam.analysis.palette import (
    ASR_COLOURS,
    CELL_COLOURS,
    CELL_DISPLAY,
    CELL_ORDER,
    OKABE_ITO,
    RAGAS_COLOURS,
    apply_default_style,
)
from redteam.analysis.plots import (
    make_all_plots,
    plot_asr_deny_by_cell,
    plot_asr_r_vs_k,
    plot_asr_triple_by_cell,
    plot_channel_objective_heatmap,
    plot_planner_adaptation,
    plot_ragas_triple_violins,
    plot_rank_shift_ecdf,
)
from redteam.analysis.stats import (
    CELL_ORDER as STATS_CELL_ORDER,
    INTEGRITY_DEGRADED_THRESHOLD,
    asr_r_at_k,
    asr_r_curve,
    bootstrap_mean_ci,
    bootstrap_proportion_ci,
    build_summary_tables,
    cohen_h,
    paired_differences_vs_ipi,
    ragas_by_cell,
    summary_by_cell,
    write_summary_json,
    write_tables,
)

__all__ = [
    # palette / style
    "OKABE_ITO",
    "CELL_COLOURS",
    "CELL_ORDER",
    "CELL_DISPLAY",
    "ASR_COLOURS",
    "RAGAS_COLOURS",
    "apply_default_style",
    # loaders
    "ExperimentData",
    "load_experiment",
    "load_clean_baseline",
    "load_bundles_for_k_curve",
    "validate_clean_baseline",
    "DEFAULT_BASELINE_PATH",
    "EXPECTED_CLEAN_QUERIES",
    "CORPUS_CHANNEL_CELLS",
    # stats
    "STATS_CELL_ORDER",
    "INTEGRITY_DEGRADED_THRESHOLD",
    "bootstrap_mean_ci",
    "bootstrap_proportion_ci",
    "cohen_h",
    "asr_r_at_k",
    "asr_r_curve",
    "summary_by_cell",
    "ragas_by_cell",
    "paired_differences_vs_ipi",
    "build_summary_tables",
    "write_tables",
    "write_summary_json",
    # plots
    "plot_asr_triple_by_cell",
    "plot_channel_objective_heatmap",
    "plot_asr_r_vs_k",
    "plot_asr_deny_by_cell",
    "plot_ragas_triple_violins",
    "plot_rank_shift_ecdf",
    "plot_planner_adaptation",
    "make_all_plots",
]
