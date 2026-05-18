# Dashboard

Streamlit dashboard for the agentic red-team framework. Build A
(Overview + Run Detail) shipped Day 14; Build B (Aggregate page, dark
mode, DuckDB query layer) rolled in Day 15.

## Quickstart

```powershell
# Windows / PowerShell — from the repo root, with the project's venv active
pip install -r requirements.txt
.\scripts\09_run_dashboard.ps1
```

```bash
# Linux / macOS / WSL / Git-Bash
pip install -r requirements.txt
bash scripts/09_run_dashboard.sh
```

Opens at <http://localhost:8501>. Bundles are read recursively from
`results/runs/` first and `data/runs/` as a fallback (both are searched
through the nested `batch_<id>/` folders that `BundleStore` writes to).

### Re-loading `.env`

The launcher sources `.env` (at the repo root) on every invocation via
`python-dotenv`, but Streamlit itself **does not** re-read process
environment variables on its in-browser "Rerun". Dashboard env knobs
(`REDTEAM_DASHBOARD_THEME`, `REDTEAM_DASHBOARD_DUCKDB`, …) only take
effect at server start. To pick up `.env` edits, restart the
launcher with the `-Restart` flag, which terminates any process
already listening on `$STREAMLIT_PORT` first:

```powershell
.\scripts\09_run_dashboard.ps1 -Restart
```

```bash
bash scripts/09_run_dashboard.sh --restart
```

Empty values in `.env` are skipped, so a shell-level override set
before launch (e.g. `$env:STREAMLIT_PORT = "8600"`) is preserved
even if the same key is present-but-blank in the file.

## Pages

| Page | Path | Role |
| --- | --- | --- |
| Overview | `dashboard/Home.py` | Landing page — metric tiles, verdict legend, ASR-t bar chart, Faithfulness overlay histogram, per-cell summary (family × channel from bundles), manifest-aware per-cell summary + RAGAS-by-cell + paired-diffs-vs-IPI tables (when `experiment_manifest.json` is present), recent runs. |
| Run Detail | `dashboard/pages/02_run_detail.py?run_id=…` | Drill-down — config, retrieved docs (poisoned highlighted), generator output, RAGAS + ASR, raw bundle. Empty-state card if `run_id` is missing or unknown. |

The standalone Aggregate page was retired and its three tables were
folded into the Overview page directly under the family×channel
summary, each with an inline column-reference expander. The Aggregate
page's split-violin and `rank_shift@k` ECDF charts were not migrated
(the static figure stack in `results/figures/` covers the same
information for the dissertation).

## Environment variables

| Variable | Effect |
| --- | --- |
| `REDTEAM_DASHBOARD_THEME=dark` | Switch every page to the dark palette. Read once at page-import time, so toggling requires a server restart (`-Restart` / `--restart`). Default `light`. |
| `REDTEAM_DASHBOARD_DUCKDB=1` | Replace the default recursive-glob + per-file `json.load` backend with an in-process DuckDB view over `results/runs/**/*_bundle.json` and `data/runs/**/*_bundle.json`. The schema is locked by the smoke test `test_duckdb_query_select_42`. Default off. |
| `STREAMLIT_PORT` | TCP port (default 8501). |
| `STREAMLIT_HEADLESS` | `true`/`false` (default `true`). |

## File map

```
dashboard/
├── Home.py                       # Overview page (Streamlit entry point)
├── pages/
│   └── 02_run_detail.py          # Run Detail page
├── .streamlit/
│   └── config.toml               # Streamlit theme + server settings
└── README.md                     # this file

src/redteam/dashboard/
├── __init__.py
├── _css.py                       # inject_css(theme=…) — light + dark
├── components.py                 # badge, doc_card, score_bar, kv_grid,
│                                 # asr_cell, asr_grid, page_header,
│                                 # verdict_legend, empty_state
├── data.py                       # load_bundles (dispatches to DuckDB if enabled),
│                                 # load_one_bundle, bootstrap_ci,
│                                 # summary_by_family_channel
├── charts.py                     # asr_bar_chart, faithfulness_overlay_hist,
│                                 # ragas_violins, rank_shift_ecdf,
│                                 # current_theme, dark_layout
├── filters.py                    # apply_filters(df, sel), available_options(df)
└── duck.py                       # DuckDB façade (opt-in)

scripts/
├── 09_run_dashboard.sh           # bash launcher
└── 09_run_dashboard.ps1          # PowerShell launcher

tests/
└── test_dashboard_smoke.py       # 10 smoke tests covering imports + pure helpers
```

## Notes

- **Schema**: helpers project the live Pydantic schema in
  `src/redteam/bundles/schema.py`, including the Day-7.5 additive fields
  (`attack_channel`, `payload_source`, `asr_deny`, `iteration_history`,
  `baseline_top1_doc_id`).
- **No live updates.** `load_bundles` is wrapped in
  `@st.cache_data(ttl=300)`; click "Rerun" or wait 5 minutes for new
  bundles to surface.
- **Verdict-to-visual inversion.** The bundle's `evaluation.verdict`
  literal is one of `success | partial | failure`, where `failure`
  means *the defence held*. The badge helper maps `failure → green
  (verdict-defended)` so a defender reading the dashboard sees the
  defence-held signal as green, not the attack-failed literal as red.
  The verdict-legend strip on the Overview page surfaces this
  inversion explicitly.
- **Poisoned doc rendering**: bundle `retrieved_docs` entries don't
  carry chunk text. Run Detail renders the poisoned doc using
  `attack.payload` as its body; clean docs render headers only.
- **DuckDB backend (opt-in).** Set `REDTEAM_DASHBOARD_DUCKDB=1`
  before launch. The view's schema is *explicitly* declared in
  `duck.py` (DuckDB's `read_json_auto` hits OOM during schema
  inference on the 600-bundle tree). The column set is locked
  byte-for-byte against the glob path so downstream code does not
  branch.
