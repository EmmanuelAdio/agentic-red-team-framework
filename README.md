# Agentic Red-Team Framework for LLM-Augmented Search Systems

**Author:** Emmanuel Adio (F229639) &nbsp;·&nbsp; **Supervisor:** Dr Georgina Cosma &nbsp;·&nbsp; **Module:** 25COD290 (MSci, Loughborough University)

An open-source agentic framework that autonomously plans, executes, and evaluates adversarial attacks
(prompt injection + corpus poisoning) against a Retrieval-Augmented Generation pipeline.
Each run produces a reproducible exploit-bundle JSON scored by RAGAS and custom attack-success metrics.

## Setup

```powershell
python -m venv .venv ; .venv\Scripts\activate
pip install -r requirements.txt
Copy-Item .env.example .env   # then edit .env and set OPENAI_API_KEY
python scripts/01_build_corpus.py ; python scripts/02_run_baseline.py
```

## Dashboard

A Streamlit dashboard (Build A + B: *Overview*, *Aggregate*, *Run
Detail*) reads the exploit bundles in [results/runs/](results/runs/)
(with [data/runs/](data/runs/) as a fallback) and renders the
experiment matrix interactively.

- **Overview** — four metric tiles (total runs, overall ASR-t with a
  95 % bootstrap confidence interval, integrity-degraded share at the
  Faithfulness < 0.65 threshold, attack-family count), a Plotly
  horizontal-bar chart of ASR-t grouped by *(attack_family, attack_channel)*
  with bootstrap-CI whiskers, a Faithfulness overlay histogram
  (clean vs attacked, threshold line at 0.65), a per-cell summary
  table, and the last twenty runs as a clickable table. Sidebar
  filters: seeds + attack family + attack channel + payload source +
  verdict (rendered as `st.pills` on Streamlit ≥ 1.36).
- **Aggregate** — cell-aware view of the Day-9 manifest: three RAGAS
  metric tiles (clean vs attacked delta), a per-cell summary table
  with bootstrap CIs, split violins of the RAGAS triple per cell, an
  ECDF of `rank_shift@k`, a paired-differences-vs-IPI table with
  Cohen's-*h*, and a RAGAS-by-cell Faithfulness-drop table. Mirrors
  Figures F1–F7 from [scripts/08_make_plots.py](scripts/08_make_plots.py)
  in interactive form.
- **Run Detail** — per-bundle drill-down: configuration grid,
  retrieved-document list with the poisoned doc highlighted (body
  rendered from `attack.payload`), generator output, RAGAS score
  bars, the ASR cell grid (`ASR-r`, `ASR-a`, `ASR-t`, `rank_shift@k`,
  `ASR-deny`), collapsed iteration history, and a `Download bundle
  JSON` button. Missing or malformed `run_id` URLs land on a
  friendly empty-state card linking back to Overview.

```powershell
# Windows / PowerShell
.\scripts\09_run_dashboard.ps1

# Dark mode (server restart required to toggle)
$env:REDTEAM_DASHBOARD_THEME = "dark" ; .\scripts\09_run_dashboard.ps1

# Opt-in DuckDB backend (SQL-style filterable view over the bundle tree)
$env:REDTEAM_DASHBOARD_DUCKDB = "1" ; .\scripts\09_run_dashboard.ps1
```

```bash
# Linux / macOS / WSL / Git-Bash
bash scripts/09_run_dashboard.sh
```

Opens at <http://localhost:8501>. See
[dashboard/README.md](dashboard/README.md) for the file map +
environment variable reference and [DIAGRAMS.md](DIAGRAMS.md) §8 for
the design rationale (component layout, the verdict-to-visual
inversion, the choice of Streamlit, the DuckDB schema-flattening
view, and the Build-A vs Build-B scope split).

*Screenshots:* `docs/img/dashboard_overview_light.png`,
`docs/img/dashboard_overview_dark.png`,
`docs/img/dashboard_aggregate.png`, and
`docs/img/dashboard_run_detail.png` *(populated during the Day-15
demo dry-run)*.
