# Agentic Red-Team Framework for LLM-Augmented Search Systems

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

## Headline results (matrix, 600 runs)

50 Natural Questions × 4 attack cells × 3 seeds. Per-objective
attribution: each cell is judged by its own success metric.

| Cell | Family / strategy | Channel × objective | Headline | 95 % CI |
| --- | --- | --- | --- | --- |
| `ipi`  | prompt injection / instruction override | corpus × integrity    | **0.96** ASR-t    | [0.92, 0.99] |
| `poiA` | corpus poisoning / answer replacement   | corpus × integrity    | **0.80** ASR-t    | [0.73, 0.86] |
| `poiJ` | corpus poisoning / jamming              | corpus × availability | **0.46** ASR-deny | [0.39, 0.54] |
| `qInj` | query injection / prefix injection      | query × integrity     | **0.96** ASR-a    | [0.93, 0.99] |

figures F1–F8 in [results/figures/](results/figures/); the headline-
success metric per cell drives the dashboard's two top tiles
(ASR-t over integrity cells, ASR-deny over availability cells) so the
two objectives are never pooled.

## Dashboard

A Streamlit dashboard (Build A + B: *Overview*, *Aggregate*, *Run
Detail*) reads the exploit bundles in [results/runs/](results/runs/)
(with [data/runs/](data/runs/) as a fallback) and renders the
experiment matrix interactively.

- **Overview** — four metric tiles with per-objective attribution
  (total runs, *ASR-t (integrity)* averaged over the three
  integrity-objective cells, *ASR-deny (availability)* averaged over
  the jamming cell only, integrity-degraded share at the
  Faithfulness < 0.65 threshold), a Plotly horizontal-bar chart of
  each cell's *own headline metric* (ASR-t for integrity cells,
  ASR-deny for poiJ) with bootstrap-CI whiskers, a Faithfulness
  overlay histogram (clean vs attacked, threshold line at 0.65), a
  per-cell summary table with the `cell × family × strategy ×
  objective × success_metric` columns, and the last twenty runs as
  a clickable table. Sidebar filters: seeds + attack family +
  attack channel + payload source + verdict (rendered as `st.pills`
  on Streamlit ≥ 1.36).
- **Aggregate** — cell-aware view of the Day-9 manifest: three RAGAS
  metric tiles (clean vs attacked delta), a per-cell summary table
  with bootstrap CIs, split violins of the RAGAS triple per cell, an
  ECDF of `rank_shift@k`, a paired-differences-vs-IPI table with
  Cohen's-*h*, and a RAGAS-by-cell Faithfulness-drop table. Mirrors
  Figures F1–F8 from [scripts/08_make_plots.py](scripts/08_make_plots.py)
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

Opens at <http://localhost:8501>. 
