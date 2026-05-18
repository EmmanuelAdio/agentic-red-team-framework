# Dashboard Design System — Agentic Red-Team Framework

**Stance:** A (Observability) + C (Research notebook) hybrid.
**Pages:** Overview · Aggregate · Run Detail. No live monitor.
**Stack:** Streamlit.
**Build window:** Day 14–15 (May 17–18) for the minimal version, *only if* the gate at §11 passes. Otherwise Build B post-submission.

This document is the canonical spec. Hand it to Claude Code alongside `DASHBOARD_CLAUDE_CODE_PROMPT.md` and the existing `PROJECT_SPEC.md`.

> **Repo-alignment note (2026-05-09).** The schema, paths, and analysis helpers here are reconciled to the actual repo as of commit `21a2470`: bundles live in `data/runs/batch_<id>/` *and* `results/runs/batch_<id>/`, the bundle schema is the Pydantic model in `src/redteam/bundles/schema.py` (with the §7 additive fields), and a `cell-aware` loader already exists at `src/redteam/analysis/loaders.py`. The dashboard's data layer is a thin façade over those.

---

## 1. Design principles

1. **Density over prettiness.** A user scanning 300 runs needs to see five rows at once, not three.
2. **Mono-space for identity, sans-serif for content.** Hashes, IDs, attack slugs, and metric values are mono. Questions, doc content, and headings are sans.
3. **Colour encodes verdict, not aesthetics.** Red = attack succeeded (alarming for the defender). Amber = partial. Green = defended. Neutral grey = configuration metadata.
4. **Filters over views.** A filter chip on the header replaces a separate Compare page. Comparison is "filter to two seeds and look at the chart" — not a dedicated diff UI.
5. **Every screen reachable in two clicks.** Sidebar nav → page → optional row drill-down. No deeper nesting.
6. **Aesthetic continuity with the dissertation.** The matplotlib figures from `scripts/08_make_plots.py` are the canonical evidence. The dashboard reuses the same verdict palette so a screenshot dropped into Chapter 6 reads as part of the same artefact.

---

## 2. Colour tokens

### 2.1 Base palette

| Token | Hex (light) | Hex (dark) | Use |
| --- | --- | --- | --- |
| `--bg-base` | `#FAFAF7` | `#161614` | page background |
| `--bg-surface` | `#FFFFFF` | `#1F1F1B` | panels, cards |
| `--bg-elevated` | `#F5F4EE` | `#2A2A26` | metric tiles, code blocks |
| `--border` | `#E5E4DC` | `#3A3A35` | 0.5px on most surfaces |
| `--border-strong` | `#C8C7BD` | `#525248` | hover, focus |
| `--text-primary` | `#1F1F1B` | `#F1EFE8` | headings, main copy |
| `--text-secondary` | `#5F5E5A` | `#B4B2A9` | labels, captions |
| `--text-tertiary` | `#888780` | `#888780` | metadata, timestamps |
| `--accent` | `#D85A30` | `#F0997B` | primary action, active nav |

For Day 14–15 ship light only; dark is a Build-B add.

### 2.2 Semantic palette (verdict-aligned)

The bundle's `evaluation.verdict` literal is one of `"success" | "partial" | "failure"` — i.e. the **attack** verdict, not the defence verdict. The CSS class names below name the visual treatment, not the literal, so a `verdict == "failure"` (defence held) maps to the `verdict-defended` class.

| Bundle verdict | Surface (light) | Text | Streamlit equiv | Meaning |
| --- | --- | --- | --- | --- |
| `success` | `#FCEBEB` | `#791F1F` | `error` | red — the attack succeeded end-to-end |
| `partial` | `#FAEEDA` | `#633806` | `warning` | amber — landed in retrieval, not in answer |
| `failure` | `#EAF3DE` | `#27500A` | `success` | green — system defended |
| info / neutral | `#E6F1FB` | `#0C447C` | `info` | blue — configuration, metadata |

These map directly onto Streamlit's native `st.success` / `st.warning` / `st.error` / `st.info` semantic surfaces. **Do not invent new colour roles** — every chip, badge, and bar fill must trace back to one of these four roles plus the base palette.

### 2.3 Optional Loughborough accent

If institutional branding helps in viva or portfolio framing: `#4B0082` on the active sidebar item only. Do not let it bleed into chart fills, badges, or any data-encoding role — it competes with the verdict palette.

---

## 3. Typography

| Role | Font | Size | Weight |
| --- | --- | --- | --- |
| Page title (h1) | Inter / system-sans | 22px | 500 |
| Panel title | Inter | 13px | 500 |
| Body copy | Inter | 14px | 400 |
| Small / labels | Inter | 12px | 400 |
| Uppercase eyebrow | Inter | 10px, letter-spacing 0.5px, uppercase | 500 |
| **Mono** (IDs, hashes, attack slugs, numeric values) | JetBrains Mono / IBM Plex Mono | 11–12px | 400/500 |

**Two weights only.** 400 regular and 500 semi-bold. Never 600+ — it reads heavy against neutral surfaces.

Streamlit ships with its system font. To get Inter and JetBrains Mono cleanly, either set `font="sans serif"` in the theme and inject `@import url(...)` via custom CSS, or accept the system fallback for Day 14–15 and add fonts in Build B.

---

## 4. Spacing, radii, and motion

- **Spacing scale:** 4, 8, 12, 16, 24, 32 (px). Vertical rhythm in 16px increments.
- **Radii:** 4px (chips, badges, score bars), 6px (buttons, inputs, doc cards), 10px (panels, surfaces).
- **Borders:** 0.5px default, 1px on hover/focus, 2px only as a feature accent.
- **Motion:** none. No hover slides, no fade-ins. Streamlit reruns are visible enough.

---

## 5. Streamlit theme configuration

Put this in `dashboard/.streamlit/config.toml`:

```toml
[theme]
base = "light"
primaryColor = "#D85A30"
backgroundColor = "#FAFAF7"
secondaryBackgroundColor = "#F5F4EE"
textColor = "#1F1F1B"
font = "sans serif"
```

Inject this CSS via `st.markdown('<style>...</style>', unsafe_allow_html=True)` once at the top of every page (extract to `src/redteam/dashboard/_css.py`):

```css
.mono, code { font-family: "JetBrains Mono", ui-monospace, "SF Mono", Menlo, monospace; font-size: 11px; }

[data-testid="stMetric"] { background: #F5F4EE; border-radius: 8px; padding: 10px 12px; }
[data-testid="stMetricLabel"] { font-size: 10px; text-transform: uppercase; letter-spacing: 0.5px; color: #5F5E5A; }
[data-testid="stMetricValue"] { font-size: 22px; font-weight: 500; }

.verdict { display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: 500; }
.verdict-success  { background: #FCEBEB; color: #791F1F; }   /* attack succeeded */
.verdict-partial  { background: #FAEEDA; color: #633806; }   /* attack partial   */
.verdict-defended { background: #EAF3DE; color: #27500A; }   /* defence held     */
.verdict-info     { background: #E6F1FB; color: #0C447C; }

.doc-card { display: flex; align-items: flex-start; gap: 10px; padding: 10px; border: 0.5px solid #E5E4DC; border-radius: 6px; margin-bottom: 6px; }
.doc-card.poisoned { border-color: #E24B4A; background: #FDF5F4; }
.doc-rank { font-family: "JetBrains Mono", monospace; font-size: 11px; color: #5F5E5A; min-width: 22px; text-align: right; }
.doc-id   { font-family: "JetBrains Mono", monospace; font-size: 11px; font-weight: 500; }
.doc-score{ font-family: "JetBrains Mono", monospace; font-size: 11px; color: #5F5E5A; margin-left: 8px; }
.doc-flag { display: inline-block; font-size: 10px; padding: 1px 6px; border-radius: 3px; background: #FCEBEB; color: #791F1F; margin-left: 8px; font-weight: 500; }
.doc-content { font-size: 12px; color: #5F5E5A; margin-top: 4px; line-height: 1.5; white-space: pre-wrap; }

.score-bar-track { height: 6px; background: #F5F4EE; border-radius: 3px; position: relative; overflow: hidden; }
.score-bar-fill  { position: absolute; top: 0; bottom: 0; left: 0; border-radius: 3px; }
.score-bar-fill.good { background: #97C459; }
.score-bar-fill.mid  { background: #EF9F27; }
.score-bar-fill.bad  { background: #E24B4A; }

.kv { display: grid; grid-template-columns: auto 1fr; gap: 6px 16px; font-family: "JetBrains Mono", monospace; font-size: 11px; }
.kv .k { color: #5F5E5A; }
.kv .v { color: #1F1F1B; }
```

This single CSS block plus `config.toml` produces 80% of the visual fidelity in the mockups. The remaining 20% you'll trade off against build time.

---

## 6. Component inventory

Each component lists its role, native Streamlit primitive, and the custom CSS it depends on.

| # | Component | Streamlit primitive | Notes |
| --- | --- | --- | --- |
| 1 | Page header | `st.markdown` HTML | breadcrumb + title + filter chips, right-aligned |
| 2 | Metric tile | `st.metric` | with custom CSS in §5 |
| 3 | Verdict badge | `st.markdown` with `.verdict` class | helper: `badge(verdict)` |
| 4 | Run row table | `st.dataframe` with `column_config` | use `LinkColumn` to drill into Run Detail |
| 5 | Filter chips | `st.pills` (Streamlit ≥ 1.36) or `st.multiselect` | sidebar by default, header on Overview |
| 6 | Doc card | `st.markdown` HTML | `.doc-card.poisoned` for the highlighted variant |
| 7 | Score bar | `st.markdown` HTML using `.score-bar-*` | three colour tiers based on threshold |
| 8 | ASR cell | `st.markdown` HTML | small bordered box with mono label + ✓/✗ |
| 9 | Configuration grid | `st.markdown` HTML using `.kv` | 2-col grid, mono keys and values |
| 10 | ASR bar chart with CIs | Plotly `go.Bar` with `error_x` | horizontal bars, CI whiskers from bootstrap |
| 11 | Faithfulness distribution | Plotly `go.Histogram` (overlay) | clean (green) + attacked (red), 50% opacity |
| 12 | RAGAS triple panel | three `st.metric` in `st.columns(3)` | clean vs attacked delta as `delta` |
| 13 | JSON viewer | `st.json` | for raw exploit-bundle inspection |
| 14 | Action button row | `st.button` | inherit Streamlit defaults; trigger callbacks |
| 15 | Empty state | `st.markdown` centred + neutral icon | "No runs match these filters" |
| 16 | Loading state | `st.spinner` | wrap the data load |

Helpers to put in `src/redteam/dashboard/components.py`:

```python
def badge(verdict: str) -> str:
    """Map the bundle's `evaluation.verdict` literal to a CSS class.

    Bundle literals: "success" (attack succeeded), "partial",
    "failure" (defence held). The `verdict-defended` class renders
    `failure` as a green chip.
    """
    cls_map   = {"success": "success", "partial": "partial", "failure": "defended"}
    label_map = {"success": "success", "partial": "partial", "failure": "defended"}
    cls   = cls_map[verdict]
    label = label_map[verdict]
    return f'<span class="verdict verdict-{cls}">{label}</span>'

def doc_card(rank: int, doc_id: str, score: float,
             content: str | None, is_poisoned: bool) -> str:
    """Render one retrieved-doc row.

    `content` is None for clean docs by default — the bundle does not
    carry chunk text (only `doc_id`, `rank`, `score`, `is_poisoned`).
    For the poisoned doc, pass the bundle's `attack.payload` as
    `content` so the reviewer sees what landed in the index. To show
    full text for clean docs, the caller must look the chunk up in
    `data/corpus/` by `doc_id`.
    """
    cls  = "doc-card poisoned" if is_poisoned else "doc-card"
    flag = '<span class="doc-flag">poisoned</span>' if is_poisoned else ""
    body = f'<div class="doc-content">{content}</div>' if content else ""
    return (
        f'<div class="{cls}">'
        f'<span class="doc-rank">#{rank}</span>'
        f'<div style="flex:1;min-width:0">'
        f'<span class="doc-id">{doc_id}</span>'
        f'<span class="doc-score">score {score:.2f}</span>{flag}{body}'
        f'</div></div>'
    )

def score_bar(label: str, value: float | None,
              threshold_good: float = 0.85, threshold_mid: float = 0.65) -> str:
    """RAGAS triple bar. Renders 'n/a' when the scorer returned None."""
    if value is None:
        return (
            f'<div style="margin-bottom:10px">'
            f'<div style="display:flex;justify-content:space-between;font-size:11px;margin-bottom:4px">'
            f'<span style="color:#5F5E5A;font-family:JetBrains Mono,monospace">{label}</span>'
            f'<span style="font-family:JetBrains Mono,monospace;color:#888780">n/a</span></div>'
            f'<div class="score-bar-track"></div></div>'
        )
    tier = "good" if value >= threshold_good else "mid" if value >= threshold_mid else "bad"
    pct  = round(value * 100)
    return (
        f'<div style="margin-bottom:10px">'
        f'<div style="display:flex;justify-content:space-between;font-size:11px;margin-bottom:4px">'
        f'<span style="color:#5F5E5A;font-family:JetBrains Mono,monospace">{label}</span>'
        f'<span style="font-family:JetBrains Mono,monospace">{value:.2f}</span></div>'
        f'<div class="score-bar-track"><div class="score-bar-fill {tier}" style="width:{pct}%"></div></div>'
        f'</div>'
    )

def kv_grid(items: dict[str, str]) -> str:
    """2-col mono grid for Configuration cards."""
    rows = "".join(f'<span class="k">{k}</span><span class="v">{v}</span>'
                   for k, v in items.items())
    return f'<div class="kv">{rows}</div>'
```

---

## 7. Page wireframes

### 7.1 Overview (`dashboard/Home.py`)

```
+-------------------------------------------------------------+
| O Overview                          [retrievers v][7d v]    |
| redteam-rag - last run 2 min ago                            |
+-------------------------------------------------------------+
|                                                             |
| +------+ +------+ +------+ +------+                         |
| | 312  | | 61%  | | 47%  | |  3   |  <- st.metric x 4       |
| | runs | |ASR-t | |integ | |attack|                         |
| +------+ +------+ +------+ +------+                         |
|                                                             |
| +-------------------------+ +-------------------------+     |
| | ASR-t by attack*channel | | Faithfulness clean/atk  |     |
| | [horizontal bars w/ CI] | | [overlay histogram]     |     |
| +-------------------------+ +-------------------------+     |
|                                                             |
| Recent runs                                                 |
| +---------------------------------------------------------+ |
| | run_id     query              family   chan  ASR-t  V   | |
| | …test1195  when did 5th gra…  IPI      corp  1     [r]  | |
| | …test1208  capital of Aus…    QI       qry   0     [g]  | |
| | ...                                                     | |
| +---------------------------------------------------------+ |
```

The chart's grouping is `(attack_family, attack_channel)` — the cross-channel taxonomy in the schema. There are three families (`prompt_injection`, `corpus_poisoning`, `query_injection`) and two channels (`corpus`, `query`); not every combination is populated, so the chart drops empty cells rather than rendering zero-height bars.

### 7.2 Aggregate (`dashboard/pages/01_aggregate.py`)

Build B only. Uses `load_experiment(...)` from `redteam.analysis.loaders` for cell-aware aggregation across the manifest's `cell_registry`. Sketch:

```
Filters             | 300 runs - 3 attacks x 2 channels x 50q x 3 seeds
- attack family     |
- attack channel    | +--ASR-r--+ +--ASR-a--+ +--ASR-t--+
- payload source    | |  84%    | |  72%    | |  61%    |
- seed              | | +-3.1   | | +-4.0   | | +-4.2   |
                    | +---------+ +---------+ +---------+
                    | Faithfulness · Answer Rel · Context Rel  (violins)
                    | Rank-shift@5 distribution (histogram)
                    | Per-cell table — uses `cell_registry` labels
```

### 7.3 Run Detail (`dashboard/pages/02_run_detail.py?run_id=...`)

Sections, top to bottom:

1. **Header strip** — breadcrumb · query · verdict badge (mapped through the table in §2.2: `success → red`, `partial → amber`, `failure → green`).
2. **Configuration card** — kv grid pulling from `target_system.*` + `attack.family/strategy/payload_source/attack_channel` + `seed` + `attack.iteration` + `execution.index_state_hash`.
3. **Retrieved documents** — vertical doc-card list. Render the poisoned doc with `is_poisoned=True` and `content=bundle.attack.payload` (full text auto-expanded). Clean docs render with `content=None`; offer a "Load chunk" affordance that reads from `data/corpus/` by `doc_id` if Build A time allows, otherwise leave the link as Future Work.
4. **Generator output card** — `execution.generator_output` rendered as plain text inside a `.cd` div (sans-serif, no Markdown interpretation).
5. **Evaluation card** — two columns. Left: three `score_bar` calls for the RAGAS triple, each handling `None` gracefully. Right: 2x2 grid of ASR cells (`asr_retrieval`, `asr_answer`, `asr_target`, `rank_shift_at_k`) plus an optional `asr_deny` row.
6. **Iteration history** *(collapsed expander, optional)* — a small table built from `evaluation.iteration_history` so reviewers can see how the planner adapted across iterations within this run.
7. **Action row** — three buttons. Only one needs to work in Build A: `st.download_button` for the raw bundle JSON. Stub the others with `st.toast("Not implemented in Build A")`.

Routing: pass `run_id` via `st.query_params`. Bookmarkable URLs are essential for the viva demo — you'll want to deep-link to specific failures.

---

## 8. Data layer

The dashboard reads exploit-bundle JSONs from the two run roots the project actually writes to:

* `data/runs/batch_<batch_id>/run_<query_id>_<batch_id>_bundle.json` — Day-8 dry-run bundles.
* `results/runs/batch_<batch_id>/run_<query_id>_<batch_id>_bundle.json` — Day-9 full-experiment bundles, plus `experiment_manifest.json` and per-batch `*_summary.json` rollups.

Both roots share the same nested layout enforced by `redteam.bundles.store.BundleStore`. Don't read either tree on every page load — that's slow at 300+ files.

```python
# src/redteam/dashboard/data.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import streamlit as st

from redteam.config import EXPERIMENT_RUNS_DIR, RUNS_DIR


def _iter_bundle_paths(roots: Iterable[Path]) -> list[Path]:
    """Recursively find every `run_*_bundle.json` under each root.

    Skips `*_summary.json` rollups (different schema) and any `*.tmp`
    sidecars left behind by interrupted writes.
    """
    paths: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        for p in root.glob("**/run_*_bundle.json"):
            if p.name.endswith(".tmp"):
                continue
            paths.append(p)
    return sorted(paths)


@st.cache_data(ttl=300)
def load_bundles(*roots: Path) -> pd.DataFrame:
    """Load every bundle under `roots` into a flat DataFrame.

    Defaults to (`EXPERIMENT_RUNS_DIR`, `RUNS_DIR`) when called with
    no args, so the Overview page sees Day-9 results first and falls
    back to the Day-8 dry-run bundles when the experiment matrix is
    not yet written.
    """
    roots = roots or (EXPERIMENT_RUNS_DIR, RUNS_DIR)
    rows: list[dict] = []
    for path in _iter_bundle_paths(roots):
        with open(path, encoding="utf-8") as fh:
            b = json.load(fh)
        s = b.get("summary", {})  # headline projection — see schema.py BundleSummary
        rows.append({
            "run_id":          b["run_id"],
            "timestamp":       b["timestamp_utc"],
            "seed":            b["seed"],
            "batch_id":        path.parent.name.removeprefix("batch_"),
            "query":           b["execution"]["query"],
            "query_id":        b["execution"]["query_id"],
            "attack_family":   b["attack"]["family"],
            "attack_strategy": b["attack"]["strategy"],
            "attack_channel":  b["attack"]["attack_channel"],
            "payload_source":  b["attack"]["payload_source"],
            "iteration":       b["attack"]["iteration"],
            "embedding_model": b["target_system"]["embedding_model"],
            "llm_model":       b["target_system"]["llm_model"],
            "asr_r":           bool(b["evaluation"]["asr_retrieval"]),
            "asr_a":           bool(b["evaluation"]["asr_answer"]),
            "asr_t":           bool(b["evaluation"]["asr_target"]),
            "asr_deny":        b["evaluation"].get("asr_deny"),
            "faithfulness":    b["evaluation"].get("ragas_faithfulness"),
            "answer_rel":      b["evaluation"].get("ragas_answer_relevance"),
            "context_rel":     b["evaluation"].get("ragas_context_relevance"),
            "rank_shift":      b["evaluation"]["rank_shift_at_k"],
            "verdict":         b["evaluation"]["verdict"],
            "latency_ms":      b["execution"]["generator_latency_ms"],
            "_path":           str(path),
        })
    return pd.DataFrame(rows)


@st.cache_data(ttl=300)
def load_one_bundle(path: str) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def bootstrap_ci(values: np.ndarray, n_resamples: int = 1000,
                 ci: float = 0.95) -> tuple[float, float, float]:
    rng = np.random.default_rng(42)
    means = np.array([rng.choice(values, size=len(values), replace=True).mean()
                      for _ in range(n_resamples)])
    lo, hi = np.quantile(means, [(1 - ci) / 2, 1 - (1 - ci) / 2])
    return float(values.mean()), float(lo), float(hi)
```

> **Reuse the existing analysis loader for Aggregate.** `src/redteam/analysis/loaders.py` already exposes `load_experiment()` which reads `experiment_manifest.json`, the per-batch summaries, and the cell registry into typed DataFrames. The Build-A Overview deliberately sidesteps it (it just wants every bundle in one frame), but the Build-B Aggregate page must call `load_experiment(EXPERIMENT_RUNS_DIR)` rather than re-implement cell joining.

Cache TTL of 5 minutes is enough for batch experiments where you re-run, then refresh the dashboard. For Build B, swap to DuckDB over the JSON tree for sub-second filters at scale.

---

## 9. Build A — Day 14–15 minimal (~12–16 productive hours)

If, and only if, the gate at §11 passes by 23:59 Saturday May 16, you may build:

- **One** Streamlit app, two pages: `Overview` and `Run Detail`. **Drop the Aggregate page** — its content lives on Overview as filters.
- Theme via `config.toml` and the CSS block in §5.
- Components 1, 2, 3, 4, 6, 9, 10, 13, 14 from §6. Skip the rest.
- One Plotly chart on Overview: ASR-t bars with CI whiskers, grouped by `(attack_family, attack_channel)`.
- No filtering except a single seed multiselect in the sidebar.
- Read bundles from disk via `load_bundles(EXPERIMENT_RUNS_DIR, RUNS_DIR)`. No DuckDB.
- Two screenshots in the dissertation: Overview page and Run Detail page. Adds one figure to Chapter 6 (Results) showing the dashboard rendering a real exploit. **Do not let dashboard screenshots replace your matplotlib results figures** — those are still the canonical evidence.

**Deliberate cuts:** Aggregate page, comparison views, filter pills, dark mode, fonts beyond system, JSON download as ZIP, "re-run" button, Loughborough purple, sidebar collapse, mobile responsive, clean-doc chunk lookup against `data/corpus/`.

If at any point during Day 14–15 you are behind on the dashboard but ahead on Conclusion + Abstract, **stop the dashboard, write the Conclusion**. The Conclusion chapter is graded; the dashboard is bonus.

---

## 10. Build B — post-submission full

After May 20:

- Full Aggregate page using `redteam.analysis.loaders.load_experiment` with violin/histogram per metric and the cell-registry-aware per-condition table.
- Sidebar filters with `st.pills` for attack family + attack channel + payload source + verdict, plus seed multiselect.
- DuckDB query layer over `results/runs/**/*.json` for sub-second filters.
- Faithfulness overlay histogram on Overview, joined against `results/baseline/baseline_latest.json` for the clean reference.
- Inter and JetBrains Mono font import.
- Dark mode (Streamlit `base = "dark"`).
- Loughborough purple as the active-nav accent (subtle, opt-in via env var).
- Three-pane Run Detail comparison (pin two runs, diff their configs and outputs).
- Iteration-history timeline (uses `evaluation.iteration_history`).
- Export-bundle-as-ZIP for sharing reproducible failures.
- Deploy to Streamlit Community Cloud or Hugging Face Spaces with a public URL on your CV / portfolio.

Build B is approximately 3 days of focused work on top of Build A. Do it the week of June 1, after viva.

---

## 11. The gate — must pass by 23:59 Sat 16 May (Day 13)

Build A is permitted *only if*, by end of Saturday May 16:

1. ✅ Chapter 4 (Methodology) — first draft complete (≥1500 words, all sections present).
2. ✅ Chapter 5 (Experimentation) — first draft complete.
3. ✅ Chapter 6 (Results) — first draft complete with at least the headline ASR-t and Faithfulness Δ figures embedded.
4. ✅ Chapter 7 (Discussion) — first draft complete, mapped to RQ1–RQ4.
5. ✅ All experiments have run; `results/runs/` (or `data/runs/`) has ≥150 bundles.

Plus the residual writing budget for May 17–18 must be confirmed achievable:

6. ✅ Chapter 8 (Conclusion), Abstract, and the Chapter 3 rewrite can fit into ~6 hours of Day 14 morning.

If any of (1)–(6) is not green by midnight Saturday, you do not build the dashboard. You write. The matplotlib figures from `scripts/08_make_plots.py` already get you full marks on Practical Abilities; the dashboard adds nothing the markers haven't already credited.

If the gate passes: kick off Build A by Sunday May 17 morning. Submission Tuesday May 20 EOD.

---

## 12. File layout (additions to the existing repo)

```
src/redteam/dashboard/
├── __init__.py
├── _css.py              # CSS injection helper
├── components.py        # badge(), doc_card(), score_bar(), kv_grid()
├── data.py              # load_bundles(), load_one_bundle(), bootstrap_ci()
└── charts.py            # Plotly factories: asr_bar_chart(), faith_hist()

dashboard/
├── Home.py              # Overview page (Streamlit entry point)
├── pages/
│   ├── 01_aggregate.py  # Build B only - reuses redteam.analysis.loaders
│   └── 02_run_detail.py
└── .streamlit/
    └── config.toml

scripts/
└── 09_run_dashboard.sh  # streamlit run dashboard/Home.py
                          # — numbered 09 because 01-08 are already taken
```

Run with: `streamlit run dashboard/Home.py --server.port=8501`.

---

## 13. Acceptance criteria for Build A

By 18:00 Monday May 18 you must be able to:

- [ ] `streamlit run dashboard/Home.py` opens to the Overview page with real data drawn from `results/runs/` (and `data/runs/` as fallback).
- [ ] Overview shows 4 metric tiles, the ASR-t bar chart grouped by `(attack_family, attack_channel)`, and the recent-runs table — all populated from the bundle tree.
- [ ] Clicking a row navigates to Run Detail with the correct `run_id`.
- [ ] Run Detail renders configuration, retrieved docs (poisoned highlighted with `attack.payload` shown), generator output, evaluation, and one working button (`st.download_button` for the raw bundle JSON).
- [ ] One screenshot of each page is added to Chapter 6 of the dissertation.
- [ ] Repo README has a "Dashboard" section with a screenshot and the run command.
- [ ] No functionality regression in `scripts/08_make_plots.py` — the matplotlib path still works.

If any are red at 18:00 Monday, cut to "Overview only" and ship that.
