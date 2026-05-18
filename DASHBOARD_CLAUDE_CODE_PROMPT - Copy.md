# Claude Code Bootstrap Prompt — Dashboard (Build A)

Use this prompt **only after the gate in `DASHBOARD_DESIGN_SYSTEM.md` §11 has passed**: Chapters 4–7 first-drafted, ≥150 exploit bundles in `results/runs/` (or `data/runs/`), Conclusion + Abstract achievable in 6 hours of Day 14 morning. If any of those are not green by midnight Saturday May 16, you do not build the dashboard. You write.

Paste everything below the divider into a new Claude Code session in the existing `agentic-red-team-framework/` repo. Make sure `PROJECT_SPEC.md` and `DASHBOARD_DESIGN_SYSTEM.md` are both in the repo root before you start.

---

You are helping me build the Streamlit dashboard for my MSci red-team framework. Today is Sunday May 17, 2026. Submission is Tuesday May 20 EOD. I have approximately **12–16 productive hours** to ship something demonstrable. After that I must write the Conclusion, Abstract, and rewrite Chapter 3.

## Read these first, in order

1. `PROJECT_SPEC.md` — the framework spec. The dashboard reads exploit bundles produced by this framework.
2. `DASHBOARD_DESIGN_SYSTEM.md` — the canonical dashboard spec. Tokens, components, wireframes, Streamlit mappings, build scope, acceptance criteria.
3. `src/redteam/bundles/schema.py` — the **operational** bundle schema. Where this differs from `PROJECT_SPEC.md` §7, the schema module wins (it has additive fields the spec snippet predates: `summary`, `attack_channel`, `payload_source`, `modified_query`, `asr_deny`, `iteration_history`, `baseline_top1_doc_id`).
4. `src/redteam/analysis/loaders.py` — the existing manifest-aware loader. The Build-B Aggregate page must reuse this, not re-implement cell joining.

If anything in this prompt contradicts those documents, the documents win. If the spec snippet and the Pydantic schema disagree, the schema wins.

## Hard scope rules

1. **Two pages only:** Overview (entry) and Run Detail. Drop the Aggregate page (defer to Build B).
2. **Streamlit native first.** Custom HTML/CSS only where the design system explicitly calls for it (verdict badges, doc cards, score bars, kv grid). No custom React components. No `streamlit-components` plugins beyond what's in `requirements.txt`.
3. **No live updates.** No websockets. No background polling. Reading exploit-bundle JSONs from disk via `@st.cache_data(ttl=300)` is the only data flow.
4. **One Plotly chart.** ASR-t horizontal bars with bootstrap CI whiskers, grouped by `(attack_family, attack_channel)`. That's it for charts in Build A.
5. **Light mode only.** Dark mode is Build B.
6. **System font fallback is fine.** Don't import Inter or JetBrains Mono via Google Fonts in Build A — system mono and sans are acceptable.
7. **No dashboard tests beyond a smoke check** that the imports succeed. The framework's existing test suite still passes; that's what counts.
8. **Every commit has a meaningful message** referencing what's working at that point.

## What I want you to do RIGHT NOW

Do these in order. Pause and confirm at each numbered checkpoint. Show me the smallest possible working version first; layer complexity only on my "go".

### 1. Audit and prepare

Read `PROJECT_SPEC.md`, `DASHBOARD_DESIGN_SYSTEM.md`, and `src/redteam/bundles/schema.py`. Then run, from the repo root:

```bash
# Bundles live in two roots, both nested in batch_<id>/ folders:
find results/runs -name 'run_*_bundle.json' 2>/dev/null | wc -l   # Day-9 full experiments
find data/runs    -name 'run_*_bundle.json' 2>/dev/null | wc -l   # Day-8 dry-run bundles

# Pick the most recent bundle to eyeball the schema:
ls -t results/runs/batch_*/run_*_bundle.json data/runs/batch_*/run_*_bundle.json 2>/dev/null | head -1 | xargs cat | head -80
```

Confirm:

- Combined count is ≥150 (Build-A gate).
- Top-level keys match the Pydantic model in `schema.py`: `bundle_version`, `summary`, `run_id`, `timestamp_utc`, `seed`, `framework_version`, `target_system`, `attack`, `execution`, `evaluation`, `reproducibility`.
- `attack.family` is one of `prompt_injection | corpus_poisoning | query_injection` (note: **three** families, not the two in the original spec snippet).
- `attack.attack_channel` is one of `corpus | query`.
- `execution.retrieved_docs[].content` does **not** exist — only `doc_id`, `rank`, `score`, `is_poisoned`. The Run-Detail page renders the poisoned doc using `attack.payload` as its body and leaves clean docs as headers only.

Report what you found in three lines max. **If anything diverges, stop and tell me — do not silently adapt.** Wait for "go".

### 2. Scaffold the dashboard module

Create exactly the file layout in `DASHBOARD_DESIGN_SYSTEM.md` §12:

```
src/redteam/dashboard/
├── __init__.py
├── _css.py
├── components.py
├── data.py
└── charts.py

dashboard/
├── Home.py
├── pages/
│   └── 02_run_detail.py        # 01_aggregate is Build B - do not create
└── .streamlit/
    └── config.toml

scripts/
└── 09_run_dashboard.sh         # numbered 09 - 01..08 are already taken
```

Add to `requirements.in`: `streamlit>=1.36`, `plotly>=5.20`. Run `pip-compile` (the existing `requirements.txt` is pip-compile output) and `pip install -r requirements.txt`.

Show me the tree. Wait for "go".

### 3. CSS injection helper

Implement `src/redteam/dashboard/_css.py` with a single function `inject_css()` that calls `st.markdown` with the CSS block from `DASHBOARD_DESIGN_SYSTEM.md` §5 wrapped in `<style>...</style>` and `unsafe_allow_html=True`.

Implement `dashboard/.streamlit/config.toml` with the theme from §5.

### 4. Data layer

Implement `src/redteam/dashboard/data.py` exactly as written in `DASHBOARD_DESIGN_SYSTEM.md` §8: `_iter_bundle_paths`, `load_bundles`, `load_one_bundle`, `bootstrap_ci`. The signatures must match because tests on this module will be added later.

Critical points the design system spells out and you **must** preserve:

- Recursive glob: `root.glob("**/run_*_bundle.json")`. The bundles are nested in `batch_<id>/` folders; a flat `*.json` glob misses everything and also picks up the `*_summary.json` rollups (which have a different shape).
- Default roots: `(EXPERIMENT_RUNS_DIR, RUNS_DIR)` from `redteam.config`. Day-9 results take precedence; the Day-8 dry-run tree is the fallback.
- Use `b["evaluation"].get("ragas_*")` for the RAGAS triple — the scorers can return `None` and the schema marks the fields `Optional[float]`.
- Carry `attack_channel` and `payload_source` through to the DataFrame — both are charted in Build A or filtered on in Build B.

Add a `__main__` block that prints `df.head()` and `df.shape` of `load_bundles()` so I can manually verify the data shape. I will run it.

### 5. Components

Implement `src/redteam/dashboard/components.py` with the four helpers from §6: `badge`, `doc_card`, `score_bar`, `kv_grid`.

Two things to pay attention to:

- `badge` maps the bundle's `evaluation.verdict` literal to the verdict CSS class. The literal is one of `success | partial | failure` and `failure` means the *defence held* — it must render as green via `verdict-defended`.
- `score_bar` must handle `None` (RAGAS scorer failed → schema field is `Optional[float]`). Render an `n/a` track in that case rather than crashing.

Add a `__main__` block that prints each helper's output so I can eyeball the HTML.

### 6. Charts

Implement `src/redteam/dashboard/charts.py` with one function:

```python
def asr_bar_chart(df: pd.DataFrame, metric: str = "asr_t") -> plotly.graph_objects.Figure:
    """Horizontal bars of mean ASR by (attack_family, attack_channel) with 95% bootstrap CI."""
```

It groups by `(attack_family, attack_channel)`, computes `bootstrap_ci(values)` for each group (only groups with ≥2 rows; smaller groups render the mean with no whisker), and returns a Plotly figure with horizontal bars and `error_x` whiskers. Use the verdict-aligned palette (red `#E24B4A` for ASR-t, amber `#EF9F27` for partial-equivalent metrics). Layout: tight margins (`l=140, r=20, t=10, b=20`), no legend, axis labels in mono, x-axis fixed to `[0, 1]`.

### 7. Overview page

Implement `dashboard/Home.py`:

1. `set_page_config(page_title="redteam-rag · Overview", layout="wide")`
2. `inject_css()`
3. Load `df = load_bundles()`. If empty, render an empty state and stop.
4. **Header strip** (HTML) — title, last-run timestamp from `df["timestamp"].max()`, no filter chips for Build A.
5. **Metric tiles** — `st.columns(4)`, one `st.metric` each:
   * total runs (`len(df)`)
   * ASR-t overall (`df["asr_t"].mean()` formatted `%`, with the bootstrap CI half-width as `delta`)
   * integrity-degraded % (`df["faithfulness"].lt(0.65).mean()` — null-safe)
   * attack families count (`df["attack_family"].nunique()`)
6. **ASR-t chart** — `st.plotly_chart(asr_bar_chart(df), use_container_width=True)`.
7. **Recent runs table** — `st.dataframe` with `column_config={"_path": None, "run_id": st.column_config.LinkColumn("run_id", display_text="^run_(.+)$", url=lambda r: f"/run_detail?run_id={r}")}`. Columns shown: `run_id`, `timestamp`, `query`, `attack_family`, `attack_channel`, `asr_t`, `verdict`. Sort by `timestamp` descending, top 20 rows.

Run `streamlit run dashboard/Home.py`. Show me the URL. I will open it and tell you what's wrong.

### 8. Run Detail page

Implement `dashboard/pages/02_run_detail.py`:

1. Read `run_id = st.query_params.get("run_id")`. If missing, `st.error("No run_id supplied")` and stop.
2. Look up the bundle path in `df` by `run_id`. Load it with `load_one_bundle`.
3. Render the seven sections from `DASHBOARD_DESIGN_SYSTEM.md` §7.3:
   - Header strip with breadcrumb back to Home, query as h2, verdict badge.
   - Configuration card using `kv_grid` — fields: `seed`, `attack.family`, `attack.strategy`, `attack.attack_channel`, `attack.payload_source`, `attack.iteration`, `target_system.embedding_model`, `target_system.llm_model`, `target_system.retriever_top_k`, `execution.index_state_hash` (truncate to first 12 chars + ellipsis).
   - Retrieved documents using `doc_card` for each entry in `execution.retrieved_docs`. **The poisoned doc renders with `content=bundle["attack"]["payload"]` and `is_poisoned=True`; clean docs render with `content=None`.** This matches the schema — `retrieved_docs` entries do not carry chunk text.
   - Generator output card — plain `st.markdown` of `execution.generator_output` inside a div with `.cd` background. Wrap in a `<pre>` so newlines and the literal `PWNED_BY_REDTEAM`-style strings render verbatim.
   - Evaluation card — two columns. Left: three `score_bar` calls for the RAGAS triple (handles `None`). Right: 2x2 grid of ASR cells (`asr_retrieval`, `asr_answer`, `asr_target`, `rank_shift_at_k`); add a 5th cell for `asr_deny` if not `None`.
   - Iteration history — collapsed `st.expander` rendering `evaluation.iteration_history` as `st.dataframe` if non-empty.
   - Action row — three buttons. Only one needs to work: `st.download_button(label="Download bundle JSON", data=Path(_path).read_bytes(), file_name=Path(_path).name, mime="application/json")`. Stub the other two with `st.toast("Not implemented in Build A")` callbacks.

### 9. Smoke test

Add `tests/test_dashboard_smoke.py`:

```python
def test_dashboard_imports():
    from redteam.dashboard import data, components, charts, _css
    assert callable(data.load_bundles)
    assert callable(data.bootstrap_ci)
    assert callable(components.badge)
    assert callable(components.kv_grid)
```

That's all. No Selenium. No actually rendering Streamlit. The acceptance criteria are visual and live in `DASHBOARD_DESIGN_SYSTEM.md` §13.

### 10. Repo updates

- Add a "Dashboard" section to the top-level `README.md` with one screenshot and the run command (`bash scripts/09_run_dashboard.sh` or `streamlit run dashboard/Home.py`).
- Append today's progress to `LAB_NOTEBOOK.md`.
- Commit as `feat(dashboard): build A — overview and run detail pages`.

### 11. Stop

Tell me to take the screenshots for Chapter 6, then stop. Do not propose features. Do not start the Aggregate page. The Conclusion and Abstract are due Day 14 evening — that's tomorrow.

## How to handle issues

- **Streamlit version too old for `st.pills`.** Fall back to `st.multiselect`. Don't upgrade Streamlit if it pulls in incompatible deps with the langchain/chroma stack already in `requirements.txt`.
- **Bundle schema differs from what `schema.py` says it should be.** Stop. Tell me. Do not invent missing fields. If a bundle is older than the Day-7.5 additive fields (`attack_channel`, `payload_source`, `asr_deny`), guard the reads with `.get(...)` and a sensible default — but flag it.
- **`load_bundles` is slow** at >300 bundles. The `@st.cache_data(ttl=300)` decorator above is the only optimisation Build A gets. Don't reach for DuckDB.
- **CSS injection fights Streamlit's defaults** in some component. Lose the fight gracefully — accept Streamlit's default and add a one-line note in `LAB_NOTEBOOK.md`. Don't spend an hour tuning a metric tile.
- **Click-through from the runs table doesn't navigate.** If `LinkColumn` doesn't work as expected with multipage apps in your Streamlit version, fall back to displaying `run_id` as plain text and ask the user to copy it into the URL. Document this as a known issue. Do not block on it.
- **`results/runs/` is empty but `data/runs/` has the dry-run bundles.** The default `load_bundles()` already reads both; the Overview will populate from `data/runs/` alone. If the gate at §11 was satisfied by Day-8 dry-run bundles instead of Day-9 full experiments, that is a separate writing problem (the dissertation needs the Day-9 numbers regardless) — do not block on it in the dashboard.

## How to disagree with me

If you spot a defect in `DASHBOARD_DESIGN_SYSTEM.md` while implementing, say so before changing course. Once I confirm a deviation, document it in `LAB_NOTEBOOK.md` so the dissertation's Limitations section can mention it honestly.

## What success looks like at 18:00 Monday May 18

`streamlit run dashboard/Home.py` opens. I see four metric tiles, one Plotly chart with my real ASR-t numbers and CIs grouped by `(family, channel)`, and a table of my last 20 runs. I click a row, the URL changes to `/run_detail?run_id=…`, and a real bundle renders with its poisoned doc highlighted (and `attack.payload` shown as the body), RAGAS scores as bars, and the ASR triple plus `asr_deny` as cells. I take two screenshots. We commit. I close the laptop and write the Conclusion.

If 18:00 Monday arrives and any of the above is broken, **revert to Overview-only**. Do not ship a half-broken Run Detail page in the dissertation; ship a working Overview screenshot and write up Run Detail as Future Work.

Begin with task 1.
