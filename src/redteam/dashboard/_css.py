"""CSS injection for the Streamlit dashboard.

The light stylesheet is the §5 block from ``DASHBOARD_DESIGN_SYSTEM.md``,
inlined verbatim. The dark overrides use the §2.1 dark palette and
re-bind only the CSS variables — every component selector continues to
read its colours from ``var(--bg-surface)`` etc., so the dark mode is
a single ``:root`` override on top of the light base.

Streamlit doesn't have a "stylesheet per multipage app" hook, so every
page calls :func:`inject_css` once at the top.
"""

from __future__ import annotations

import streamlit as st


_CSS = """
<style>
:root {
  --bg-base: #FAFAF7;
  --bg-surface: #FFFFFF;
  --bg-elevated: #F5F4EE;
  --border: #E5E4DC;
  --border-strong: #C8C7BD;
  --text-primary: #1F1F1B;
  --text-secondary: #5F5E5A;
  --text-tertiary: #888780;
  --accent: #D85A30;
}

html, body, [class*="css"] { color: var(--text-primary); }

.mono, code {
  font-family: "JetBrains Mono", ui-monospace, "SF Mono", Menlo, monospace;
  font-size: 11px;
}

/* ---- Streamlit metric tiles -------------------------------------- */
[data-testid="stMetric"] {
  background: var(--bg-elevated);
  border-radius: 8px;
  padding: 10px 12px;
}
[data-testid="stMetricLabel"] {
  font-size: 10px; text-transform: uppercase; letter-spacing: 0.5px;
  color: var(--text-secondary);
}
[data-testid="stMetricValue"] { font-size: 22px; font-weight: 500; }
[data-testid="stMetricDelta"] { font-size: 11px; }

/* ---- Page header strip ------------------------------------------- */
.dh {
  display: flex; align-items: center; justify-content: space-between;
  padding: 0 4px 12px; border-bottom: 0.5px solid var(--border);
  margin-bottom: 16px;
}
.dh .mt { font-size: 22px; font-weight: 500; margin: 0; }
.dh .mm {
  font-size: 11px; color: var(--text-tertiary);
  font-family: "JetBrains Mono", monospace; margin-top: 2px;
}
.dh .fr { display: flex; gap: 6px; }
.dh .fc {
  font-size: 11px; padding: 4px 10px; border: 0.5px solid var(--border);
  border-radius: 4px; color: var(--text-secondary);
}
.dh .fc.on {
  background: var(--bg-elevated); color: var(--text-primary);
  border-color: var(--border-strong);
}

/* ---- Verdict badges ---------------------------------------------- */
.verdict {
  display: inline-block; padding: 2px 8px; border-radius: 4px;
  font-size: 11px; font-weight: 500;
}
.verdict-success  { background: #FCEBEB; color: #791F1F; }   /* attack succeeded */
.verdict-partial  { background: #FAEEDA; color: #633806; }   /* attack partial   */
.verdict-defended { background: #EAF3DE; color: #27500A; }   /* defence held     */
.verdict-info     { background: #E6F1FB; color: #0C447C; }

/* ---- Doc cards --------------------------------------------------- */
.doc-card {
  display: flex; align-items: flex-start; gap: 10px; padding: 10px;
  border: 0.5px solid var(--border); border-radius: 6px; margin-bottom: 6px;
  background: var(--bg-surface);
}
.doc-card.poisoned { border-color: #E24B4A; background: #FDF5F4; }
/* Baseline top-1 highlight — calm blue tint, designed to read as
 * informational rather than alarming. When a doc is both poisoned
 * AND baseline-top-1 (rare; would mean the attacker's doc_id
 * collides with the gold doc) the poisoned border wins via
 * source-order specificity below. */
.doc-card.baseline-top1 { border-color: #5C8FBE; background: #F0F6FC; }
.doc-card.poisoned.baseline-top1 { border-color: #E24B4A; }
.doc-rank {
  font-family: "JetBrains Mono", monospace; font-size: 11px;
  color: var(--text-secondary); min-width: 22px; text-align: right;
}
.doc-id {
  font-family: "JetBrains Mono", monospace; font-size: 11px; font-weight: 500;
}
.doc-score {
  font-family: "JetBrains Mono", monospace; font-size: 11px;
  color: var(--text-secondary); margin-left: 8px;
}
.doc-flag {
  display: inline-block; font-size: 10px; padding: 1px 6px; border-radius: 3px;
  background: #FCEBEB; color: #791F1F; margin-left: 8px; font-weight: 500;
}
/* Baseline-top-1 chip — re-uses the info palette (light blue + dark
 * navy text) so it reads as informational, not as a warning. */
.doc-flag.doc-flag-baseline {
  background: #E6F1FB; color: #0C447C;
}
.doc-content {
  font-size: 12px; color: var(--text-secondary); margin-top: 6px;
  line-height: 1.5; white-space: pre-wrap;
  font-family: "JetBrains Mono", monospace;
}

/* ---- Score bars (RAGAS triple) ----------------------------------- */
.score-row { margin-bottom: 10px; }
.score-row .lbl {
  display: flex; justify-content: space-between; font-size: 11px;
  margin-bottom: 4px;
  font-family: "JetBrains Mono", monospace;
}
.score-row .lbl .k { color: var(--text-secondary); }
.score-bar-track {
  height: 6px; background: var(--bg-elevated); border-radius: 3px;
  position: relative; overflow: hidden;
}
.score-bar-fill {
  position: absolute; top: 0; bottom: 0; left: 0; border-radius: 3px;
}
.score-bar-fill.good { background: #97C459; }
.score-bar-fill.mid  { background: #EF9F27; }
.score-bar-fill.bad  { background: #E24B4A; }

/* ---- ASR cell grid ----------------------------------------------- */
.asr-grid {
  display: grid; grid-template-columns: 1fr 1fr; gap: 8px;
}
.asr-cell {
  border: 0.5px solid var(--border); border-radius: 6px; padding: 10px 12px;
  background: var(--bg-surface);
}
.asr-cell .k {
  font-size: 10px; text-transform: uppercase; letter-spacing: 0.5px;
  color: var(--text-secondary); font-family: "JetBrains Mono", monospace;
}
.asr-cell .v {
  font-size: 18px; font-weight: 500; margin-top: 4px;
  font-family: "JetBrains Mono", monospace;
}
.asr-cell.hit { border-color: #E24B4A; }
.asr-cell.hit .v { color: #791F1F; }
.asr-cell.miss .v { color: #27500A; }

/* ---- Configuration kv grid --------------------------------------- */
.kv {
  display: grid; grid-template-columns: auto 1fr; gap: 6px 16px;
  font-family: "JetBrains Mono", monospace; font-size: 11px;
}
.kv .k { color: var(--text-secondary); }
.kv .v { color: var(--text-primary); word-break: break-all; }

/* Configuration card wrapper — themed surface used by Run Detail §2.
 * Kept as a class (not an inline style) so the dark override below
 * can re-tint it via the CSS variables, instead of locking it to
 * white-on-light. */
.config-card {
  background: var(--bg-surface);
  border: 0.5px solid var(--border);
  border-radius: 10px;
  padding: 14px;
}

/* ---- Generator output card --------------------------------------- */
.cd {
  background: var(--bg-elevated); border-radius: 6px; padding: 14px 16px;
  font-size: 13px; line-height: 1.55; color: var(--text-primary);
  white-space: pre-wrap;
}

/* ---- Breadcrumb -------------------------------------------------- */
.crumb {
  font-size: 11px; color: var(--text-secondary);
  font-family: "JetBrains Mono", monospace; margin-bottom: 4px;
}
.crumb a { color: var(--text-secondary); text-decoration: none; }
.crumb a:hover { color: var(--accent); }

/* ---- Empty state ------------------------------------------------- */
.empty {
  border: 0.5px dashed var(--border); border-radius: 8px; padding: 40px;
  text-align: center; color: var(--text-tertiary); font-size: 13px;
}
.empty .empty-reason {
  color: var(--text-primary); font-size: 14px; font-weight: 500;
  margin-bottom: 6px;
}
.empty .empty-detail {
  color: var(--text-secondary); font-size: 12px;
  font-family: "JetBrains Mono", monospace; margin-bottom: 16px;
  white-space: pre-wrap;
}
.empty .empty-back { margin-top: 8px; }
.empty .empty-back a {
  color: var(--accent); text-decoration: none; font-size: 12px;
  font-family: "JetBrains Mono", monospace;
}
.empty .empty-back a:hover { text-decoration: underline; }

/* ---- Verdict legend strip --------------------------------------- */
.legend-strip {
  display: flex; align-items: center; gap: 8px; flex-wrap: wrap;
  padding: 8px 12px; margin: 8px 0 16px;
  border: 0.5px solid var(--border); border-radius: 6px;
  background: var(--bg-surface);
  font-size: 11px; font-family: "JetBrains Mono", monospace;
}
.legend-strip .legend-k {
  color: var(--text-tertiary);
  text-transform: uppercase; letter-spacing: 0.5px; font-size: 10px;
  margin-right: 4px;
}
.legend-strip .legend-v { color: var(--text-secondary); }
.legend-strip .legend-sep { color: var(--text-tertiary); margin: 0 4px; }

/* Hide Streamlit chrome we don't need */
#MainMenu, footer { visibility: hidden; }
</style>
"""


_DARK_OVERRIDES = """
<style>
:root {
  --bg-base: #161614;
  --bg-surface: #1F1F1B;
  --bg-elevated: #2A2A26;
  --border: #3A3A35;
  --border-strong: #525248;
  --text-primary: #F1EFE8;
  --text-secondary: #B4B2A9;
  --text-tertiary: #888780;
  --accent: #F0997B;
}

/* ---- Page surfaces ----------------------------------------------- */
/* Streamlit's default app surface is white-on-light. Re-tint everything
 * we can reach via CSS testids; the dataframe widget is canvas-rendered
 * and is fixed by setting STREAMLIT_THEME_BASE=dark in the launcher. */
html, body,
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
[data-testid="stHeader"],
[data-testid="stSidebar"],
[data-testid="stSidebar"] > div,
[data-testid="stSidebarContent"] {
  background: var(--bg-base) !important;
  color: var(--text-primary) !important;
}

/* ---- Markdown / heading text ------------------------------------- */
[data-testid="stMarkdownContainer"],
[data-testid="stMarkdown"],
[data-testid="stHeading"],
[data-testid="stCaptionContainer"],
[data-testid="stText"],
.stMarkdown, .stMarkdown p, .stMarkdown li, .stMarkdown td, .stMarkdown th,
.stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4,
.stMarkdown h5, .stMarkdown h6,
h1, h2, h3, h4, h5, h6, p, span, label, li, td, th, small {
  color: var(--text-primary) !important;
}
/* Captions and helper text stay one tone down for hierarchy. */
[data-testid="stCaptionContainer"],
.stMarkdown small,
.stMarkdown em { color: var(--text-secondary) !important; }
/* The .mm header sub-line + chart caption div stays tertiary. */
.dh .mm, .crumb { color: var(--text-tertiary) !important; }

/* ---- Inline code + monospace ------------------------------------ */
code, pre, .mono {
  background: var(--bg-elevated) !important;
  color: var(--text-primary) !important;
}

/* ---- Streamlit metric tiles ------------------------------------- */
[data-testid="stMetric"] {
  background: var(--bg-elevated) !important;
}
[data-testid="stMetricLabel"] { color: var(--text-secondary) !important; }
[data-testid="stMetricLabel"] p,
[data-testid="stMetricLabel"] div { color: var(--text-secondary) !important; }
[data-testid="stMetricValue"] { color: var(--text-primary) !important; }
[data-testid="stMetricValue"] div { color: var(--text-primary) !important; }
[data-testid="stMetricDelta"] { color: var(--text-secondary) !important; }

/* ---- Sidebar widgets -------------------------------------------- */
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] div { color: var(--text-primary); }

/* ---- Multiselect + pills tag pills ------------------------------ */
[data-baseweb="tag"],
[data-baseweb="tag"] span {
  background: var(--bg-elevated) !important;
  color: var(--text-primary) !important;
}

/* ---- Expander -------------------------------------------------- */
[data-testid="stExpander"] details summary,
[data-testid="stExpander"] details summary * {
  color: var(--text-primary) !important;
}
[data-testid="stExpander"] {
  background: var(--bg-surface) !important;
  border-color: var(--border) !important;
}

/* ---- Buttons, radio, etc. --------------------------------------- */
[data-testid="stRadio"] label,
[data-testid="stRadio"] div { color: var(--text-primary) !important; }

/* ---- Custom HTML helpers (legend strip, doc cards, etc.) -------- */
.verdict-success  { background: #5A1B1B; color: #FCC9C9; }
.verdict-partial  { background: #553011; color: #FAD9A8; }
.verdict-defended { background: #1F3B0D; color: #C5E5A8; }
.verdict-info     { background: #0C2C56; color: #C9DEF6; }

.doc-card { background: var(--bg-surface); color: var(--text-primary); }
.doc-card.poisoned { background: #2E1B1A; border-color: #B23A38; }
.doc-card.baseline-top1 { background: #1A2434; border-color: #5589BD; }
.doc-card.poisoned.baseline-top1 { background: #2E1B1A; border-color: #B23A38; }
.doc-card .doc-content { color: var(--text-secondary); }

/* Baseline-top-1 chip — dark-mode navy bg + light-blue text. */
.doc-flag.doc-flag-baseline {
  background: #0C2C56; color: #C9DEF6;
}

.cd { background: var(--bg-elevated); color: var(--text-primary); }

.empty { background: var(--bg-surface); color: var(--text-secondary); }
.empty .empty-reason { color: var(--text-primary); }
.empty .empty-detail { color: var(--text-secondary); }

.legend-strip { background: var(--bg-surface); color: var(--text-primary); }
.legend-strip .legend-v { color: var(--text-secondary); }
.legend-strip .legend-k { color: var(--text-tertiary); }

.kv .k { color: var(--text-secondary); }
.kv .v { color: var(--text-primary); }

.config-card {
  background: var(--bg-surface) !important;
  border-color: var(--border) !important;
}

.score-row .lbl .k { color: var(--text-secondary); }
.score-row .lbl span { color: var(--text-primary); }
.score-bar-track { background: var(--bg-elevated); }

.asr-cell { background: var(--bg-surface); color: var(--text-primary); }
.asr-cell .k { color: var(--text-secondary); }
</style>
"""


def inject_css(theme: str = "light") -> None:
    """Call once at the top of every dashboard page.

    Parameters
    ----------
    theme:
        ``"light"`` (default) or ``"dark"``. The dark override is a
        second ``<style>`` block layered on top of the light base; it
        re-binds the CSS variables and overrides the few surfaces
        Streamlit hard-codes to white.
    """
    import streamlit as _st  # local import keeps unit tests cheap
    _st.markdown(_CSS, unsafe_allow_html=True)
    if theme.lower() == "dark":
        _st.markdown(_DARK_OVERRIDES, unsafe_allow_html=True)
