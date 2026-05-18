"""Run Detail page — single-bundle inspection.

Routing: ``?run_id=<run_id>``. The page looks the bundle path up in the
in-memory DataFrame and reads the JSON via :func:`load_one_bundle`.

See ``DASHBOARD_DESIGN_SYSTEM.md`` §7.3 for the section-by-section spec.
"""

from __future__ import annotations

import html as _html
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from redteam.dashboard._css import inject_css  # noqa: E402
from redteam.dashboard.charts import current_theme  # noqa: E402

_THEME = current_theme()
from redteam.dashboard.components import (  # noqa: E402
    asr_cell,
    asr_grid,
    badge,
    doc_card,
    empty_state,
    kv_grid,
    score_bar,
)
from redteam.dashboard.data import load_bundles, load_one_bundle  # noqa: E402


# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="redteam-rag · Run Detail",
    # page_icon="🔴",
    layout="wide",
    initial_sidebar_state="collapsed",
)
inject_css(theme=_THEME)


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------

run_id = st.query_params.get("run_id")
if isinstance(run_id, list):  # older Streamlit versions return a list
    run_id = run_id[0] if run_id else None

if not run_id:
    st.markdown(
        empty_state(
            reason="No run_id supplied.",
            detail=(
                "Run Detail expects ?run_id=<id>. Open any row from the "
                "Overview page, or paste a run_id manually:\n"
                "    /run_detail?run_id=run_<query_id>_<batch_id>"
            ),
        ),
        unsafe_allow_html=True,
    )
    st.stop()

with st.spinner("Looking up bundle..."):
    df = load_bundles()

match = df[df["run_id"] == run_id]
if match.empty:
    st.markdown(
        empty_state(
            reason=f"No bundle found for run_id={run_id}.",
            detail=(
                "Searched recursively under results/runs/ (Day-9 full matrix)\n"
                "and data/runs/ (Day-8 dry-run bundles). The id may belong to\n"
                "a deleted batch, or you may need to wait for the 5-minute\n"
                "cache TTL to refresh after a new experiment write."
            ),
        ),
        unsafe_allow_html=True,
    )
    st.stop()

row = match.iloc[0]
bundle_path = row["_path"]
bundle = load_one_bundle(bundle_path)

attack = bundle["attack"]
execution = bundle["execution"]
evaluation = bundle["evaluation"]
target = bundle["target_system"]


# ---------------------------------------------------------------------------
# Section 1 — Header strip
# ---------------------------------------------------------------------------

st.markdown(
    '<div class="crumb"><a href="./">← Overview</a> · '
    f'run <span class="mono">{run_id}</span></div>',
    unsafe_allow_html=True,
)

verdict = evaluation["verdict"]
query_text = execution["query"]
st.markdown(
    f'<div class="dh">'
    f'<div><h2 class="mt">{query_text}</h2>'
    f'<div class="mm">'
    f'query_id={execution["query_id"]} · seed={bundle["seed"]} · '
    f'batch={row["batch_id"]} · iter={attack.get("iteration", 0)}'
    f"</div></div>"
    f"<div>{badge(verdict)}</div>"
    f"</div>",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Section 2 — Configuration card
# ---------------------------------------------------------------------------


def _trunc(s: str, n: int = 18) -> str:
    return s if len(s) <= n else s[:n] + "…"


config_items = {
    "seed": str(bundle["seed"]),
    "framework_version": bundle["framework_version"],
    "attack.family": attack["family"],
    "attack.strategy": attack["strategy"],
    "attack.channel": attack.get("attack_channel", "—"),
    "attack.payload_source": attack.get("payload_source", "—"),
    "attack.iteration": str(attack.get("iteration", 0)),
    "attack.payload_id": attack["payload_id"],
    "embedding_model": target["embedding_model"],
    "vector_store": target["vector_store"],
    "retriever_top_k": str(target["retriever_top_k"]),
    "llm_model": target["llm_model"],
    "llm_temperature": str(target["llm_temperature"]),
    "prompt_template_hash": _trunc(target["prompt_template_hash"]),
    "execution.index_state_hash": _trunc(execution["index_state_hash"]),
    "execution.baseline_top1_doc_id": str(execution.get("baseline_top1_doc_id") or "—"),
    "generator_latency_ms": f"{execution['generator_latency_ms']:.1f}",
}

st.markdown(
    '<h3 style="font-size:13px;font-weight:500;margin:16px 0 6px">'
    "Configuration</h3>",
    unsafe_allow_html=True,
)
st.markdown(
    f'<div class="config-card">{kv_grid(config_items)}</div>',
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Section 3 — Retrieved documents
# ---------------------------------------------------------------------------

st.markdown(
    '<h3 style="font-size:13px;font-weight:500;margin:24px 0 6px">'
    f"Retrieved documents (top-{target['retriever_top_k']})</h3>",
    unsafe_allow_html=True,
)
st.caption(
    "Bundle entries carry only `doc_id`, `rank`, `score`, `is_poisoned`. "
    "Poisoned content is rendered from `attack.payload`. Clean-doc text "
    "lives in `data/corpus/` and is not loaded in Build A."
)

baseline_top1_id = execution.get("baseline_top1_doc_id")
baseline_in_topk = False
cards = []
for d in execution["retrieved_docs"]:
    is_poisoned = bool(d.get("is_poisoned"))
    is_baseline = bool(baseline_top1_id and d.get("doc_id") == baseline_top1_id)
    if is_baseline:
        baseline_in_topk = True
    content = attack["payload"] if is_poisoned else None
    # rank_shift for the baseline doc = (attacked_rank − 1). The bundle's
    # global ``evaluation.rank_shift_at_k`` records the same value at the
    # run level; computing it here keeps the chip self-contained per card.
    cards.append(
        doc_card(
            rank=d["rank"],
            doc_id=d["doc_id"],
            score=float(d["score"]),
            content=content,
            is_poisoned=is_poisoned,
            is_baseline_top1=is_baseline,
            baseline_rank_shift=(int(d["rank"]) - 1) if is_baseline else None,
        )
    )
st.markdown("".join(cards), unsafe_allow_html=True)

# If the clean-baseline top-1 doc isn't in this attacked top-k at all,
# render an inline notice. rank_shift_at_k = k is the encoded "fell out"
# sentinel (see redteam/metrics/rank_shift.py); we surface it explicitly
# so the reader doesn't have to cross-reference the evaluation card.
if baseline_top1_id and not baseline_in_topk:
    rs = evaluation.get("rank_shift_at_k")
    k = target.get("retriever_top_k")
    st.markdown(
        '<div class="empty" style="text-align:left;padding:14px 16px;'
        'margin-top:8px;border-style:solid">'
        '<span class="doc-flag doc-flag-baseline">baseline top-1 evicted</span>'
        '<span style="margin-left:10px;font-size:12px">'
        f"The clean-baseline top-1 doc <code>{_html.escape(str(baseline_top1_id))}</code> "
        f"is <strong>not</strong> in this attacked top-{k} retrieval "
        f"(rank_shift@k = {rs}, the saturating max — interpret as "
        "&ldquo;fell out of top-k entirely&rdquo;)."
        "</span></div>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Section 4 — Generator output
# ---------------------------------------------------------------------------

st.markdown(
    '<h3 style="font-size:13px;font-weight:500;margin:24px 0 6px">'
    "Generator output</h3>",
    unsafe_allow_html=True,
)
st.markdown(
    f'<div class="cd">{_html.escape(execution["generator_output"])}</div>',
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Section 5 — Evaluation card (RAGAS triple + ASR cells)
# ---------------------------------------------------------------------------

st.markdown(
    '<h3 style="font-size:13px;font-weight:500;margin:24px 0 6px">'
    "Evaluation</h3>",
    unsafe_allow_html=True,
)

left, right = st.columns([1, 1])
with left:
    st.markdown(
        '<div style="font-size:11px;color:#888780;margin-bottom:6px;'
        'text-transform:uppercase;letter-spacing:0.5px">RAGAS triple</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        score_bar("faithfulness", evaluation.get("ragas_faithfulness")),
        unsafe_allow_html=True,
    )
    st.markdown(
        score_bar("answer_rel", evaluation.get("ragas_answer_relevance")),
        unsafe_allow_html=True,
    )
    st.markdown(
        score_bar("context_rel", evaluation.get("ragas_context_relevance")),
        unsafe_allow_html=True,
    )

with right:
    st.markdown(
        '<div style="font-size:11px;color:#888780;margin-bottom:6px;'
        'text-transform:uppercase;letter-spacing:0.5px">ASR cells</div>',
        unsafe_allow_html=True,
    )
    cells = [
        asr_cell("ASR-r retrieval", bool(evaluation["asr_retrieval"])),
        asr_cell("ASR-a answer", bool(evaluation["asr_answer"])),
        asr_cell("ASR-t target", bool(evaluation["asr_target"])),
        asr_cell("rank Δ@k", evaluation["rank_shift_at_k"]),
    ]
    asr_deny = evaluation.get("asr_deny")
    if asr_deny is not None:
        cells.append(asr_cell("ASR-deny", bool(asr_deny)))
        cells.append(asr_cell("verdict", verdict))
    st.markdown(asr_grid(cells), unsafe_allow_html=True)

if evaluation.get("evaluator_notes"):
    st.caption(f"Evaluator note: {evaluation['evaluator_notes']}")


# ---------------------------------------------------------------------------
# Section 6 — Iteration history (collapsed)
# ---------------------------------------------------------------------------

history = evaluation.get("iteration_history") or []
if history:
    with st.expander(f"Iteration history ({len(history)} step{'s' if len(history) != 1 else ''})"):
        st.dataframe(pd.DataFrame(history), use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Section 7 — Action row
# ---------------------------------------------------------------------------

st.markdown(
    '<h3 style="font-size:13px;font-weight:500;margin:24px 0 6px">Actions</h3>',
    unsafe_allow_html=True,
)

a1, a2, a3 = st.columns(3)
with a1:
    bundle_bytes = Path(bundle_path).read_bytes()
    st.download_button(
        label="Download bundle JSON",
        data=bundle_bytes,
        file_name=Path(bundle_path).name,
        mime="application/json",
        use_container_width=True,
    )
with a2:
    if st.button("Copy reproducibility config", use_container_width=True):
        st.toast("Not implemented in Build A", icon="⏳")
with a3:
    if st.button("Re-run with different seed", use_container_width=True):
        st.toast("Not implemented in Build A", icon="⏳")

with st.expander("Raw bundle JSON"):
    st.json(bundle)
