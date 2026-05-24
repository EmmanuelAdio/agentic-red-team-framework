"""Extract one template-path + one llm-path bundle per cell, preferring test1195."""
from __future__ import annotations
import json, glob, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def find_bundle(cell_family: str, strategy_filter: str | None, source: str, prefer_qid: str = "test1195"):
    """Find first bundle whose attack matches the filters."""
    fallback = None
    for f in sorted(glob.glob(str(ROOT / "results/runs/batch_seed*_*/run_test*_bundle.json"))):
        with open(f, "r", encoding="utf-8") as fh:
            b = json.load(fh)
        a = b["attack"]
        if a["family"] != cell_family:
            continue
        if strategy_filter and a.get("strategy") != strategy_filter:
            continue
        if a["payload_source"] != source:
            continue
        qid = b["run_id"].rsplit("_", 1)[1]
        if qid == prefer_qid:
            return f, b
        if fallback is None:
            fallback = (f, b)
    return fallback

CELLS = [
    ("Cell 1 IPI",               "prompt_injection",  None,                  "payload"),
    ("Cell 2 PoisonedRAG-Answer","corpus_poisoning",  "answer_replacement",  "payload"),
    ("Cell 3 PoisonedRAG-Jam",   "corpus_poisoning",  "jamming",             "payload"),
    ("Cell 4 Query-Injection",   "query_injection",   None,                  "modified_query"),
]

def find_query_text(obj):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in ("query_text", "query") and isinstance(v, str):
                return v
            r = find_query_text(v)
            if r:
                return r
    elif isinstance(obj, list):
        for v in obj:
            r = find_query_text(v)
            if r:
                return r
    return None

for label, family, strat, field in CELLS:
    for source in ("template", "llm"):
        hit = find_bundle(family, strat, source)
        print(f"\n=== {label} | source={source} ===")
        if not hit:
            print("  (no bundle found)")
            continue
        f, b = hit
        qid = b["run_id"].rsplit("_", 1)[1]
        print(f"  file:        {Path(f).relative_to(ROOT)}")
        print(f"  run_id:      {b['run_id']}")
        print(f"  query_id:    {qid}")
        print(f"  query_text:  {find_query_text(b)!r}")
        print(f"  iteration:   {b['attack'].get('iteration')}")
        print(f"  strategy:    {b['attack'].get('strategy')}")
        if source == "llm":
            print(f"  exploit_prompt_template_hash: {b['attack'].get('exploit_prompt_template_hash')}")
        val = b["attack"].get(field)
        print(f"  --- {field} ---")
        if val:
            for line in str(val).splitlines():
                print(f"  | {line}")
