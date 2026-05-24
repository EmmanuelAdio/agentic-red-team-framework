"""Throwaway helper: extract appendix data from real bundles + compute hashes.

Run from repo root with the project venv activated:
    python scripts/_appendix_helper.py

Prints:
  1. SHA-256 hashes for the five prompt templates (verbatim).
  2. The matching `prompt_template_hash` value in a real bundle.
  3. The query_text for test1195.
  4. For each of the 4 cells: the iteration-0 (template path) payload and
     the iteration>=1 (LLM path) payload — sources as run_id + file path.

NOT committed. Reads files only; no writes.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Make src importable without installing.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from redteam.target.generator import PROMPT_TEMPLATE, PROMPT_TEMPLATE_HASH
from redteam.attacks.prompt_injection import INSTRUCTION_OVERRIDE_TEMPLATE
from redteam.attacks.corpus_poisoning import ANSWER_REPLACEMENT_TEMPLATE, JAMMING_TEMPLATE
from redteam.agents.exploit_generator import (
    IPI_PROMPT_TEMPLATE,
    POISON_PROMPT_TEMPLATE,
    QUERY_INJECTION_PROMPT_TEMPLATE,
    IPI_PROMPT_TEMPLATE_HASH,
    POISON_PROMPT_TEMPLATE_HASH,
    QUERY_INJECTION_PROMPT_TEMPLATE_HASH,
    _hash,
)


def banner(s: str) -> None:
    print("\n" + "=" * 78)
    print(s)
    print("=" * 78)


banner("Hashes")
print(f"RAG generator prompt:                {PROMPT_TEMPLATE_HASH}")
print(f"IPI instruction-override (template): {_hash(INSTRUCTION_OVERRIDE_TEMPLATE)}")
print(f"Answer-replacement (template):       {_hash(ANSWER_REPLACEMENT_TEMPLATE)}")
print(f"Jamming (template):                  {_hash(JAMMING_TEMPLATE)}")
print(f"LLM IPI exploit prompt:              {IPI_PROMPT_TEMPLATE_HASH}")
print(f"LLM poison exploit prompt:           {POISON_PROMPT_TEMPLATE_HASH}")
print(f"LLM query-injection exploit prompt:  {QUERY_INJECTION_PROMPT_TEMPLATE_HASH}")

banner("Verify against real bundle")
sample = ROOT / "results/runs/batch_seed42_ipi_20260508T230413Z/run_test1195_seed42_ipi_20260508T230413Z_bundle.json"
with sample.open("r", encoding="utf-8") as f:
    b = json.load(f)
print(f"Bundle file: {sample.relative_to(ROOT)}")
print(f"bundle.target_system.prompt_template_hash = {b['target_system']['prompt_template_hash']}")
print(f"computed PROMPT_TEMPLATE_HASH             = {PROMPT_TEMPLATE_HASH}")
print(f"match: {b['target_system']['prompt_template_hash'] == PROMPT_TEMPLATE_HASH}")
print(f"\nQuery text: {b['attack'].get('query_text') or b.get('summary', {}).get('query_text')}")
# fall back to scanning the bundle for query_text
def find_query(obj):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in ("query_text", "query"):
                if isinstance(v, str):
                    return v
            r = find_query(v)
            if r:
                return r
    elif isinstance(obj, list):
        for v in obj:
            r = find_query(v)
            if r:
                return r
    return None

query_text = find_query(b)
print(f"detected query_text: {query_text!r}")

banner("Extract payloads for test1195 across 4 cells")

CELLS = [
    ("Cell 1 IPI", "ipi", "20260508T230413Z", "payload"),
    ("Cell 2 PoisonedRAG answer-replacement", "poiA", "20260508T230413Z", "payload"),
    ("Cell 3 PoisonedRAG jamming", "poiJ", "20260519T205329Z", "payload"),
    ("Cell 4 Query injection", "qInj", "20260508T230413Z", "modified_query"),
]

for label, code, ts, field in CELLS:
    bundle_path = ROOT / f"results/runs/batch_seed42_{code}_{ts}/run_test1195_seed42_{code}_{ts}_bundle.json"
    print(f"\n--- {label} ---")
    print(f"file: {bundle_path.relative_to(ROOT)}")
    if not bundle_path.exists():
        print("  MISSING")
        continue
    with bundle_path.open("r", encoding="utf-8") as f:
        bundle = json.load(f)
    attack = bundle.get("attack", {})
    print(f"  run_id:           {bundle.get('run_id')}")
    print(f"  family:           {attack.get('family')}")
    print(f"  strategy:         {attack.get('strategy')}")
    print(f"  iteration:        {attack.get('iteration')}")
    print(f"  payload_source:   {attack.get('payload_source')}")
    print(f"  attack_channel:   {attack.get('attack_channel')}")
    val = attack.get(field)
    print(f"  {field}:")
    if val:
        for line in str(val).splitlines():
            print(f"    | {line}")
    else:
        print("    (empty / not set)")
    # Also report execution.history if multiple iterations exist
    hist = bundle.get("execution", {}).get("history") or bundle.get("history") or []
    if hist:
        print(f"  history length:   {len(hist)}")
        for i, h in enumerate(hist):
            src = h.get("payload_source") or h.get("attack", {}).get("payload_source")
            it = h.get("iteration") if "iteration" in h else h.get("attack", {}).get("iteration")
            print(f"    iter {it}: payload_source={src}")
