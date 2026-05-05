"""Write data/queries.json — the 50-query test set used for evaluation.

Calls the same `select_test_queries` that the corpus loader uses, so the queries
written here are guaranteed to have at least one gold doc in the indexed slice.

Run from repo root:
    python scripts/04_build_query_set.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from redteam.config import DATA_DIR, load_env
from redteam.target.corpus import select_test_queries

QUERIES_PATH = DATA_DIR / "queries.json"


def main() -> None:
    load_env()

    selected = select_test_queries(n_queries=50, seed=42)

    payload = [
        {"query_id": qid, "query_text": text, "gold_doc_ids": gold_ids}
        for qid, text, gold_ids in selected
    ]

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    QUERIES_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"Wrote {len(payload)} queries -> {QUERIES_PATH}")
    if payload:
        print("First entry:")
        print(json.dumps(payload[0], indent=2))


if __name__ == "__main__":
    main()
