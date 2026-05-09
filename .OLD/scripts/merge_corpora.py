from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

try:
    from OLD.scripts.common import read_jsonl, write_jsonl
except ModuleNotFoundError:  # pragma: no cover - support direct script execution
    from OLD.scripts.common import read_jsonl, write_jsonl

REQUIRED_FIELDS = {
    "doc_id",
    "entity_id",
    "entity_type",
    "title",
    "source",
    "source_type",
    "content",
    "doc_type",
    "attack_type",
    "tags",
    "metadata",
}


def validate_retrieval_row(row: dict[str, Any]) -> None:
    missing = [field for field in sorted(REQUIRED_FIELDS) if field not in row]
    if missing:
        raise ValueError(f"Missing fields {missing} in row doc_id={row.get('doc_id')}")


def merge_corpora(paths: list[Path], dedupe: bool = False, allow_missing: bool = False) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    seen: set[str] = set()
    for path in paths:
        if not path.exists():
            if allow_missing:
                continue
            raise FileNotFoundError(
                f"Corpus input file not found: {path}. "
                "Build it first, or run with --allow-missing to skip missing optional sources."
            )
        for row in read_jsonl(path):
            validate_retrieval_row(row)
            doc_id = str(row.get("doc_id"))
            if doc_id in seen:
                if dedupe:
                    continue
                raise ValueError(f"Duplicate doc_id encountered: {doc_id}")
            seen.add(doc_id)
            merged.append(row)
    return merged


def summarize(rows: list[dict[str, Any]]) -> dict[str, Counter[str]]:
    return {
        "entity_type": Counter(str(row.get("entity_type")) for row in rows),
        "doc_type": Counter(str(row.get("doc_type")) for row in rows),
        "attack_type": Counter(str(row.get("attack_type")) for row in rows if row.get("attack_type") is not None),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge structured/wiki/poisoned corpora into one retrieval corpus JSONL.")
    parser.add_argument("--structured", type=Path, default=Path("data/corpus_structured.jsonl"))
    parser.add_argument("--wiki", type=Path, default=Path("data/corpus_wiki.jsonl"))
    parser.add_argument("--poisoned", type=Path, default=Path("data/corpus_poisoned.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data/corpus_retrieval.jsonl"))
    parser.add_argument("--dedupe", action="store_true", help="Skip duplicate doc_ids instead of failing.")
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Skip any missing input corpus files (useful if wiki corpus is intentionally omitted).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = merge_corpora(
        [args.structured, args.wiki, args.poisoned],
        dedupe=args.dedupe,
        allow_missing=args.allow_missing,
    )
    write_jsonl(args.output, rows)

    stats = summarize(rows)
    print(f"Wrote {len(rows)} merged retrieval documents to {args.output}")
    print(f"entity_type: {dict(stats['entity_type'])}")
    print(f"doc_type: {dict(stats['doc_type'])}")
    print(f"attack_type: {dict(stats['attack_type'])}")


if __name__ == "__main__":
    main()
