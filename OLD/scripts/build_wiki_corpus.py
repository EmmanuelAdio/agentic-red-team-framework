from __future__ import annotations

import argparse
import json
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

try:
    from OLD.scripts.common import RetrievalDocument, normalize_text, stable_id, write_jsonl
except ModuleNotFoundError:  # pragma: no cover - support direct script execution
    from OLD.scripts.common import RetrievalDocument, normalize_text, stable_id, write_jsonl

DEFAULT_WIKI_TOPICS = [
    "University",
    "Higher education",
    "Loughborough University",
    "Computer science",
    "Artificial intelligence",
    "Information retrieval",
    "Cybersecurity",
    "Computer security",
]


def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    cleaned = normalize_text(text)
    if not cleaned:
        return []
    if chunk_size <= 0:
        return [cleaned]

    words = cleaned.split(" ")
    chunks: list[str] = []
    step = max(chunk_size - chunk_overlap, 1)
    for start in range(0, len(words), step):
        piece = " ".join(words[start : start + chunk_size]).strip()
        if piece:
            chunks.append(piece)
        if start + chunk_size >= len(words):
            break
    return chunks


def fetch_wikipedia_extract(title: str, timeout_seconds: float = 15.0) -> str:
    """Fetch plain-text extract via Wikipedia API.

    Optional online fetch for enrichment; reproducibility is preserved by fixed topics,
    deterministic chunking, and persisted JSONL outputs.
    """

    params = urllib.parse.urlencode(
        {
            "action": "query",
            "format": "json",
            "prop": "extracts",
            "explaintext": 1,
            "redirects": 1,
            "titles": title,
        }
    )
    url = f"https://en.wikipedia.org/w/api.php?{params}"
    with urllib.request.urlopen(url, timeout=timeout_seconds) as response:  # noqa: S310
        payload = json.loads(response.read().decode("utf-8"))

    pages = payload.get("query", {}).get("pages", {})
    for _, page in pages.items():
        return normalize_text(page.get("extract") or "")
    return ""


def build_wiki_corpus(
    topics: list[str],
    chunk_size: int = 160,
    chunk_overlap: int = 30,
    source_name: str = "wikipedia",
) -> list[dict[str, Any]]:
    rows: list[RetrievalDocument] = []
    for topic in sorted(set(topics)):
        extracted = fetch_wikipedia_extract(topic)
        if not extracted:
            continue
        for idx, chunk in enumerate(chunk_text(extracted, chunk_size, chunk_overlap)):
            rows.append(
                RetrievalDocument(
                    doc_id=stable_id("doc", "wiki", topic, str(idx)),
                    entity_id=stable_id("wiki", topic),
                    entity_type="wiki",
                    title=f"{topic} (wiki chunk {idx + 1})",
                    source=topic,
                    source_type=source_name,
                    content=chunk,
                    doc_type="benign",
                    attack_type=None,
                    tags=["wiki", "reference"],
                    metadata={"topic": topic, "chunk_index": idx},
                )
            )
    ordered = sorted(rows, key=lambda row: (row.metadata.get("topic", ""), row.metadata.get("chunk_index", 0), row.doc_id))
    return [row.to_dict() for row in ordered]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build deterministic wiki-derived corpus JSONL.")
    parser.add_argument("--output", type=Path, default=Path("data/corpus_wiki.jsonl"))
    parser.add_argument("--topics", nargs="*", default=DEFAULT_WIKI_TOPICS)
    parser.add_argument("--chunk-size", type=int, default=160)
    parser.add_argument("--chunk-overlap", type=int, default=30)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = build_wiki_corpus(
        topics=list(args.topics),
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )
    write_jsonl(args.output, rows)
    print(f"Wrote {len(rows)} wiki retrieval documents to {args.output}")


if __name__ == "__main__":
    main()
