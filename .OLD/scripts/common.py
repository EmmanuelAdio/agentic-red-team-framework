from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


def normalize_text(text: str) -> str:
    """Normalize whitespace to improve deterministic corpus construction."""

    return re.sub(r"\s+", " ", str(text or "")).strip()


def make_slug(value: str) -> str:
    """Create a stable slug used in deterministic identifiers."""

    slug = re.sub(r"[^a-z0-9]+", "-", str(value or "").lower()).strip("-")
    return slug or "unknown"


def stable_id(prefix: str, *parts: str) -> str:
    """Create deterministic IDs from semantic parts."""

    raw = "||".join(str(part) for part in parts)
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]
    return f"{prefix}_{digest}"


def read_json(path: Path) -> Any:
    """Read a JSON file and return parsed payload."""

    return json.loads(path.read_text(encoding="utf-8"))


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write JSONL with deterministic formatting and newline termination."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read JSONL records from file."""

    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        records.append(json.loads(stripped))
    return records


@dataclass(slots=True)
class RetrievalDocument:
    """Canonical retrieval-document shape used by corpus build scripts."""

    doc_id: str
    entity_id: str
    entity_type: str
    title: str
    source: str
    source_type: str
    content: str
    doc_type: str
    attack_type: str | None
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["content"] = normalize_text(data["content"])
        data["tags"] = sorted(set(data.get("tags") or []))
        return data


@dataclass(slots=True)
class EvalSample:
    """Canonical evaluation/attack sample shape kept outside retrieval corpus."""

    eval_id: str
    dataset_name: str
    task_type: str
    input_query: str
    reference_answer: str | None = None
    reference_summary: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["input_query"] = normalize_text(data["input_query"])
        if data.get("reference_answer") is not None:
            data["reference_answer"] = normalize_text(data["reference_answer"])
        if data.get("reference_summary") is not None:
            data["reference_summary"] = normalize_text(data["reference_summary"])
        return data
