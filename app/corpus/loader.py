from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.core.utils import normalize_text
from app.corpus.schemas import AttackLabel, SourceType


@dataclass
class LoadedDocument:
    """Normalized in-memory document before chunking/storage."""

    doc_id: str
    title: str
    source_type: SourceType
    attack_label: AttackLabel
    raw_text: str
    metadata: dict[str, Any]


class CorpusLoader:
    """Loads local .txt and .json corpus files into normalized records."""

    def load_documents(self, source_dir: Path) -> list[LoadedDocument]:
        documents: list[LoadedDocument] = []
        for path in sorted(source_dir.rglob("*")):
            if path.is_dir():
                continue
            if path.suffix.lower() == ".txt":
                documents.append(self._load_txt(path))
            elif path.suffix.lower() == ".json":
                documents.extend(self._load_json(path))
        return documents

    def _load_txt(self, path: Path) -> LoadedDocument:
        text = normalize_text(path.read_text(encoding="utf-8"))
        stem = path.stem
        title = stem.replace("_", " ").title()

        inferred_label = AttackLabel.benign
        if re.search(r"poison", stem, flags=re.IGNORECASE):
            inferred_label = AttackLabel.poisoned
        elif re.search(r"mislead|false", stem, flags=re.IGNORECASE):
            inferred_label = AttackLabel.misleading

        return LoadedDocument(
            doc_id=stem,
            title=title,
            source_type=SourceType.txt,
            attack_label=inferred_label,
            raw_text=text,
            metadata={"path": str(path)},
        )

    def _load_json(self, path: Path) -> list[LoadedDocument]:
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = payload if isinstance(payload, list) else [payload]

        documents: list[LoadedDocument] = []
        for index, row in enumerate(rows):
            doc_id = str(row.get("doc_id") or f"{path.stem}_{index}")
            title = str(row.get("title") or doc_id)
            raw_text = normalize_text(str(row.get("raw_text") or ""))
            source_type = SourceType(str(row.get("source_type") or "json"))
            attack_label = AttackLabel(str(row.get("attack_label") or "benign"))
            metadata = dict(row.get("metadata") or {})
            metadata["path"] = str(path)

            documents.append(
                LoadedDocument(
                    doc_id=doc_id,
                    title=title,
                    source_type=source_type,
                    attack_label=attack_label,
                    raw_text=raw_text,
                    metadata=metadata,
                )
            )

        return documents
