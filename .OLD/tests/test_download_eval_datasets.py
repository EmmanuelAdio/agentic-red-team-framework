from __future__ import annotations

import json
from pathlib import Path

from OLD.scripts.download_eval_datasets import (
    _normalize_dataset,
    _write_raw_jsonl,
    download_hf_eval_dataset,
    download_squad,
)


def test_download_squad_writes_raw_and_normalized(tmp_path: Path, monkeypatch):
    squad_payload = {
        "data": [
            {
                "title": "Doc",
                "paragraphs": [
                    {
                        "context": "Paris is the capital of France.",
                        "qas": [
                            {
                                "id": "q1",
                                "question": "What is the capital of France?",
                                "answers": [{"text": "Paris"}],
                            }
                        ],
                    }
                ],
            }
        ]
    }

    def fake_download(raw_path: Path, timeout_seconds: float, force: bool) -> Path:
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_path.write_text(json.dumps(squad_payload), encoding="utf-8")
        return raw_path

    monkeypatch.setattr("scripts.download_eval_datasets._download_squad_raw", fake_download)

    report = download_squad(output_root=tmp_path, timeout_seconds=1.0, force=True)
    assert report["dataset_name"] == "squad"
    normalized_path = Path(report["normalized_path"])
    assert normalized_path.exists()
    first_line = normalized_path.read_text(encoding="utf-8").splitlines()[0]
    row = json.loads(first_line)
    assert row["task_type"] == "qa"


def test_download_hf_eval_dataset_writes_and_normalizes(tmp_path: Path, monkeypatch):
    def fake_hf_rows(dataset_name: str, split: str, limit: int | None):
        assert dataset_name == "cnn_dailymail"
        return [{"id": "r1", "article": "Long article", "highlights": "Short summary"}]

    monkeypatch.setattr("scripts.download_eval_datasets._load_hf_dataset_rows", fake_hf_rows)

    report = download_hf_eval_dataset(
        dataset_name="cnn_dailymail",
        output_root=tmp_path,
        split="test",
        limit=10,
        force=True,
    )
    normalized_path = Path(report["normalized_path"])
    assert normalized_path.exists()
    row = json.loads(normalized_path.read_text(encoding="utf-8").splitlines()[0])
    assert row["task_type"] == "summarization"


def test_write_raw_jsonl_respects_force(tmp_path: Path):
    path = tmp_path / "eval" / "x.jsonl"
    _write_raw_jsonl(path, [{"a": 1}], force=True)
    _write_raw_jsonl(path, [{"a": 2}], force=False)
    row = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    assert row["a"] == 1


def test_normalize_dataset_uses_existing_loader(tmp_path: Path):
    raw_path = tmp_path / "eval" / "cnn.jsonl"
    normalized_path = tmp_path / "eval" / "cnn_eval.jsonl"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_text('{"article":"A","highlights":"B"}\n', encoding="utf-8")

    _normalize_dataset("cnn_dailymail", raw_path, normalized_path)
    row = json.loads(normalized_path.read_text(encoding="utf-8").splitlines()[0])
    assert row["task_type"] == "summarization"
