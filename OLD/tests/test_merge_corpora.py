import pytest

from OLD.scripts.merge_corpora import merge_corpora, summarize
from OLD.scripts.common import write_jsonl


def _row(doc_id: str, entity_type: str, doc_type: str, attack_type: str | None = None):
    return {
        "doc_id": doc_id,
        "entity_id": f"{entity_type}::{doc_id}",
        "entity_type": entity_type,
        "title": doc_id,
        "source": "test",
        "source_type": "test",
        "content": "sample",
        "doc_type": doc_type,
        "attack_type": attack_type,
        "tags": [],
        "metadata": {},
    }


def test_merge_corpora_respects_order_and_schema(tmp_path):
    p1 = tmp_path / "structured.jsonl"
    p2 = tmp_path / "wiki.jsonl"
    p3 = tmp_path / "poisoned.jsonl"
    write_jsonl(p1, [_row("a", "hall", "benign")])
    write_jsonl(p2, [_row("b", "wiki", "benign")])
    write_jsonl(p3, [_row("c", "poisoned", "poisoned", "prompt_injection")])

    merged = merge_corpora([p1, p2, p3])
    assert [row["doc_id"] for row in merged] == ["a", "b", "c"]

    stats = summarize(merged)
    assert stats["doc_type"]["benign"] == 2
    assert stats["doc_type"]["poisoned"] == 1


def test_merge_corpora_duplicate_doc_id_behavior(tmp_path):
    p1 = tmp_path / "p1.jsonl"
    p2 = tmp_path / "p2.jsonl"
    write_jsonl(p1, [_row("dup", "hall", "benign")])
    write_jsonl(p2, [_row("dup", "wiki", "benign")])

    with pytest.raises(ValueError):
        merge_corpora([p1, p2], dedupe=False)

    merged = merge_corpora([p1, p2], dedupe=True)
    assert len(merged) == 1
    assert merged[0]["doc_id"] == "dup"


def test_merge_corpora_allow_missing_input(tmp_path):
    p1 = tmp_path / "structured.jsonl"
    missing = tmp_path / "wiki_missing.jsonl"
    p3 = tmp_path / "poisoned.jsonl"
    write_jsonl(p1, [_row("a", "hall", "benign")])
    write_jsonl(p3, [_row("c", "poisoned", "poisoned", "prompt_injection")])

    merged = merge_corpora([p1, missing, p3], allow_missing=True)
    assert [row["doc_id"] for row in merged] == ["a", "c"]
