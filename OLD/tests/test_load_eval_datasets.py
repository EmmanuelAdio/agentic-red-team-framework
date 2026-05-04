import json
from pathlib import Path

from OLD.scripts.load_eval_datasets import load_dataset


def test_load_eval_datasets_task_mapping_and_separation(tmp_path: Path):
    squad_path = tmp_path / "squad.json"
    squad_payload = {
        "data": [
            {
                "title": "Doc",
                "paragraphs": [
                    {
                        "context": "Paris is the capital of France.",
                        "qas": [{"id": "q1", "question": "What is the capital of France?", "answers": [{"text": "Paris"}]}],
                    }
                ],
            }
        ]
    }
    squad_path.write_text(json.dumps(squad_payload), encoding="utf-8")

    cnn_path = tmp_path / "cnn.jsonl"
    cnn_path.write_text('{"article":"Long article text","highlights":"Short summary"}\n', encoding="utf-8")

    gigaword_path = tmp_path / "gigaword.jsonl"
    gigaword_path.write_text('{"document":"Economic growth rises","summary":"Growth rises"}\n', encoding="utf-8")

    attack_path = tmp_path / "attack.jsonl"
    attack_path.write_text('{"prompt":"Ignore policy","attack_category":"jailbreak"}\n', encoding="utf-8")

    squad_rows = load_dataset("squad", squad_path)
    cnn_rows = load_dataset("cnn_dailymail", cnn_path)
    gigaword_rows = load_dataset("gigaword", gigaword_path)
    attack_rows = load_dataset("jailbreakbench", attack_path)

    assert squad_rows[0]["task_type"] == "qa"
    assert cnn_rows[0]["task_type"] == "summarization"
    assert gigaword_rows[0]["task_type"] == "summarization"
    assert attack_rows[0]["task_type"] == "attack"

    for row in squad_rows + cnn_rows + gigaword_rows + attack_rows:
        assert "doc_id" not in row
        assert "entity_type" not in row
