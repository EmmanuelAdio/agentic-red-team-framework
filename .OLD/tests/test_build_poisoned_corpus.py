from OLD.scripts.build_poisoned_corpus import build_poisoned_corpus


def test_build_poisoned_corpus_contains_all_attack_types_and_is_deterministic():
    first = build_poisoned_corpus()
    second = build_poisoned_corpus()

    assert first == second
    attack_types = {row["attack_type"] for row in first}
    assert attack_types == {
        "prompt_injection",
        "instruction_override",
        "misleading_fact",
        "retrieval_manipulation",
        "data_exfiltration_attempt",
    }

    assert all(row["doc_type"] == "poisoned" for row in first)
    assert all(row["entity_type"] == "poisoned" for row in first)
