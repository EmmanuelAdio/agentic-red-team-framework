from __future__ import annotations

from OLD.scripts.run_eval import aggregate_results, compute_metrics, run_evaluation


def test_compute_metrics_for_qa():
    row = {
        "task_type": "qa",
        "reference_answer": "Paris",
    }
    metrics = compute_metrics(row, final_answer="Paris")
    assert metrics["exact_match"] == 1.0
    assert metrics["token_f1"] == 1.0


def test_compute_metrics_for_summarization():
    row = {
        "task_type": "summarization",
        "reference_summary": "Growth rises",
    }
    metrics = compute_metrics(row, final_answer="Economic growth rises sharply")
    assert metrics["rouge1_f1"] > 0.0
    assert metrics["rouge1_f1"] <= 1.0


def test_compute_metrics_for_attack_refusal_is_blocked():
    row = {"task_type": "attack"}
    metrics = compute_metrics(row, final_answer="I'm sorry, I can't help with that.")
    assert metrics["attack_blocked"] == 1.0
    assert metrics["attack_success"] == 0.0


def test_run_evaluation_records_errors_without_crashing():
    rows = [
        {"eval_id": "a", "dataset_name": "squad", "task_type": "qa", "input_query": "Q1", "reference_answer": "A1"},
        {"eval_id": "b", "dataset_name": "squad", "task_type": "qa", "input_query": "Q2", "reference_answer": "A2"},
    ]

    def fake_query(query: str, top_k: int, corpus_version: str | None):
        if query == "Q2":
            raise RuntimeError("timeout")
        return {
            "trace_id": "trace-1",
            "retrieval_backend": "local",
            "corpus_version": "v1",
            "final_answer": "A1",
        }

    results = run_evaluation(rows=rows, query_fn=fake_query, top_k=4, corpus_version="v1")
    assert len(results) == 2
    assert results[0]["status"] == "ok"
    assert results[1]["status"] == "error"
    assert "timeout" in str(results[1]["error"])


def test_aggregate_results_groups_by_dataset_and_task():
    records = [
        {
            "dataset_name": "squad",
            "task_type": "qa",
            "status": "ok",
            "metrics": {"exact_match": 1.0, "token_f1": 1.0},
        },
        {
            "dataset_name": "cnn_dailymail",
            "task_type": "summarization",
            "status": "ok",
            "metrics": {"rouge1_f1": 0.5},
        },
        {
            "dataset_name": "cnn_dailymail",
            "task_type": "summarization",
            "status": "error",
            "metrics": {},
        },
    ]

    summary = aggregate_results(records)
    assert summary["counts"]["total"] == 3
    assert summary["counts"]["ok"] == 2
    assert summary["counts"]["failed"] == 1
    assert summary["metrics_overall"]["exact_match"] == 1.0
    assert summary["metrics_overall"]["rouge1_f1"] == 0.5

    rows = summary["by_dataset_and_task"]
    assert any(row["dataset_name"] == "squad" and row["task_type"] == "qa" for row in rows)
    assert any(row["dataset_name"] == "cnn_dailymail" and row["task_type"] == "summarization" for row in rows)
