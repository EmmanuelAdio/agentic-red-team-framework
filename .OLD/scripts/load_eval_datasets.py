from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

try:
    from OLD.scripts.common import EvalSample, read_json, read_jsonl, stable_id, write_jsonl
except ModuleNotFoundError:  # pragma: no cover - support direct script execution
    from OLD.scripts.common import EvalSample, read_json, read_jsonl, stable_id, write_jsonl

# Retrieval corpus documents are grounded knowledge sources for RAG retrieval.
# Evaluation and adversarial datasets are kept separate so baseline metrics and
# red-team experiments stay controlled and reproducible.


def _load_squad(path: Path, dataset_name: str = "squad") -> list[EvalSample]:
    payload = read_json(path)
    articles = payload.get("data", []) if isinstance(payload, dict) else payload
    samples: list[EvalSample] = []
    for article in articles:
        title = str(article.get("title") or "untitled")
        for paragraph in article.get("paragraphs", []):
            context = str(paragraph.get("context") or "")
            for qa in paragraph.get("qas", []):
                question = str(qa.get("question") or "")
                answers = qa.get("answers") or []
                reference = ""
                if answers:
                    reference = str((answers[0] or {}).get("text") or "")
                sample_id = stable_id("eval", dataset_name, title, str(qa.get("id") or question))
                samples.append(
                    EvalSample(
                        eval_id=sample_id,
                        dataset_name=dataset_name,
                        task_type="qa",
                        input_query=question,
                        reference_answer=reference,
                        metadata={
                            "title": title,
                            "question_id": qa.get("id"),
                            "context": context,
                        },
                    )
                )
    return samples


def _load_cnn_dailymail(path: Path, dataset_name: str = "cnn_dailymail") -> list[EvalSample]:
    rows = read_jsonl(path)
    samples: list[EvalSample] = []
    for idx, row in enumerate(rows):
        article = str(row.get("article") or row.get("document") or "")
        summary = str(row.get("highlights") or row.get("summary") or "")
        if not article:
            continue
        samples.append(
            EvalSample(
                eval_id=stable_id("eval", dataset_name, str(row.get("id") or idx)),
                dataset_name=dataset_name,
                task_type="summarization",
                input_query=article,
                reference_summary=summary,
                metadata={"row_index": idx},
            )
        )
    return samples


def _load_gigaword(path: Path, dataset_name: str = "gigaword") -> list[EvalSample]:
    rows = read_jsonl(path)
    samples: list[EvalSample] = []
    for idx, row in enumerate(rows):
        document = str(row.get("document") or row.get("article") or "")
        summary = str(row.get("summary") or row.get("headline") or "")
        if not document:
            continue
        samples.append(
            EvalSample(
                eval_id=stable_id("eval", dataset_name, str(row.get("id") or idx)),
                dataset_name=dataset_name,
                task_type="summarization",
                input_query=document,
                reference_summary=summary,
                metadata={"row_index": idx},
            )
        )
    return samples


def _load_attack_prompts(path: Path, dataset_name: str) -> list[EvalSample]:
    """Load adversarial prompts for replay/sampling in the red-team loop.

    Future attack agents can sample these `task_type=attack` prompts and adapt
    them per target policy/model while preserving dataset provenance.
    """

    try:
        rows = read_jsonl(path)
    except Exception:
        payload = read_json(path)
        rows = payload if isinstance(payload, list) else payload.get("data", [])

    samples: list[EvalSample] = []
    for idx, row in enumerate(rows):
        prompt = str(
            row.get("prompt")
            or row.get("input")
            or row.get("query")
            or row.get("jailbreak_prompt")
            or ""
        )
        if not prompt:
            continue
        samples.append(
            EvalSample(
                eval_id=stable_id("eval", dataset_name, str(row.get("id") or idx), prompt[:80]),
                dataset_name=dataset_name,
                task_type="attack",
                input_query=prompt,
                reference_answer=str(row.get("target") or row.get("expected_behavior") or "") or None,
                metadata={
                    "row_index": idx,
                    "attack_category": row.get("attack_category") or row.get("category"),
                },
            )
        )
    return samples


def load_dataset(dataset_name: str, input_path: Path) -> list[dict[str, Any]]:
    key = dataset_name.lower()
    if key == "squad":
        rows = _load_squad(input_path, dataset_name="squad")
    elif key in {"cnn_dailymail", "cnn-dailymail", "cnndm"}:
        rows = _load_cnn_dailymail(input_path, dataset_name="cnn_dailymail")
    elif key == "gigaword":
        rows = _load_gigaword(input_path, dataset_name="gigaword")
    elif key in {"jailbreakbench", "promptbench", "anthropic_redteam", "anthropic-redteam"}:
        canonical = {
            "jailbreakbench": "jailbreakbench",
            "promptbench": "promptbench",
            "anthropic_redteam": "anthropic_redteam",
            "anthropic-redteam": "anthropic_redteam",
        }[key]
        rows = _load_attack_prompts(input_path, dataset_name=canonical)
    else:
        raise ValueError(f"Unsupported dataset_name: {dataset_name}")

    return [row.to_dict() for row in rows]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load eval/attack datasets into a normalized JSONL format.")
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--input-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_dataset(args.dataset_name, args.input_path)
    if args.limit is not None:
        rows = rows[: args.limit]

    write_jsonl(args.output_path, rows)
    print(
        f"Wrote {len(rows)} {args.dataset_name} records to {args.output_path}. "
        "These records are evaluation/attack inputs and are intentionally separate from retrieval corpus files."
    )


if __name__ == "__main__":
    main()
