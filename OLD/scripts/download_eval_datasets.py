from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any
from urllib import request

try:
    from OLD.scripts.common import write_jsonl
    from OLD.scripts.load_eval_datasets import load_dataset
except ModuleNotFoundError:  # pragma: no cover - support direct script execution
    from OLD.scripts.common import write_jsonl
    from OLD.scripts.load_eval_datasets import load_dataset

SQUAD_V1_DEV_URL = "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json"


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _download_bytes(url: str, timeout_seconds: float) -> bytes:
    with request.urlopen(url, timeout=timeout_seconds) as response:
        return response.read()


def _download_squad_raw(output_path: Path, timeout_seconds: float, force: bool) -> Path:
    if output_path.exists() and not force:
        return output_path

    _ensure_parent(output_path)
    payload = _download_bytes(SQUAD_V1_DEV_URL, timeout_seconds=timeout_seconds)
    output_path.write_bytes(payload)
    return output_path


def _load_hf_dataset_rows(dataset_name: str, split: str, limit: int | None) -> list[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "The 'datasets' package is required for Hugging Face downloads. "
            "Install with: pip install datasets"
        ) from exc

    if dataset_name == "cnn_dailymail":
        ds = load_dataset("cnn_dailymail", "3.0.0", split=split)
        rows: list[dict[str, Any]] = []
        for idx, row in enumerate(ds):
            if limit is not None and idx >= limit:
                break
            rows.append(
                {
                    "id": row.get("id", idx),
                    "article": row.get("article", ""),
                    "highlights": row.get("highlights", ""),
                }
            )
        return rows

    if dataset_name == "gigaword":
        ds = load_dataset("gigaword", split=split)
        rows = []
        for idx, row in enumerate(ds):
            if limit is not None and idx >= limit:
                break
            rows.append(
                {
                    "id": row.get("id", idx),
                    "document": row.get("document", ""),
                    "summary": row.get("summary", ""),
                }
            )
        return rows

    raise ValueError(f"Unsupported Hugging Face dataset: {dataset_name}")


def _write_raw_jsonl(output_path: Path, rows: list[dict[str, Any]], force: bool) -> Path:
    if output_path.exists() and not force:
        return output_path
    write_jsonl(output_path, rows)
    return output_path


def _normalize_dataset(dataset_name: str, raw_path: Path, normalized_path: Path) -> Path:
    rows = load_dataset(dataset_name=dataset_name, input_path=raw_path)
    write_jsonl(normalized_path, rows)
    return normalized_path


def download_squad(output_root: Path, timeout_seconds: float, force: bool) -> dict[str, str]:
    raw_path = output_root / "eval" / "squad.json"
    normalized_path = output_root / "eval" / "squad_eval.jsonl"
    _download_squad_raw(raw_path, timeout_seconds=timeout_seconds, force=force)
    _normalize_dataset("squad", raw_path, normalized_path)
    return {"dataset_name": "squad", "raw_path": str(raw_path), "normalized_path": str(normalized_path)}


def download_hf_eval_dataset(
    dataset_name: str,
    output_root: Path,
    split: str,
    limit: int | None,
    force: bool,
) -> dict[str, str]:
    if dataset_name == "cnn_dailymail":
        raw_path = output_root / "eval" / "cnn_dailymail.jsonl"
        normalized_path = output_root / "eval" / "cnn_eval.jsonl"
    elif dataset_name == "gigaword":
        raw_path = output_root / "eval" / "gigaword.jsonl"
        normalized_path = output_root / "eval" / "gigaword_eval.jsonl"
    else:
        raise ValueError(f"Unsupported dataset_name: {dataset_name}")

    if not raw_path.exists() or force:
        rows = _load_hf_dataset_rows(dataset_name=dataset_name, split=split, limit=limit)
        _write_raw_jsonl(raw_path, rows, force=True)

    _normalize_dataset(dataset_name, raw_path, normalized_path)
    return {"dataset_name": dataset_name, "raw_path": str(raw_path), "normalized_path": str(normalized_path)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and normalize evaluation datasets (SQuAD/CNN/Gigaword).")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["squad", "cnn_dailymail"],
        choices=["squad", "cnn_dailymail", "gigaword"],
        help="Datasets to download.",
    )
    parser.add_argument("--output-root", type=Path, default=Path("data"))
    parser.add_argument("--hf-split", default="test")
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--timeout-seconds", type=float, default=30.0)
    parser.add_argument("--force", action="store_true", help="Overwrite existing raw download files.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reports: list[dict[str, str]] = []

    for dataset_name in args.datasets:
        if dataset_name == "squad":
            reports.append(
                download_squad(
                    output_root=args.output_root,
                    timeout_seconds=args.timeout_seconds,
                    force=args.force,
                )
            )
        else:
            reports.append(
                download_hf_eval_dataset(
                    dataset_name=dataset_name,
                    output_root=args.output_root,
                    split=args.hf_split,
                    limit=args.limit,
                    force=args.force,
                )
            )

    print("Downloaded and normalized datasets:")
    for report in reports:
        print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
