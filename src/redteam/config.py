from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

DATA_DIR: Path = PROJECT_ROOT / "data"
CORPUS_DIR: Path = DATA_DIR / "corpus"
RUNS_DIR: Path = DATA_DIR / "runs"
RESULTS_DIR: Path = PROJECT_ROOT / "results"
CHROMA_DIR: Path = PROJECT_ROOT / ".chroma"
CACHE_DB: Path = PROJECT_ROOT / ".cache.sqlite"

EMBEDDING_MODEL: str = "BAAI/bge-small-en-v1.5"
LLM_MODEL: str = "gpt-4o-mini-2024-07-18"
LLM_TEMPERATURE: float = 0.0
RETRIEVER_TOP_K: int = 5


def load_env() -> None:
    load_dotenv(PROJECT_ROOT / ".env")
    # Silence Chroma's PostHog telemetry pings (harmless but noisy).
    os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Copy .env.example to .env and add your key."
        )
