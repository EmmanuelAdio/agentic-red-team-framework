from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

DATA_DIR: Path = PROJECT_ROOT / "data"
CORPUS_DIR: Path = DATA_DIR / "corpus"
RUNS_DIR: Path = DATA_DIR / "runs"
RESULTS_DIR: Path = PROJECT_ROOT / "results"
# Day 9: the experiment driver writes its bundles under
# ``results/runs/`` (per spec section 13 def-of-done line 421),
# separate from ``data/runs/`` (where the Day-8 dry run lives). Keeping
# the two roots distinct prevents Day-8 dry-run bundles from being
# accidentally aggregated into the Day-9 statistical analysis.
EXPERIMENT_RUNS_DIR: Path = RESULTS_DIR / "runs"
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
