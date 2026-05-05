"""Pytest config: put src/ on sys.path so the `redteam` package resolves
without needing `pip install -e .`."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
