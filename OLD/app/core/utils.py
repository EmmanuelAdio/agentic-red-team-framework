from __future__ import annotations

import re
import uuid
from datetime import UTC, datetime


def utc_now() -> datetime:
    """Return timezone-aware UTC timestamp."""

    return datetime.now(UTC)


def make_corpus_version() -> str:
    """Generate immutable corpus version identifier."""

    timestamp = utc_now().strftime("%Y%m%d%H%M%S")
    return f"cv_{timestamp}_{uuid.uuid4().hex[:8]}"


def make_trace_id() -> str:
    """Generate trace identifier for query execution."""

    return f"trace_{uuid.uuid4().hex}"


def normalize_text(text: str) -> str:
    """Normalize whitespace while preserving content semantics."""

    compact = re.sub(r"\s+", " ", text).strip()
    return compact
