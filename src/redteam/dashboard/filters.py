"""Pure filter helpers for the Streamlit dashboard.

The Streamlit layer (``dashboard/Home.py`` and the Build-B
``pages/01_aggregate.py``) renders the widgets — pills if available,
``st.multiselect`` as a fallback. The functions in this module are
pure and Streamlit-free, so the smoke suite can exercise them without
spinning up a runtime.

A *selection* is a dict mapping column-name → list of allowed values.
Empty list / missing key = no filter on that column.
"""

from __future__ import annotations

from typing import Any, Mapping

import pandas as pd


# Columns the dashboard exposes as user filters. Defined here in one
# place so the sidebar, the apply helper, and the Aggregate page all
# stay in lock-step.
FILTER_COLUMNS: tuple[str, ...] = (
    "attack_family",
    "attack_channel",
    "payload_source",
    "verdict",
)


def available_options(df: pd.DataFrame) -> dict[str, list[Any]]:
    """Return sorted unique values for each filter column present in ``df``.

    Missing columns silently produce an empty list — defensive against
    older bundle batches that predate the Day-7.5 additive fields.
    """
    out: dict[str, list[Any]] = {}
    for col in FILTER_COLUMNS:
        if col not in df.columns:
            out[col] = []
            continue
        values = df[col].dropna().unique().tolist()
        try:
            values.sort()
        except TypeError:
            # Mixed types — keep insertion order rather than crash.
            pass
        out[col] = values
    return out


def apply_filters(
    df: pd.DataFrame,
    selection: Mapping[str, list[Any]] | None,
) -> pd.DataFrame:
    """Filter ``df`` to rows matching every non-empty selection entry.

    An empty list (or a missing key) means "keep every value" for that
    column. The conjunction across columns is AND. The return is a
    fresh DataFrame; the caller may freely sort/index it.
    """
    if df.empty or not selection:
        return df
    mask = pd.Series(True, index=df.index)
    for col, allowed in selection.items():
        if not allowed:
            continue
        if col not in df.columns:
            continue
        mask &= df[col].isin(allowed)
    return df.loc[mask].copy()


__all__ = [
    "FILTER_COLUMNS",
    "available_options",
    "apply_filters",
]
