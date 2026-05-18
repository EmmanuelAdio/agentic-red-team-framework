"""DuckDB façade over the exploit-bundle JSON tree (Tier 3.3, opt-in).

The default ``data.load_bundles()`` path uses Python's recursive glob
plus per-file ``json.load``. This module gives the dashboard an opt-in
columnar alternative: every bundle file becomes one row in a DuckDB
``VIEW bundles`` whose schema mirrors ``data._project`` so downstream
code is shape-identical.

Enabled by setting ``REDTEAM_DASHBOARD_DUCKDB=1`` in the environment
before launching Streamlit. The DuckDB import lives inside this module
so a fresh ``git clone`` that has not installed ``duckdb`` still runs
on the default glob path.

Why an explicit ``columns=`` schema, not ``read_json_auto``
-----------------------------------------------------------
At 600 bundles with nested ``execution.retrieved_docs[]`` arrays of
mixed shape, ``read_json_auto`` hits an OOM during schema inference.
Declaring the projection explicitly bypasses inference entirely and
loads the full tree in well under a second.

Why a SQL ``VIEW`` and not a materialised table
------------------------------------------------
Re-running experiments and refreshing the dashboard should not require
rebuilding a DuckDB table. The view re-reads the JSON tree lazily on
every query; Streamlit's ``@st.cache_data(ttl=300)`` on
``load_bundles_via_duck`` is the only freshness gate.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

try:  # local import — duck.py is only loaded when the env-var is set
    import duckdb  # type: ignore
    _HAS_DUCKDB = True
except ImportError:  # pragma: no cover
    duckdb = None  # type: ignore
    _HAS_DUCKDB = False

try:
    from redteam.config import EXPERIMENT_RUNS_DIR, RUNS_DIR
except ImportError:  # pragma: no cover
    EXPERIMENT_RUNS_DIR = Path("results/runs")
    RUNS_DIR = Path("data/runs")


# Explicit JSON-to-SQL schema. Names and types match what
# ``data._project`` produces so a row from the view drops straight into
# the same downstream code. Nested fields are declared as STRUCTs and
# unpacked in the flattening SELECT below.
_JSON_COLUMNS: dict[str, str] = {
    "run_id":          "VARCHAR",
    "timestamp_utc":   "VARCHAR",
    "seed":            "INTEGER",
    "framework_version": "VARCHAR",
    "attack": (
        "STRUCT("
        "family VARCHAR, strategy VARCHAR, attack_channel VARCHAR, "
        "payload_source VARCHAR, iteration INTEGER"
        ")"
    ),
    "execution": (
        "STRUCT("
        "query VARCHAR, query_id VARCHAR, generator_latency_ms DOUBLE"
        ")"
    ),
    "evaluation": (
        "STRUCT("
        "asr_retrieval BOOLEAN, asr_answer BOOLEAN, asr_target BOOLEAN, "
        "asr_deny BOOLEAN, "
        "ragas_faithfulness DOUBLE, ragas_answer_relevance DOUBLE, "
        "ragas_context_relevance DOUBLE, "
        "rank_shift_at_k INTEGER, verdict VARCHAR"
        ")"
    ),
    "target_system": (
        "STRUCT("
        "embedding_model VARCHAR, llm_model VARCHAR"
        ")"
    ),
}

_MAX_OBJ_SIZE = 33_554_432  # 32 MiB; comfortably larger than any bundle.


# ---------------------------------------------------------------------------
# Connection lifecycle
# ---------------------------------------------------------------------------

_CON = None


def _connection():
    """Lazy module-level DuckDB connection. Raises if duckdb is missing."""
    global _CON
    if not _HAS_DUCKDB:
        raise RuntimeError(
            "duckdb is not installed. Pin `duckdb>=1.0` in requirements.in "
            "and `pip install duckdb`, or unset REDTEAM_DASHBOARD_DUCKDB."
        )
    if _CON is None:
        _CON = duckdb.connect(":memory:")
        _CON.execute("SET memory_limit='4GB';")
    return _CON


# ---------------------------------------------------------------------------
# View registration + projection
# ---------------------------------------------------------------------------


def _glob_for(root: Path) -> str:
    """DuckDB-style glob string under one bundle root."""
    return str((root / "**" / "run_*_bundle.json").as_posix())


def _columns_sql() -> str:
    """Render the ``columns={...}`` clause for ``read_json``."""
    items = ", ".join(f"'{k}': '{v}'" for k, v in _JSON_COLUMNS.items())
    return "{" + items + "}"


def _read_json_select(glob: str) -> str:
    """SELECT clause flattening the nested STRUCTs to the dashboard's flat row.

    Column names match ``data._project`` so a DuckDB row drops straight
    into the same downstream code. The ``filename`` pseudo-column
    becomes ``_path``; ``batch_id`` is derived from the parent folder
    name.
    """
    return f"""
        SELECT
            run_id,
            timestamp_utc                          AS timestamp,
            seed,
            regexp_replace(
                regexp_extract(filename, 'batch_[^/\\\\]+', 0),
                '^batch_', ''
            )                                      AS batch_id,
            execution.query                        AS query,
            execution.query_id                     AS query_id,
            attack.family                          AS attack_family,
            attack.strategy                        AS attack_strategy,
            COALESCE(attack.attack_channel, 'corpus')   AS attack_channel,
            COALESCE(attack.payload_source, 'template') AS payload_source,
            COALESCE(attack.iteration, 0)          AS iteration,
            target_system.embedding_model          AS embedding_model,
            target_system.llm_model                AS llm_model,
            evaluation.asr_retrieval               AS asr_r,
            evaluation.asr_answer                  AS asr_a,
            evaluation.asr_target                  AS asr_t,
            evaluation.asr_deny                    AS asr_deny,
            evaluation.ragas_faithfulness          AS faithfulness,
            evaluation.ragas_answer_relevance      AS answer_rel,
            evaluation.ragas_context_relevance     AS context_rel,
            evaluation.rank_shift_at_k             AS rank_shift,
            evaluation.verdict                     AS verdict,
            execution.generator_latency_ms         AS latency_ms,
            filename                               AS _path
        FROM read_json(
            '{glob}',
            columns={_columns_sql()},
            filename=true,
            union_by_name=true,
            maximum_object_size={_MAX_OBJ_SIZE}
        )
    """


def register_view(con, roots: Iterable[Path] | None = None) -> None:
    """Register ``VIEW bundles`` over the union of every bundle root.

    A second view ``VIEW bundles_count(n)`` is also registered for
    cheap row-count probes (used by the smoke test).
    """
    roots_t = list(roots) if roots is not None else [EXPERIMENT_RUNS_DIR, RUNS_DIR]
    existing = [r for r in roots_t if Path(r).exists()]
    if not existing:
        # No data on disk yet — register an empty view so SELECTs do not crash.
        con.execute(
            "CREATE OR REPLACE VIEW bundles AS "
            "SELECT NULL::VARCHAR AS run_id WHERE FALSE"
        )
        return

    parts = [_read_json_select(_glob_for(Path(r))) for r in existing]
    union_sql = "\nUNION ALL BY NAME\n".join(parts)
    con.execute(f"CREATE OR REPLACE VIEW bundles AS\n{union_sql}")


def _ensure_view() -> None:
    """Idempotent: create the view on the module connection if absent."""
    con = _connection()
    # Cheap probe — if `bundles` view does not exist this raises.
    try:
        con.execute("SELECT 1 FROM bundles LIMIT 0")
    except Exception:
        register_view(con)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def query_bundles(sql: str, params: Any | None = None) -> pd.DataFrame:
    """Run an arbitrary SQL query and return a pandas DataFrame.

    Examples
    --------
    >>> query_bundles("SELECT 42 AS answer")               # doctest: +SKIP
    >>> query_bundles("SELECT attack_family, AVG(asr_t::INTEGER) "
    ...               "FROM bundles GROUP BY 1")           # doctest: +SKIP
    """
    con = _connection()
    if "bundles" in sql.lower():
        _ensure_view()
    if params is None:
        return con.execute(sql).df()
    return con.execute(sql, params).df()


def load_bundles_via_duck() -> pd.DataFrame:
    """Drop-in replacement for ``data.load_bundles`` backed by DuckDB.

    Returns a DataFrame whose columns are the same set as the pandas
    glob path (locked by ``test_duckdb_columns_match_glob``), sorted by
    ``(timestamp DESC, run_id ASC)`` for parity with the pandas path's
    deterministic ordering.
    """
    _ensure_view()
    df = query_bundles(
        "SELECT * FROM bundles ORDER BY timestamp DESC, run_id ASC"
    )
    if not df.empty:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    return df


def is_enabled() -> bool:
    """True if ``REDTEAM_DASHBOARD_DUCKDB`` is set and duckdb is importable."""
    if not _HAS_DUCKDB:
        return False
    return os.environ.get("REDTEAM_DASHBOARD_DUCKDB", "").strip().lower() in {
        "1", "true", "yes", "on",
    }


__all__ = [
    "query_bundles",
    "load_bundles_via_duck",
    "register_view",
    "is_enabled",
]
