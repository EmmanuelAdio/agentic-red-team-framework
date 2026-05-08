"""Filesystem store for exploit bundles — batch-folder layout.

Layout
------

::

    data/runs/
      batch_<batch_id>/
        run_<query_id>_<batch_id>_bundle.json   # one per run
        run_<query_id>_<batch_id>_bundle.json
        ...
        batch_<batch_id>_summary.json           # rollup for this batch
      batch_<batch_id>/
        ...

Each invocation of the experiment driver (or the dry-run script) creates
**one batch folder** and writes every bundle from that invocation into
it, plus a single batch-level summary alongside. This matches the unit
of work users actually reason about — "the batch I ran on Tuesday with
seed 42" — rather than fragmenting one batch across many top-level
folders.

The store is a thin filesystem wrapper rather than a database:

* Bundles are *append-only* — the framework never updates an existing
  run, only writes new ones. A directory of JSON files is the simplest
  shape for that, and is trivially diff-able, grep-able, and
  Zenodo-friendly per spec §13's Definition of Done.
* The Day-9 experiment matrix produces a few hundred bundles. JSON-on-disk
  fits; a SQL store would be premature. A future migration to a cloud
  blob store / S3 is logged in `FUTURE_WORKS.md` §5; this module is the
  seam where that swap would land.

Atomicity
---------

Every write goes via a sibling ``*.tmp`` file plus ``os.replace``, which
is atomic on both POSIX and Windows when source and destination share a
filesystem. Mid-write crashes therefore leave either the previous file
or the new one — never a half-written JSON.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Iterator

from redteam.bundles.schema import ExploitBundle

# Reject identifiers that could escape the store directory. Bundle run_ids
# and batch_ids are constructed from a UTC timestamp + alphanumerics, so
# this is defence-in-depth against an adversarially-malformed state
# attempting path traversal.
_SAFE_ID = re.compile(r"^[A-Za-z0-9_\-:.]+$")


def _validate_id(value: str, kind: str) -> None:
    """Raise ``ValueError`` if `value` contains path-traversal characters."""
    if not _SAFE_ID.match(value):
        raise ValueError(
            f"{kind} {value!r} contains characters outside [A-Za-z0-9_\\-:.]; "
            "refusing to use as a path component."
        )


def _atomic_write_text(path: Path, payload: str) -> None:
    """Write `payload` to `path` atomically via a sibling ``*.tmp``."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(payload, encoding="utf-8")
    os.replace(tmp, path)


def bundle_filename(query_id: str, batch_id: str) -> str:
    """Filename for one bundle inside a batch folder. Single source of truth."""
    return f"run_{query_id}_{batch_id}_bundle.json"


def summary_filename(batch_id: str) -> str:
    """Filename for the batch-level summary inside a batch folder."""
    return f"batch_{batch_id}_summary.json"


class BundleStore:
    """Bundle store scoped to one batch folder.

    Parameters
    ----------
    root_dir:
        Top-level runs directory (typically ``data/runs/``). The batch
        folder ``batch_<batch_id>/`` is created beneath this on first
        write.
    batch_id:
        Identifier for this batch — usually a UTC timestamp string
        (e.g. ``20260508T144045Z``). All bundles written through this
        store land in the same batch folder.
    """

    def __init__(self, root_dir: Path, batch_id: str) -> None:
        _validate_id(batch_id, "batch_id")
        self.root_dir = Path(root_dir)
        self.batch_id = batch_id
        self.batch_dir = self.root_dir / f"batch_{batch_id}"
        self.batch_dir.mkdir(parents=True, exist_ok=True)

    # ---- paths -------------------------------------------------------------

    def path_for(self, query_id: str) -> Path:
        """Return the on-disk path for a given query's bundle (no I/O)."""
        _validate_id(query_id, "query_id")
        return self.batch_dir / bundle_filename(query_id, self.batch_id)

    @property
    def summary_path(self) -> Path:
        """Path to this batch's summary JSON (no I/O)."""
        return self.batch_dir / summary_filename(self.batch_id)

    # ---- write -------------------------------------------------------------

    def write(self, bundle: ExploitBundle) -> Path:
        """Atomically write `bundle` into this batch's folder.

        The on-disk filename is ``run_<query_id>_<batch_id>_bundle.json``
        — derived from the bundle's own ``execution.query_id`` so the
        store keeps its own filename convention rather than relying on
        the more verbose internal ``run_id`` field.
        """
        target = self.path_for(bundle.execution.query_id)
        _atomic_write_text(target, bundle.to_json(indent=2))
        return target

    def write_batch_summary(self, summary: dict) -> Path:
        """Atomically write the batch-level summary JSON into this batch's folder."""
        _atomic_write_text(self.summary_path, json.dumps(summary, indent=2))
        return self.summary_path

    # ---- read --------------------------------------------------------------

    def read(self, query_id: str) -> ExploitBundle:
        """Load + validate one bundle by its query_id within this batch."""
        path = self.path_for(query_id)
        if not path.exists():
            raise FileNotFoundError(
                f"No bundle for query_id={query_id!r} in batch {self.batch_id!r} at {path}"
            )
        return ExploitBundle.from_json(path.read_text(encoding="utf-8"))

    def read_path(self, path: Path) -> ExploitBundle:
        """Load + validate a bundle from an explicit path."""
        return ExploitBundle.from_json(Path(path).read_text(encoding="utf-8"))

    def read_batch_summary(self) -> dict:
        """Load the batch-level summary JSON for this batch."""
        if not self.summary_path.exists():
            raise FileNotFoundError(
                f"No batch summary at {self.summary_path}"
            )
        return json.loads(self.summary_path.read_text(encoding="utf-8"))

    # ---- iterate -----------------------------------------------------------

    def list_paths(self) -> list[Path]:
        """Bundle paths in this batch, sorted by filename (stable ordering)."""
        return sorted(self.batch_dir.glob("run_*_bundle.json"))

    def __iter__(self) -> Iterator[ExploitBundle]:
        """Iterate all bundles in this batch. Validates each on read."""
        for path in self.list_paths():
            yield self.read_path(path)

    def __len__(self) -> int:
        """Number of bundle files in this batch."""
        return len(self.list_paths())


# ---------------------------------------------------------------------------
# Cross-batch helpers
# ---------------------------------------------------------------------------


def list_batch_dirs(root_dir: Path) -> list[Path]:
    """Return every batch folder under `root_dir`, sorted by name."""
    root = Path(root_dir)
    if not root.exists():
        return []
    return sorted(p for p in root.glob("batch_*") if p.is_dir())