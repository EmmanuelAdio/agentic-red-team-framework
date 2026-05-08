"""Exploit-bundle layer — schema, builder, store.

Public surface re-exported for ergonomics:

- :class:`ExploitBundle` and its sub-models: the spec §7 JSON shape.
- :func:`build_bundle`: project a finished `RedTeamState` into a bundle.
- :class:`BundleStore`: filesystem read/write with atomic writes.
"""

from redteam.bundles.builder import build_bundle
from redteam.bundles.schema import (
    BUNDLE_VERSION,
    FRAMEWORK_VERSION,
    AttackBlock,
    BundleSummary,
    EvaluationBlock,
    ExecutionBlock,
    ExploitBundle,
    Reproducibility,
    RetrievedDocRecord,
    TargetSystem,
    utc_now_iso,
)
from redteam.bundles.store import BundleStore, list_batch_dirs

__all__ = [
    "BUNDLE_VERSION",
    "FRAMEWORK_VERSION",
    "AttackBlock",
    "BundleStore",
    "BundleSummary",
    "EvaluationBlock",
    "ExecutionBlock",
    "ExploitBundle",
    "Reproducibility",
    "RetrievedDocRecord",
    "TargetSystem",
    "build_bundle",
    "list_batch_dirs",
    "utc_now_iso",
]