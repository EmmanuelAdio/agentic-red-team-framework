"""Migrate historical run bundles to the headline-metric verdict rule.

Background
----------

Before this script, every run bundle's ``evaluation.verdict`` field was
written by the orchestrator's ``evaluate_node`` using an integrity-only
rule (``"success"`` iff ``asr_target``; else ``"partial"`` on
``asr_retrieval``; else ``"failure"``). That rule mis-classifies the
jamming / availability cell (``poiJ``, headline metric ``asr_deny``),
whose successful runs end in a refusal output — the attacker's marker
substring is never present, so ``asr_target`` is structurally False and
every successful jam landed in the bundle JSON as ``"failure"``. The
Day-10 fix patched the loop's early-exit predicate
(``should_continue``) but left the *written* verdict still keyed off
``asr_target``.

The new rule (see :func:`redteam.metrics.verdict.compute_verdict`)
consults the cell's *headline* success metric: jamming cells are
judged binary on ``asr_deny`` alone; integrity cells keep the
three-way scheme. The orchestrator was patched in the same change to
emit the corrected verdict for new runs. This script back-fills the
~600 existing bundles on disk so the dashboard's Recent-runs chips,
the per-cell aggregates, and any downstream analysis see consistent
values across the historical corpus.

What it rewrites
----------------

For each ``*_bundle.json`` under ``results/runs/`` and ``data/runs/``:

* ``summary.verdict``
* ``evaluation.verdict``
* every ``evaluation.iteration_history[*].verdict`` (so the per-iteration
  history visible in the bundle viewer matches the final verdict's
  semantics).

For each per-batch ``*_summary.json`` under the same roots, each
``runs[*].verdict`` row is also recomputed — these snapshots are read
by the dashboard's experiment-manifest tables and the Day-9 plotting
code.

Idempotent: bundles already consistent with the new rule are skipped
without a write (`--force` rewrites them anyway for byte-equality
checking).

CLI
---

::

    # Default: dry-run preview, then apply if anything would change.
    python scripts/09_migrate_verdicts.py

    # Preview only — print per-cell flip counts, do not touch disk.
    python scripts/09_migrate_verdicts.py --dry-run

    # Skip the dry-run preview and just apply.
    python scripts/09_migrate_verdicts.py --apply

    # Re-serialise every bundle even if its verdict is already correct.
    python scripts/09_migrate_verdicts.py --apply --force

Verifying the migration after it runs
-------------------------------------

* ``git diff results/runs/ | head -200`` — spot-check a handful of
  edited bundles.
* The Streamlit dashboard's Recent-runs table (with the ASR-deny column
  toggled on) should now show ``success`` on every poiJ row whose
  ASR-deny is True, and ``defended`` (the CSS label for ``failure``) on
  every poiJ row whose ASR-deny is False. Integrity-cell rows should be
  unchanged.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

# Make `redteam` importable when running the script directly without an
# editable install (matches the pattern used by the other Day-N scripts
# in this directory).
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from redteam.metrics.verdict import compute_verdict  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[1]
_BUNDLE_GLOB = "**/*_bundle.json"
_BATCH_SUMMARY_GLOB = "**/batch_*_summary.json"

# The only (family, strategy) pair whose headline is the availability
# metric. Mirrors the dashboard's CELL_REGISTRY entry for ``poiJ`` but
# kept local so this script does not import from the dashboard package
# (which carries Streamlit-side dependencies the migration shouldn't
# need at runtime).
_JAMMING_KEY: tuple[str, str] = ("corpus_poisoning", "jamming")


def _success_metric_for(family: str, strategy: str) -> str:
    """Return the success metric a bundle of this cell should be judged on.

    Matches the user-stated rule "jamming uses ASR-deny, everything else
    uses ASR-t". The CELL_REGISTRY's ``"asr_a"`` entry for qInj is a
    dashboard-aggregation choice; here we judge qInj on ``asr_target``,
    which is identical in practice (qInj forces ASR-r=True so
    ASR-t == ASR-a anyway) and avoids cross-importing the dashboard
    registry.
    """
    if (family, strategy) == _JAMMING_KEY:
        return "asr_deny"
    return "asr_target"


def _new_bundle_verdict(bundle: dict[str, Any]) -> str | None:
    """Return the corrected verdict for a bundle, or None if it's malformed.

    Reads the integrity triple + ``asr_deny`` straight off the bundle's
    ``evaluation`` block (the canonical location) and looks up the cell's
    headline metric from the ``attack`` block.
    """
    attack = bundle.get("attack") or {}
    evaluation = bundle.get("evaluation") or {}
    family = attack.get("family")
    strategy = attack.get("strategy")
    if family is None or strategy is None:
        return None
    return compute_verdict(
        success_metric=_success_metric_for(family, strategy),
        asr_retrieval=bool(evaluation.get("asr_retrieval", False)),
        asr_target=bool(evaluation.get("asr_target", False)),
        asr_deny=bool(evaluation.get("asr_deny", False)),
    )


def _new_history_verdict(
    history_row: dict[str, Any], *, success_metric: str
) -> str:
    """Return the corrected verdict for one ``iteration_history`` entry.

    The history row carries its own ASR triple + ``asr_deny`` snapshot
    (per-iteration values, which differ from the bundle's final
    evaluation block when the loop ran multiple iterations). The
    success metric is the *cell's* — every iteration in a bundle
    belongs to the same cell, so the metric is constant across the
    history list and supplied by the caller.
    """
    return compute_verdict(
        success_metric=success_metric,
        asr_retrieval=bool(history_row.get("asr_retrieval", False)),
        asr_target=bool(history_row.get("asr_target", False)),
        asr_deny=bool(history_row.get("asr_deny", False)),
    )


def _new_batch_runs_verdict(
    runs_row: dict[str, Any], *, fallback_success_metric: str
) -> str | None:
    """Return the corrected verdict for one ``runs[]`` entry in a batch summary.

    The batch summary's per-run rows carry ``attack_family`` and
    ``attack_strategy`` (so we can re-derive the metric per row) but
    not every legacy batch did — ``fallback_success_metric`` is read
    from the batch's top-level ``cell_meta.success_metric`` if
    present, else from the same per-(family, strategy) lookup the
    bundles use.
    """
    family = runs_row.get("attack_family")
    strategy = runs_row.get("attack_strategy")
    if family is not None and strategy is not None:
        metric = _success_metric_for(family, strategy)
    else:
        metric = fallback_success_metric
    # The batch summary uses the same field names as the bundle's
    # evaluation block, so no key translation needed.
    if "asr_target" not in runs_row and "asr_deny" not in runs_row:
        return None
    return compute_verdict(
        success_metric=metric,
        asr_retrieval=bool(runs_row.get("asr_retrieval", False)),
        asr_target=bool(runs_row.get("asr_target", False)),
        asr_deny=bool(runs_row.get("asr_deny", False)),
    )


def _migrate_bundle(
    path: Path, *, force: bool
) -> tuple[bool, str | None, str | None, str]:
    """Read, rewrite, and return change metadata for one bundle file.

    Returns ``(changed, old_verdict, new_verdict, cell_label)`` where
    ``changed`` is True iff the on-disk file would (or did) change.
    ``cell_label`` is a short identifier of the form ``"<family>/<strategy>"``
    used only to bucket the report's flip counts.
    """
    raw = path.read_text(encoding="utf-8")
    bundle = json.loads(raw)
    attack = bundle.get("attack") or {}
    family = attack.get("family") or "?"
    strategy = attack.get("strategy") or "?"
    cell_label = f"{family}/{strategy}"

    new_verdict = _new_bundle_verdict(bundle)
    if new_verdict is None:
        return (False, None, None, cell_label)

    summary = bundle.setdefault("summary", {})
    evaluation = bundle.setdefault("evaluation", {})
    old_verdict = evaluation.get("verdict")

    # Rewrite the top-level summary + evaluation verdicts. ``summary``
    # is just a mirror of the evaluation block (per builder.py), so they
    # must agree post-migration.
    evaluation["verdict"] = new_verdict
    summary["verdict"] = new_verdict

    # Rewrite every per-iteration history entry under the same rule.
    # The bundle stores these under ``evaluation.iteration_history``.
    history = evaluation.get("iteration_history") or []
    metric = _success_metric_for(family, strategy)
    for row in history:
        if not isinstance(row, dict):
            continue
        row["verdict"] = _new_history_verdict(row, success_metric=metric)

    changed = old_verdict != new_verdict
    if not changed and not force:
        # Idempotent skip — same verdict, same bytes, no write.
        return (False, old_verdict, new_verdict, cell_label)

    # Re-serialise with the same formatting builder.py uses
    # (json.dumps(..., indent=2) + trailing newline). The builder hard-codes
    # indent=2 and does not sort keys, so we match exactly to keep the
    # ``git diff`` minimal — only the verdict literal flips.
    new_raw = json.dumps(bundle, indent=2, ensure_ascii=False) + "\n"
    path.write_text(new_raw, encoding="utf-8")
    return (changed, old_verdict, new_verdict, cell_label)


def _migrate_batch_summary(
    path: Path, *, force: bool
) -> tuple[int, Counter[tuple[str, str, str]]]:
    """Rewrite per-run verdicts inside a batch summary. Returns (changed_count, flips).

    ``flips`` is a Counter keyed by ``(cell_label, old, new)`` so the
    report can show "poiJ partial -> failure: N" lines.
    """
    raw = path.read_text(encoding="utf-8")
    payload = json.loads(raw)
    runs = payload.get("runs") or []
    cell_meta = payload.get("cell_meta") or {}
    fallback_metric = cell_meta.get("success_metric") or "asr_target"

    flips: Counter[tuple[str, str, str]] = Counter()
    changed_rows = 0
    for row in runs:
        if not isinstance(row, dict):
            continue
        new_verdict = _new_batch_runs_verdict(
            row, fallback_success_metric=fallback_metric
        )
        if new_verdict is None:
            continue
        family = row.get("attack_family") or "?"
        strategy = row.get("attack_strategy") or "?"
        cell_label = f"{family}/{strategy}"
        old_verdict = row.get("verdict")
        if old_verdict != new_verdict:
            row["verdict"] = new_verdict
            changed_rows += 1
            flips[(cell_label, str(old_verdict), new_verdict)] += 1

    if changed_rows == 0 and not force:
        return (0, flips)

    new_raw = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    path.write_text(new_raw, encoding="utf-8")
    return (changed_rows, flips)


def _scan_bundles(roots: list[Path]) -> list[Path]:
    """Return every bundle JSON found beneath the given roots."""
    found: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        found.extend(sorted(root.glob(_BUNDLE_GLOB)))
    return found


def _scan_batch_summaries(roots: list[Path]) -> list[Path]:
    """Return every batch-summary JSON found beneath the given roots."""
    found: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        found.extend(sorted(root.glob(_BATCH_SUMMARY_GLOB)))
    return found


def _report(flips: Counter[tuple[str, str, str]], *, header: str) -> None:
    """Print a per-cell breakdown of the (old -> new) verdict flips."""
    print(header)
    if not flips:
        print("  (no changes)")
        return
    # Sort by cell, then by flip arrow for readable output.
    for (cell, old, new), n in sorted(flips.items()):
        print(f"  {cell:>40s}  {old:>8s} -> {new:<8s}  n={n}")


def _dry_run(
    bundle_paths: list[Path], batch_paths: list[Path]
) -> tuple[Counter[tuple[str, str, str]], Counter[tuple[str, str, str]]]:
    """Compute would-be flips without touching disk."""
    bundle_flips: Counter[tuple[str, str, str]] = Counter()
    for path in bundle_paths:
        bundle = json.loads(path.read_text(encoding="utf-8"))
        attack = bundle.get("attack") or {}
        family = attack.get("family") or "?"
        strategy = attack.get("strategy") or "?"
        cell_label = f"{family}/{strategy}"
        old_verdict = (bundle.get("evaluation") or {}).get("verdict")
        new_verdict = _new_bundle_verdict(bundle)
        if new_verdict is None or old_verdict == new_verdict:
            continue
        bundle_flips[(cell_label, str(old_verdict), new_verdict)] += 1

    batch_flips: Counter[tuple[str, str, str]] = Counter()
    for path in batch_paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        cell_meta = payload.get("cell_meta") or {}
        fallback_metric = cell_meta.get("success_metric") or "asr_target"
        for row in payload.get("runs") or []:
            if not isinstance(row, dict):
                continue
            new_verdict = _new_batch_runs_verdict(
                row, fallback_success_metric=fallback_metric
            )
            if new_verdict is None:
                continue
            family = row.get("attack_family") or "?"
            strategy = row.get("attack_strategy") or "?"
            cell_label = f"{family}/{strategy}"
            old_verdict = row.get("verdict")
            if old_verdict != new_verdict:
                batch_flips[(cell_label, str(old_verdict), new_verdict)] += 1

    return bundle_flips, batch_flips


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print would-be changes without touching any file.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Skip the dry-run preview and apply immediately.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-serialise every file even if its verdict is already correct.",
    )
    parser.add_argument(
        "--roots",
        nargs="+",
        default=[str(_REPO_ROOT / "results" / "runs"),
                 str(_REPO_ROOT / "data" / "runs")],
        help="Directories to scan for *_bundle.json and *_summary.json files.",
    )
    args = parser.parse_args()

    roots = [Path(r) for r in args.roots]
    bundle_paths = _scan_bundles(roots)
    batch_paths = _scan_batch_summaries(roots)
    print(
        f"Scanned {len(roots)} root(s) — found {len(bundle_paths)} "
        f"bundle(s) and {len(batch_paths)} batch summary file(s)."
    )

    # Preview pass.
    bundle_flips, batch_flips = _dry_run(bundle_paths, batch_paths)
    _report(bundle_flips, header="\nBundle verdict flips (preview):")
    _report(batch_flips, header="\nBatch summary verdict flips (preview):")
    total_changes = sum(bundle_flips.values()) + sum(batch_flips.values())

    if args.dry_run:
        print(f"\nDry-run complete — would change {total_changes} verdict(s).")
        return 0

    if total_changes == 0 and not args.force:
        print("\nNothing to migrate — every verdict is already consistent.")
        return 0

    # Confirm before writing, unless --apply was passed explicitly.
    if not args.apply:
        reply = input(
            f"\nApply {total_changes} verdict change(s) in place? [y/N] "
        ).strip().lower()
        if reply not in {"y", "yes"}:
            print("Aborted — no files written.")
            return 1

    # Apply pass.
    bundle_change_count = 0
    for path in bundle_paths:
        changed, _old, _new, _cell = _migrate_bundle(path, force=args.force)
        if changed:
            bundle_change_count += 1

    batch_change_count = 0
    batch_apply_flips: Counter[tuple[str, str, str]] = Counter()
    for path in batch_paths:
        changed_rows, file_flips = _migrate_batch_summary(path, force=args.force)
        batch_change_count += changed_rows
        batch_apply_flips.update(file_flips)

    print(
        f"\nWrote {bundle_change_count} bundle file(s) and updated "
        f"{batch_change_count} batch-summary row(s)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
