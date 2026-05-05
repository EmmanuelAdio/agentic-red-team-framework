# Notebooks

Per spec §8 these notebooks demonstrate the framework end-to-end and double as
Chapter 5/6 evidence.

| Notebook | Purpose | First touched |
| --- | --- | --- |
| `02_attack_dev.ipynb` | IPI + corpus-poisoning demos, cell-by-cell | Day 3 (IPI), extended Day 4 |

## Workflow

1. Activate the venv (`.\.venv\Scripts\Activate.ps1` on Windows).
2. Open the notebook, run **Kernel → Restart & Run All** before committing — this guarantees the persisted outputs match the source code in the same commit.
3. Commit the `.ipynb` with executed outputs. Reviewers (and examiners) read these directly without needing to re-run.

If outputs fall out of sync (e.g., you re-run only some cells), re-execute top to bottom before staging.
