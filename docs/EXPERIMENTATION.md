# Chapter 5 — Experimentation

> **Status: Day 9 first draft (messy-first, per PROJECT_SPEC.md line 367).** This chapter pins the *experiment configuration* — matrix shape, justified parameter values, execution plan, statistical reporting plan, and threats to validity. The *evaluation framework* (system, threat model, attack taxonomy, metrics) lives in `docs/METHODOLOGY.md` (Chapter 4) and is referenced rather than re-stated. Day 11 polishes both files together.

## 5.1 Scope vs. specification

`PROJECT_SPEC.md` row Day 9 fixes the headline matrix at *"~50 queries × 2 attacks × 3 seeds = ~300 runs"* (§2 line 24). The implemented system extends past this baseline in two directions, both during the Day 6–7.5 development phase:

- **Three families, not two** — the codebase implements `prompt_injection`, `corpus_poisoning`, and `query_injection` (the input-channel attack added in Day 7.5 from `FUTURE_WORKS.md` §2.1).
- **Two strategies for `corpus_poisoning`** — `answer_replacement` (the spec's PoisonedRAG-shaped integrity attack) and `jamming` (an availability attack scored by ASR-deny instead of ASR-t).

The Day-9 experimental scope evaluates **all four resulting attack cells**, not just the spec's two-family pair. This is a deliberate scope expansion, not scope creep: the four cells were already implemented, tested, and bundle-recorded by the end of Day 8, so Day 9 incurs only the *running cost* of the additional two cells, not their *implementation cost*. The dissertation framing is: *"the spec scoped two families; the implemented system extends to four attack cells covering both delivery channels and both adversarial objectives — this chapter evaluates them all."*

The spec's two-family pair (cells 1, 2 in the table below) remains the **primary integrity axis** of the analysis; cells 3 and 4 widen coverage without compromising the core comparison. All numbers in the Results chapter (Chapter 6) report per-cell so the spec's pair can be read in isolation if a reviewer wishes.

## 5.2 The four cells

The matrix is a 2 × 2 framing — *delivery channel* (corpus / query) × *adversarial objective* (integrity / availability):

| Cell | Family | Strategy | Channel | Objective | Headline metric |
| --- | --- | --- | --- | --- | --- |
| `ipi` | `prompt_injection` | `instruction_override` | corpus | integrity | ASR-t |
| `poiA` | `corpus_poisoning` | `answer_replacement` | corpus | integrity | ASR-t |
| `poiJ` | `corpus_poisoning` | `jamming` | corpus | **availability** | ASR-deny |
| `qInj` | `query_injection` | `prefix_injection` | query | integrity | ASR-a (ASR-r ≡ True) |

Cell labels (`ipi`, `poiA`, `poiJ`, `qInj`) are encoded into batch folder names so each (seed × cell) batch is identifiable from its directory alone — for example `results/runs/batch_seed42_poiJ_<UTC>/` is the seed-42 jamming-cell batch. The full registry with channel/objective metadata lives in `scripts/06_run_experiments.py:CELLS` and is also embedded in `experiment_manifest.json` so a downstream reader doesn't need to consult the source.

## 5.3 Justified parameters

Every parameter value in the matrix has a one-paragraph defence in this section. The principle is: *no value lands in the script without a written justification*, and every justification is anchored to either a citable spec section, a lab-notebook entry, or a prior empirical finding.

### 5.3.1 Number of seeds — 3 (`42`, `123`, `7`)

`PROJECT_SPEC.md` §6.4 (line 186) mandates *"Mean ± bootstrap 95% CI (1000 resamples) over n=3 seeds"*. Three is the smallest seed count that allows bootstrap CI estimation while still being computationally tractable inside the $50 OpenAI budget. The specific values — 42 (the canonical-meme seed), 123 (a triple-digit value), 7 (a single-digit prime) — are deliberately heterogeneous so any RNG-state coincidence between two of them is unlikely. All three values are short integers so the resulting `run_id` strings (`run_<ts>_seed<N>_<cell>_<query_id>`) stay under filesystem-path length limits on Windows.

The seed value drives three separable RNGs: (i) `Planner._rng` (controls the ε-greedy exploration choices in the sidecar), (ii) the per-payload RNG inside `generate_ipi`, `generate_poison_payload`, `generate_query_injection_payload` (controls within-template variant selection — e.g. topical-anchor term shuffling), (iii) any RAGAS-internal sampling. With `temperature=0.0` on the LLM the seed is the *only* source of inter-replica variance, which is what the n=3 statistic is measuring.

### 5.3.2 Number of queries per seed × cell — 50

Spec §2 line 24 sets the target at *"~50 queries"*. The `data/queries.json` test set is exactly 50 queries, stratified-sampled from BEIR/NQ to guarantee gold-doc coverage in the 1k-document Chroma slice (`LAB_NOTEBOOK.md` §0; `scripts/04_build_query_set.py`). Stratification ensures every query has at least one ground-truth correct doc indexed, which makes ASR-r interpretable: a 0 on a query with no gold doc would conflate "attack failed" with "no clean alternative existed".

50 samples × 3 seeds = **150 samples per cell** — sufficient for bootstrap 95% CIs at the per-cell level (rule-of-thumb ~30 minimum for stable CI estimation). A larger query set is logged in `FUTURE_WORKS.md` §3 as a power-improvement extension once the BEIR/NQ slice is generalised to additional benchmarks (HotpotQA, FiQA).

### 5.3.3 Iteration cap — 3 with early-exit on success

`max_iter = 3` with the existing `should_continue` early-exit (`graph.py:478`) means each cell runs its template-driven payload on iteration 0, then up to two LLM-generated variants (iterations 1 and 2) only if no prior iteration achieved the cell's success metric. The bound choice has two motivations:

- **RQ2 (planner adaptivity) needs more than one iteration.** A single-iteration sweep records only the template's success rate per cell — the dissertation's "agentic adaptation" contribution (the LLM-driven variant generator) is not exercised. With `max_iter ≥ 2`, the LLM exploit generator is invoked on at least one retry per failed iteration-0 attempt, producing the prompt-conditioned variants whose success rate forms part of RQ2's evidence.
- **Cost discipline caps at 3.** Each LLM-path iteration adds ~1 generator + 3 RAGAS calls = 4 LLM calls. At `max_iter = 3` the worst-case per cell is 12 LLM calls (1 baseline + 3 × 4), absorbed by `SQLiteCache` for repeat patterns. `max_iter = 5` or higher inflates the budget linearly without strong returns from observed Day-6/Day-8 LLM-exploit-gen behaviour (the LLM tends to converge on the same handful of variants by iteration 3).

Early-exit on success preserves the dissertation's *dominant-strategy* framing: when the template already wins on iteration 0, the LLM path is never invoked and the run reports `iterations_used = 1`. This statistic is recorded in each bundle's evaluation block and is reported in Chapter 6 as a *cost-per-success* signal.

### 5.3.4 RAGAS — ON for all 600 runs

`PROJECT_SPEC.md` §6.2 line 180 establishes the integrity-degradation criterion as *"a drop of ≥ 0.2 in Faithfulness between baseline and attacked condition counts as 'integrity-degraded'"*. Without RAGAS scores this criterion cannot be evaluated, and Chapter 6 loses one of its three contribution metrics (the spec's contribution stack is ASR + RAGAS + rank-shift).

The cost: each run with RAGAS adds three LLM calls (Faithfulness + Answer Relevance + Context Relevance) on top of the one generator call. For the 600-run matrix this is ~2,400 LLM calls in the worst case. Empirically, the LangChain `SQLiteCache` absorbs the bulk of duplicates because the *baseline* (clean-query) retrieval is identical across the four cells per (seed, query), so 75% of baseline-pass LLM calls hit cache after the first cell of each (seed, query) is run. The expected wall-clock budget is 60–90 minutes total; the spend tripwire is 90 minutes per spec line 368.

A sub-sampled-RAGAS variant (RAGAS on, e.g., 60 of 600 bundles) was considered and rejected during planning: it introduces a sub-sampling caveat that would need defending in the discussion chapter, in exchange for a small budget saving. The full-coverage option is cleaner.

### 5.3.5 Sweep strategy — forced Cartesian + ε-greedy sidecar

The ε-greedy planner (`Planner(epsilon=0.3)`) picks ONE family per query stochastically — appropriate behaviour when the agentic adaptation IS the experiment (RQ2), but the wrong behaviour when measuring per-cell ASR with statistical power (RQ1, RQ3). A planner-driven sweep concentrates on whichever family converges fastest in the first ~10 queries and undersamples the others, breaking per-cell CIs.

The chosen design is a **hybrid**:

- **Forced Cartesian for the headline matrix.** Each (seed, cell, query) cell is run exactly once via a `ForcedCellPlanner(family, strategy)` that pins both axes (`graph.py:103-138`). 600 bundles, balanced per cell.
- **ε-greedy planner as a per-seed sidecar log.** After the four cells finish for one seed, a fresh `Planner(epsilon=0.3, seed=seed)` is run against the same 50 queries; for each query the planner's `select()` choice is recorded, and the planner's `update()` is fed the actual ASR-t verdict from the just-completed cell sweep. Output: a 50-element selection sequence + a terminal `success_rate` snapshot, saved as `results/runs/sidecar_seed<N>_<ts>.json`. RQ2's answer is this sequence's convergence behaviour.

The sidecar's mapping caveat (planner has 3 families; matrix has 4 cells because `corpus_poisoning` has two strategies) is resolved by feeding the planner the `answer_replacement` cell's verdict whenever it picks `corpus_poisoning`, since `_DEFAULT_STRATEGY[corpus_poisoning] = answer_replacement`. The jamming cell remains a separate per-cell ASR-deny report.

### 5.3.6 Bundle layer — Day-8 schema, no changes

The experiment writes Day-8-shaped bundles unchanged (one `summary` block at the top, then `target_system`, `attack`, `execution`, `evaluation`, `reproducibility`). No new schema fields were needed — the schema's `attack.family` + `attack.strategy` + `attack.attack_channel` already distinguish all four cells, and `evaluation.asr_deny` was wired into `evaluate_node` on Day 8 specifically so the jamming cell would be Day-9-ready.

### 5.3.7 Output layout — 12 batches under `results/runs/`

The plan originally proposed *one batch per seed* (3 batches × 200 bundles each). Implementation review surfaced a filename-collision: `BundleStore.path_for(query_id)` produces `run_<query_id>_<batch_id>_bundle.json`, which would collide across the four cells run against the same query inside one batch. Two options were considered:

1. Modify `BundleStore` to disambiguate by cell — rejected, because the bundle layer is a frozen Day-8 contract (per the Day-9 plan's "do not touch `src/redteam/bundles/`" directive).
2. Use **one batch per (seed, cell) — 12 batches total**. Cell label in the batch_id (e.g., `seed42_poiJ_<ts>`), so the existing store stays frozen and Day 10 plotting can recover the (seed, cell) pair from the batch folder name without parsing each bundle.

Option 2 is what shipped. Each batch contains 50 bundles + 1 summary; the cross-batch link is `results/runs/experiment_manifest.json`, which lists every batch and its (seed, cell) tag. Day 10 reads the manifest first, then iterates over the listed batches.

### 5.3.8 Bundle gzipping — deferred to Day 13

`PROJECT_SPEC.md` §13 line 421 requires gzipped bundles at submission. Day 9 writes raw JSON to keep Day 10 plotting development friction-free (matplotlib + pandas readers can hit raw JSON without a decompression step in the inner loop). The Day-13 submission-prep step gzips the bundles in place; the file extensions become `.json.gz` and Day 10's plotting code is parameterised over the suffix.

### 5.3.9 Failure handling — per-cell try/except, no batch abort

If a single (seed, cell, query) run raises (network blip, RAGAS edge-case, etc.) the experiment driver catches the exception, prints a one-line error, and continues with the next query. This is the same pattern as `scripts/05_run_dryrun.py:178-182`. The rationale: one bad LLM response should not nuke 599 sibling bundles. The error count appears in the per-cell summary's `n_runs` (which is the count of successful bundles, not the input query count); a discrepancy is visible at a glance.

### 5.3.10 Index-rollback assertion — hard fail per cell

Corpus-channel attacks (cells 1, 2, 3) add the payload to Chroma, run the attacked retrieval, and remove the payload — the *index rollback contract* enforced by the executor (`graph.py:319-323`). The experiment driver verifies the contract at the cell level: pre/post `index_state_hash` must match, otherwise `SystemExit` is raised and the remaining cells in that seed are skipped. Failure of this check means a payload from cell N leaked into cell N+1's view of the corpus, and any subsequent ASR computation is contaminated.

## 5.4 Execution plan

Single command:

```
python scripts/06_run_experiments.py
```

defaults are: `--seeds 42 123 7 --limit 50 --max-iter 3 --with-ragas --out-dir results/runs --cells all`. Outputs:

- 12 batch folders under `results/runs/` (one per seed × cell), each with 50 bundle JSONs + 1 batch summary.
- 3 sidecar JSON files (`sidecar_seed<N>_<ts>.json`) — one per seed.
- 1 manifest at `results/runs/experiment_manifest.json` linking everything.

### 5.4.1 Wall-clock estimate

Day-8 dry run benchmark: 50 bundles in 14.7 s with RAGAS off and `max-iter = 1`. Day-9 multipliers:
- 4 cells (vs. 1 family in dry run) → ×4 base, less the cache hits on repeated baseline passes.
- RAGAS on (vs. off) → ~×4 LLM calls per run, mostly absorbed by SQLiteCache for cross-cell repeats of identical baseline retrievals.
- `max-iter = 3` (vs. 1) → up to ×3 in the worst case, but early-exit on the well-performing cells limits the actual factor.
- 3 seeds → ×3 outer.

Combined estimate: 60–90 minutes total. **Spend tripwire at 90 minutes** — pause the run and check the OpenAI dashboard before continuing.

### 5.4.2 Restart semantics

The seed loop is restartable. If seed 42 completes successfully and seed 123 fails partway, the user can:

```
python scripts/06_run_experiments.py --seeds 123
```

to redo only seed 123. Each invocation produces a fresh `<batch_ts>` so the redone seed's batch folders do not collide with the original's; the manifest will need a manual merge (a Day-10 helper for this is logged in `FUTURE_WORKS.md`).

A single-cell investigation (e.g. *"why is `poiJ` ASR-deny so low?"*) is supported by `--cells poiJ`.

## 5.5 Verification gates

Three gates, all required to pass before the experiment is taken as authoritative.

### 5.5.1 Unit gate

`python -m pytest tests/` must remain at 76/76 green (75 pre-Day-9 + 1 new `test_forced_cell_planner_drives_graph`). Confirms the `ForcedCellPlanner` + `make_plan_node` strategy-override don't regress any earlier behaviour.

Result for the Day-9 build: **76/76 passed** in 157s.

### 5.5.2 Smoke gate

```
python scripts/06_run_experiments.py --smoke
```

is hard-coded to `--limit 2 --max-iter 1 --no-ragas --seeds <first-only>` and writes to `results/runs_smoke/`. Expected: 8 bundles (2 queries × 4 cells × 1 seed), all four batches with `rollback_ok = true`, and a manifest. Wall-clock target: under 90 s.

Result for the Day-9 build: **8/8 bundles in 5.0 s**, all four cells exercised, `summary` block at top of each JSON, attack family/strategy correctly recorded per cell, manifest produced. The smoke directory is removed after passing.

### 5.5.3 Full-run gate

The 600-run job. Post-run sanity checks:

- `find results/runs -name '*_bundle.json' | wc -l` → 600.
- All 12 batch summaries report `rollback_ok = true`.
- Per-seed roll-up (`experiment_manifest.json`) shows non-zero ASR-t for cells `ipi`, `poiA`, `qInj` and non-zero ASR-deny for cell `poiJ` — confirming all four cells are firing on at least some queries.
- Spot-check one bundle from each cell × each seed (12 bundles): `summary` block first, expected `attack.family/strategy` pair, `evaluation.ragas_faithfulness` is a float (not `None`).
- Each `sidecar_seed<N>_<ts>.json` has 50 entries in `selections` and a non-trivial `final_snapshot.success_rate`.

Failure of any post-run check **blocks Day 10**. Investigation pattern: re-run the affected seed in isolation (`--seeds <N>`) and inspect the per-bundle `evaluator_notes` field for RAGAS errors or per-cell anomalies.

## 5.6 Statistical reporting plan (Chapter 6 will lift this)

For each cell, three reports:

1. **Headline success rate**: ASR-t (cells 1, 2, 4) or ASR-deny (cell 3) — mean across 150 samples (50q × 3 seeds), bootstrap 95% CI from 1000 resamples. Plot: bar chart with CI whiskers, four bars (one per cell).
2. **Faithfulness drop**: mean (Faithfulness_baseline − Faithfulness_attacked) per cell, bootstrap 95% CI. The criterion *"Faithfulness drop ≥ 0.2 = integrity-degraded"* (PROJECT_SPEC.md line 180) is reported as the proportion of runs satisfying the criterion.
3. **Rank-shift footprint**: distribution of `rank_shift_at_k` per cell (0 = no shift, ..., 5 = top-1 dropped out of top-5). Plot: stacked bar.

Pairwise comparisons against the IPI cell (cell 1, the spec's headline integrity attack) by paired bootstrap difference:
- For each (seed, query), pair the IPI verdict with the corresponding cell-N verdict.
- Compute `Δ = ASR-t_cellN − ASR-t_ipi` for each pair.
- Bootstrap-resample the pair list 1000 times; report the 95% CI of `Δ`.
- A 95% CI excluding 0 is the dissertation's significance criterion (see PROJECT_SPEC.md §12 marking-scheme line 405-407 for the rigour bar).

Effect-size reporting: for each cell, alongside the CI, report Cohen's *h* between the cell's success rate and IPI's success rate.

## 5.7 Threats to validity

Each receives a sentence and a forward pointer; the discussion chapter (Chapter 7) will expand them.

- **Seed correlation**: with n=3 seeds, the CI is wide and small per-cell mean differences may not be detectable. Mitigation: report effect sizes alongside CIs; pre-register the IPI-baseline comparison.
- **Query-set bias**: the 50-query NQ slice is stratified for gold-doc coverage but not for question type or domain. Generalisation to other RAG benchmarks is a Future Work item.
- **Refusal-lexicon brittleness for ASR-deny**: the lexicon is gpt-4o-mini-shaped (`asr.py:120-122`). A model that refuses with a lexically distinct phrase ("I don't believe so", "There's no consensus") would produce ASR-deny = 0 even when the attack succeeded behaviourally. Mitigation: the lexicon is module-level, swap-in-friendly; an LLM-judge variant is logged in `FUTURE_WORKS.md` §5.2.
- **gpt-4o-mini specificity**: the attack-success rates are model-specific. The paper does not generalise to "all RAG systems" — claims are scoped to the specific (retriever, embedder, LLM) triple.
- **Index-leak detection is hash-based**: the executor guarantees rollback; the script asserts pre/post `index_state_hash` equality. A pathological attack that adds AND removes the same content via different code paths would leave the hash unchanged but might leave a residual side-effect; this is theoretically possible but no observed instance.
- **Cache-warming may bias latency**: `generator_latency_ms` is recorded as wall-clock from prompt-send to response-receive. Cached responses report sub-millisecond latency; un-cached responses report network + model time. Plots that sum or average raw latency need to filter for cache hits — a Day-10 plotting note.

## 5.8 Reproduction recipe (for the dissertation appendix)

For a third party to reproduce the experiment from a clean checkout:

```
git clone <repo>
cd agentic-red-team-framework
pip install -e .
cp .env.example .env  # add OPENAI_API_KEY
python scripts/01_build_corpus.py        # builds Chroma index
python scripts/04_build_query_set.py     # writes data/queries.json
python -m pytest tests/                  # 76/76 green
python scripts/06_run_experiments.py --smoke   # 8 bundles in <90s, sanity check
python scripts/06_run_experiments.py     # full 600-run job, ~60-90 minutes
```

The expected output is `results/runs/` populated as described in §5.4. Bundles are byte-deterministic given the `git_commit` recorded in each bundle's `reproducibility.git_commit` field, modulo the API-key-dependent gpt-4o-mini cache state. The cache is included in the repo's `.cache.sqlite` so the second run matches the first under the same git commit.
