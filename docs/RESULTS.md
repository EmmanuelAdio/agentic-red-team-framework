# Chapter 6 — Results

**Source files**
- Tables: `results/tables/{summary_by_cell,ragas_by_cell,paired_differences_vs_ipi,baseline_summary}.csv`
- Figures: `results/figures/{asr_triple_by_cell,channel_objective_heatmap,asr_r_vs_k,asr_deny_by_cell,ragas_triple_clean_vs_attacked,rank_shift_ecdf,planner_adaptation,poij_outcome_decomposition}.pdf` (PNG counterparts beside each PDF; eight figures total, F1–F8)
- Flat summary: `results/summary.json`
- Demo notebook: `notebooks/03_results_analysis.ipynb` (mirrors this chapter cell-by-cell)
- Plotter: `python scripts/08_make_plots.py` regenerates every figure and table

---

## 6.1 Reading guide

This chapter reports the **Day-9 attack matrix**: 50 Natural Questions queries x 4 attack cells x 3 seeds = **600 reproducible exploit bundles**, evaluated against a 50-query clean baseline. The four cells span the framework's 2-channel x 2-objective taxonomy:

| Cell | Family | Strategy | Channel | Objective | Success metric |
| --- | --- | --- | --- | --- | --- |
| `ipi`  | prompt injection  | instruction override  | corpus | integrity    | `asr_target` |
| `poiA` | corpus poisoning  | answer replacement    | corpus | integrity    | `asr_target` |
| `poiJ` | corpus poisoning  | jamming               | corpus | availability | `asr_deny`   |
| `qInj` | query injection   | prefix injection      | query  | integrity    | `asr_answer` |

Every percentage in this chapter is a **mean with a 95% percentile bootstrap CI over 1000 resamples** (seed pinned at 12345 for byte-reproducibility). The seven figures (F1–F7) follow conventions from the comparable literature — PoisonedRAG (Zou et al., 2024), AgentPoison (Chen et al., 2024), GASLITE (2024), and the IR rank-shift convention — so the reader can place this work alongside the references without retraining their eye.

## 6.2 Clean baseline validation

Before attacking the pipeline we measure how it behaves under no attack: `scripts/07_run_clean_baseline.py` runs each of the 50 queries through the unattacked RAG pipeline and persists the RAGAS triple per query to `results/baseline/baseline_latest.json`.

`baseline_summary.csv`:

- `asr_retrieval_clean_rate` = **1.00** — the gold-passage doc appears in the top-5 retrieval on every one of the 50 queries.
- `top1_is_gold_rate` = **1.00** — and is in fact at rank 1 every time.
- `mean_ragas_faithfulness` = **0.875**, `mean_ragas_answer_relevance` = **0.666**, `mean_ragas_context_relevance` = **0.850**.

The retrieval ceiling is therefore not the bottleneck; the clean RAGAS triple establishes a stable reference point against which the attacked-condition figures (F5) are compared.

## 6.3 Attack success across the four cells

`summary_by_cell.csv` rows (headline-success ± 95% CI, n=150 per cell):

| Cell | Channel × Objective | Headline success | 95% CI |
| --- | --- | --- | --- |
| `ipi`  | corpus × integrity    | **0.960** | [0.920, 0.987] |
| `poiA` | corpus × integrity    | **0.800** | [0.733, 0.860] |
| `poiJ` | corpus × availability | **0.460** | [0.387, 0.540] |
| `qInj` | query × integrity     | **0.960** | [0.927, 0.987] |

![Figure F1 — ASR-r / ASR-a / ASR-t per attack cell](../results/figures/asr_triple_by_cell.pdf)

![Figure F2 — Headline attack success by channel × objective](../results/figures/channel_objective_heatmap.pdf)

**Reading F1.** ASR-r is 1.00 for every cell — every poisoned document made the top-5 retrieval, and `qInj`'s ASR-r is True by construction (the attack rewrites the query, it does not poison the corpus). The drop from ASR-r to ASR-a/ASR-t localises where each attack loses ground: `ipi` and `qInj` keep ~96% of their retrieval success through generation; `poiA` loses 20% of trials between retrieval and answer (the LLM caught the false-answer span on those queries). The `poiJ` bar shows ASR-t = 0.34 — *not* an integrity success rate to celebrate but the slice of jamming queries on which the LLM emitted the attacker's target span instead of refusing; the cell's headline metric is ASR-deny, reported in F4, and the full outcome decomposition is in F8.

**Reading F2.** The 2×2 heatmap restates the same result in the framework's own taxonomy. Both integrity cells are very effective (0.80 and 0.96), and the corpus × availability cell lands at 0.46 — jamming is now a working availability attack rather than the negative-result placeholder it appeared to be before the Day-10 orchestration fix (§6.5.1). The empty query × availability quadrant is the part of the threat surface the framework deliberately does not implement — there is no query-channel attack in this scope that targets availability.

## 6.4 Dose-response: ASR-r vs retrieval depth k

PoisonedRAG (Zou et al., 2024) reports ASR vs retrieval depth k to expose the trade-off a defender faces between recall and exposure. F3 reproduces that view for our three corpus-channel cells; `qInj` is excluded because its ASR-r is independent of k by construction.

![Figure F3 — ASR-r vs retrieval depth k](../results/figures/asr_r_vs_k.pdf)

**Reading F3.** `poiA` and `poiJ` both reach ASR-r ≈ 0.85 at k = 1 — their exploit-generated payloads almost always claim the rank-1 slot, because the generator optimises payload surface-similarity to the query. `ipi` starts at 0.30 at k = 1 and ramps to 1.00 by k = 4: the injection payload is a short imperative string whose embedding similarity to the query is *lower* than the original gold passage, so the poisoned doc sits mid-rank. The implication for a defender is concrete: truncating retrieval at k = 1 cuts `ipi`'s ASR-r by two-thirds, but pays nothing against `poiA`/`poiJ` whose payloads survive aggressive truncation.

## 6.5 The availability cell: jamming as a coerced refusal

The jamming cell's success metric is **ASR-deny**: did the LLM refuse to answer, or otherwise return a non-answer? F4 reports ASR-deny for every cell, and the companion figure F8 decomposes the `poiJ` runs into the four mutually-exclusive outcomes defined by the `(asr_deny, asr_target)` pair.

![Figure F4 — ASR-deny per cell](../results/figures/asr_deny_by_cell.pdf)

![Figure F8 — poiJ outcome decomposition](../results/figures/poij_outcome_decomposition.pdf)

**Reading F4.** The three integrity cells sit at exactly 0% — they do not aim to coerce a refusal, and an integrity success does not accidentally trigger a refusal prefix. `poiJ` lands at **0.46** [0.387, 0.540] — the framework's jamming payload, delivered via the iter-0 template path, reliably coerces the LLM into a refusal on close to half of all queries. The CI is tight because the iter-0 payload is deterministic per query, and an identical 23/50 ASR-deny is observed in each of the three seeds (43/150 across the full matrix) — the seed dimension contributes zero additional variance for this metric.

**Reading F8.** F4 reports the headline but does not show what happened on the 54% of `poiJ` queries that did *not* refuse. F8 splits the 150 runs into four buckets:

- **Refused** (asr_deny=True, asr_target=False) — 46%, the jamming win.
- **Target hit** (asr_deny=False, asr_target=True) — 34%, runs where the LLM emitted the attacker's target span rather than refusing. This is best read as a **collapsed-attack-mode artefact** of the exploit generator's iter-0 template: when the refusal does not fire, the same payload structure (which advertises a fabricated "fact" about the query topic) is also a plausible answer-replacement vehicle. Chapter 7 §7.3 distinguishes this from a deliberate integrity attack.
- **Other answer** (asr_deny=False, asr_target=False) — 20%, benign-looking responses the attacker did not control.
- **Refused + target** (both True) — 0% by construction; a refusal prefix and a target marker substring are structurally incompatible. Drawn for completeness so a future regression would surface visibly.

The 34% ASR-t for `poiJ` should therefore not be read as "the jamming attack achieves 34% integrity success on top of 46% availability success" — they're disjoint outcomes on the same set of queries. See Chapter 7 for why temperature-0 RAGs with an "answer from context" instruction are nevertheless coercible into refusal by the project's jamming payload, and `FUTURE_WORKS.md` §3 for the harder availability attack families (retrieval-flooding, context-overflow) the 46% baseline now motivates.

### 6.5.1 Methodology note: the early-exit fix

The original Day-9 measurement of this cell reported **0.000 ASR-deny** — a structurally wrong number caused by a bug in the orchestration loop's exit predicate, not by a property of the underlying attack. The `should_continue` function in [src/redteam/orchestration/graph.py:515-524](../src/redteam/orchestration/graph.py#L515-L524) terminated only when `verdict == "success"`, and `verdict` was derived solely from `asr_target`. For the integrity cells this aligned: ASR-t success caused early exit and the bundle on disk reflected that success. For the availability cell it did not: an iter-0 `asr_deny=True` did *not* match `verdict=="success"`, so the loop continued into iter 1+, where the LLM exploit generator was told the previous attempt "failed" and silently switched attack mode — typically rewriting the payload into an answer-replacement variant. The new iteration overwrote the bundle's evaluation block, and the original availability win was lost from the on-disk record.

Forensic scan of the 150 pre-fix `poiJ` bundles' `iteration_history`: **69 runs** achieved `asr_deny=True` on some iteration but **0 retained it at the recorded exit**; of those 69, **51** flipped to `asr_target=True` at iter 1+ (silently re-attributed as integrity wins, inflating the cell's reported ASR-t to 0.72) and **18** ended in `verdict=partial` with both `asr_target=False` and `asr_deny=False` (legitimate availability wins lost outright). The bug therefore both under-reported ASR-deny (0% vs 46%) and over-reported ASR-t (72% vs 34%) for this cell.

The fix is small: `ForcedCellPlanner` gained a `success_metric: str = "asr_target"` field; `make_plan_node` surfaces it into state; `should_continue` now consults `state["success_metric"]` and treats the named boolean as the terminal signal. The default preserves the legacy behaviour for the three integrity cells and for callers that don't set the metric (the ε-greedy and round-robin planners). Eight unit tests in [tests/test_orchestration.py](../tests/test_orchestration.py) pin the predicate's behaviour across both the integrity-default and availability-explicit paths. Per-objective attribution on the dashboard side and a `summary_by_cell` per-cell rollup keep the integrity headline (averaged over `ipi + poiA + qInj`) and the availability headline (over `poiJ`) cleanly separated in the Overview UI.

## 6.6 Integrity degradation under attack

`ragas_by_cell.csv` (mean ± 95% CI):

| Cell | Attacked Faithfulness | Faithfulness drop vs clean | Integrity degraded ≥ 0.2 |
| --- | --- | --- | --- |
| `ipi`  | 0.613 [0.546, 0.681] | **+0.262** [+0.168, +0.351] | 46.7% |
| `poiA` | 0.919 [0.883, 0.953] | -0.044 [-0.093, +0.010]     |  9.3% |
| `poiJ` | 0.933 [0.893, 0.969] | -0.058 [-0.115, +0.005]     |  8.0% |
| `qInj` | 0.282 [0.219, 0.348] | **+0.593** [+0.507, +0.673] | 73.3% |

![Figure F5 — RAGAS triple: clean baseline vs each attacked cell](../results/figures/ragas_triple_clean_vs_attacked.pdf)

**Reading F5.** The Faithfulness panel surfaces this chapter's most important methodological caveat: a high ASR-t **does not** imply a Faithfulness drop. `poiA` reaches 80% ASR-t while its mean Faithfulness *rises* slightly above the clean baseline (0.919 vs 0.875); the poisoned context literally contains a statement supporting the attacker's false answer, so the RAGAS Faithfulness scorer (which asks "is every claim in the answer supported by the context?") rates the answer as faithful even though it is factually wrong.

`ipi` and `qInj` show the expected pattern — large Faithfulness drops (0.262 and 0.593) and high integrity-degraded rates (47% and 73%). The Answer-Relevance panel separates the cells into three regimes: `ipi` and `qInj` collapse AR to ~0.03 (the injected text is unrelated to the original query); `poiA` keeps AR high (~0.79) because its answers are still topically relevant, just wrong; `poiJ` lands in the middle at **AR ≈ 0.42** — refusals are concise and do not reference the query text, so RAGAS scores them as only marginally relevant even though they're a legitimate availability win on the cell's own metric. This is the same decoupling story as `poiA` mirrored: Faithfulness is *higher* than baseline (0.933 — refusals are by definition faithful to context, they make no claims), AR is low, and the cell succeeds on the metric that matters for availability (ASR-deny) without any of the integrity-degradation surface visible to a defender. The Context-Relevance panel adds one further finding: `qInj`'s query-rewrite leaves the retrieved documents tangentially relevant at best (0.58 vs clean 0.85) — the attacker shifted the conversation away from the user's original intent, not just the answer.

**This decoupling is the Chapter 7 hook.** RAGAS is a useful diagnostic *of generator behaviour given the context provided*, not a sufficient detector of *adversarial content in the context*. A defence based on RAGAS Faithfulness alone would miss `poiA`-style attacks entirely.

## 6.7 Retrieval impact: rank-shift@5 distributions

`rank_shift@5` is the change in rank of the originally top-1 clean document under the attack. F6 reports the empirical CDF per cell, which preserves the tail in a way the per-cell means and stacked-bar charts do not.

![Figure F6 — rank_shift@5 distribution per cell](../results/figures/rank_shift_ecdf.pdf)

**Reading F6.** `ipi` and `poiJ`/`poiA` have tight rank-shift distributions: >88% of `poiA` runs and >70% of `ipi` runs leave the original top-1 unmoved (CDF reaches 1.0 at shift = 1). `qInj` has a markedly heavier tail extending to shift = 5: rewriting the query causes much larger swings in retrieval order than the corpus-side attacks. Per `summary_by_cell.csv`: `mean_rank_shift_at_k` is 0.30 (`ipi`), 0.88 (`poiA`), 0.74 (`poiJ`), and 1.82 (`qInj`). The post-fix `poiJ` rank-shift is slightly *smaller* than the pre-fix 0.92 — the early-exit fix terminates 46% of poiJ runs at iter 0 (before any iter 1+ payload rewrite touches the index again), so the rank-shift mean is now closer to a clean single-iteration measurement.

## 6.8 Planner adaptation: does ε-greedy learn?

F7 is a two-panel view of the planner sidecar log. Left panel: running success rate per attack family vs the query-selection index (1..50), averaged across the 3 seeds with shaded 95% bootstrap CIs. Right panel: arm-pull histogram — total selections per family per seed.

![Figure F7 — Planner adaptation](../results/figures/planner_adaptation.pdf)

**Reading F7.** Within ~30 selections, the planner has identified prompt-injection and query-injection as the high-yield families (running ASR-t ~ 0.95). Corpus-poisoning settles around 0.55 — reflecting that the family-default mapped to in the sidecar is `answer_replacement`, which fails on ~20% of trials. The arm-pull histogram confirms the exploitation step is dominant: prompt-injection was selected 40 times in seed 42, vs ~4 selections of corpus-poisoning. The ε = 0.3 exploration rate ensures every family is still pulled enough that the success-rate estimates remain unbiased.

This answers RQ2 affirmatively: the ε-greedy planner does converge on the higher-success arms within the 50-query horizon, and the convergence is reproducible across the three seeds.

## 6.9 Paired comparisons vs IPI

`paired_differences_vs_ipi.csv` (paired by seed × query_id, n=150 pairs each):

| Comparison | Cell rate | IPI rate | Difference | 95% CI | Cohen's h |
| --- | --- | --- | --- | --- | --- |
| `poiA - ipi` | 0.800 | 0.960 | **-0.160** | [-0.247, -0.087] | -0.525 (moderate) |
| `poiJ - ipi` | 0.460 | 0.960 | **-0.500** | [-0.587, -0.420] | -1.248 (large)    |
| `qInj - ipi` | 0.960 | 0.960 |  0.000 | [-0.047, +0.040] |  0.000 (none)     |

**Reading the table.** Pairing within (seed, query_id) absorbs the per-query difficulty and per-seed RNG variance, producing tighter CIs than an unpaired between-cell comparison would. The three findings: query-injection is statistically indistinguishable from prompt-injection in end-to-end success, answer-replacement is a moderate effect-size weaker, and jamming under its own success metric (ASR-deny) is a large effect (h = -1.25) — substantially less effective than the integrity attacks but reliably non-zero, with a CI that comfortably excludes zero. The pre-fix comparison reported `poiJ` at -0.96 (h = -2.74, "huge"); that number was an artefact of the orchestration bug described in §6.5.1, not a property of the underlying attack.

## 6.10 Mapping to research questions

| RQ | Question                                                          | Primary evidence                                  |
| -- | ------------------------------------------------------------------ | -------------------------------------------------- |
| RQ1 | Do attacks succeed end-to-end against a black-box RAG?            | F1, F2, `summary_by_cell.csv`                     |
| RQ2 | Does the ε-greedy planner adapt to attack difficulty?             | F7, planner sidecar logs                          |
| RQ3 | Where in the pipeline does each attack succeed or fail?           | F1 (ASR-r vs ASR-a gap), F3 (k-curve), F6 (rank-shift) |
| RQ4 | Does attack success correlate with measurable integrity loss?     | F5, `ragas_by_cell.csv`, `paired_differences_vs_ipi.csv` |

Chapter 7 (Discussion) interprets these findings against the comparable literature (PoisonedRAG, AgentPoison, BadRAG), and Chapter 8 (Conclusion) maps the Chapter-6 evidence onto the original research questions one final time.
