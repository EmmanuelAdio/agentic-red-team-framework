# Chapter 6 — Results

**Source files**
- Tables: `results/tables/{summary_by_cell,ragas_by_cell,paired_differences_vs_ipi,baseline_summary}.csv`
- Figures: `results/figures/{asr_triple_by_cell,channel_objective_heatmap,asr_r_vs_k,asr_deny_by_cell,ragas_triple_clean_vs_attacked,rank_shift_ecdf,planner_adaptation}.pdf` (PNG counterparts beside each PDF)
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
| `poiJ` | corpus × availability | **0.000** | [0.000, 0.000] |
| `qInj` | query × integrity     | **0.960** | [0.927, 0.987] |

![Figure F1 — ASR-r / ASR-a / ASR-t per attack cell](../results/figures/asr_triple_by_cell.pdf)

![Figure F2 — Headline attack success by channel × objective](../results/figures/channel_objective_heatmap.pdf)

**Reading F1.** ASR-r is 1.00 for every cell — every poisoned document made the top-5 retrieval, and `qInj`'s ASR-r is True by construction (the attack rewrites the query, it does not poison the corpus). The drop from ASR-r to ASR-a/ASR-t localises where each attack loses ground: `ipi` and `qInj` keep ~96% of their retrieval success through generation; `poiA` loses 20% of trials between retrieval and answer (the LLM caught the false-answer span on those queries); `poiJ` loses 28% (a structural finding revisited in §6.5).

**Reading F2.** The 2×2 heatmap restates the same result in the framework's own taxonomy. Both integrity cells are very effective (0.80 and 0.96); the corpus × availability cell is null (0.00). The empty query × availability quadrant is the part of the threat surface the framework deliberately does not implement — there is no query-channel attack in this scope that targets availability.

## 6.4 Dose-response: ASR-r vs retrieval depth k

PoisonedRAG (Zou et al., 2024) reports ASR vs retrieval depth k to expose the trade-off a defender faces between recall and exposure. F3 reproduces that view for our three corpus-channel cells; `qInj` is excluded because its ASR-r is independent of k by construction.

![Figure F3 — ASR-r vs retrieval depth k](../results/figures/asr_r_vs_k.pdf)

**Reading F3.** `poiA` and `poiJ` both reach ASR-r ≈ 0.85 at k = 1 — their exploit-generated payloads almost always claim the rank-1 slot, because the generator optimises payload surface-similarity to the query. `ipi` starts at 0.30 at k = 1 and ramps to 1.00 by k = 4: the injection payload is a short imperative string whose embedding similarity to the query is *lower* than the original gold passage, so the poisoned doc sits mid-rank. The implication for a defender is concrete: truncating retrieval at k = 1 cuts `ipi`'s ASR-r by two-thirds, but pays nothing against `poiA`/`poiJ` whose payloads survive aggressive truncation.

## 6.5 The availability cell: an honest negative result

The jamming cell's success metric is **ASR-deny**: did the LLM refuse to answer, or otherwise return a non-answer? F4 reports ASR-deny for every cell.

![Figure F4 — ASR-deny per cell](../results/figures/asr_deny_by_cell.pdf)

**Reading F4.** Every cell sits at exactly 0.00. For the integrity cells this is expected — they target ASR-t, not ASR-deny. For the jamming cell it is a substantive failure to the *availability* objective: the framework's jamming prompt does not push the LLM into refusal under the project's prompt template ("Answer the question using only the context below."). What jamming did achieve was an ASR-t of 0.72 [0.65, 0.79] (`summary_by_cell.csv:row=poiJ:asr_target_rate`) — the jamming payload often coerced the LLM into emitting the attacker's target span rather than refusing, collapsing jamming into a weaker form of answer-replacement.

This is reported as a negative result rather than hidden: see Chapter 7 §7.3 for the discussion of why a temperature-0 RAG instructed to "answer from context" is hard to coerce into refusal, and `FUTURE_WORKS.md` §3 for the deferred attack families (retrieval-flooding, context-overflow) the result motivates.

## 6.6 Integrity degradation under attack

`ragas_by_cell.csv` (mean ± 95% CI):

| Cell | Attacked Faithfulness | Faithfulness drop vs clean | Integrity degraded ≥ 0.2 |
| --- | --- | --- | --- |
| `ipi`  | 0.613 [0.546, 0.681] | **+0.262** [+0.168, +0.351] | 46.7% |
| `poiA` | 0.919 [0.883, 0.953] | -0.044 [-0.093, +0.010]     |  9.3% |
| `poiJ` | 0.902 [0.857, 0.941] | -0.027 [-0.086, +0.034]     | 10.7% |
| `qInj` | 0.282 [0.219, 0.348] | **+0.593** [+0.507, +0.673] | 73.3% |

![Figure F5 — RAGAS triple: clean baseline vs each attacked cell](../results/figures/ragas_triple_clean_vs_attacked.pdf)

**Reading F5.** The Faithfulness panel surfaces this chapter's most important methodological caveat: a high ASR-t **does not** imply a Faithfulness drop. `poiA` reaches 80% ASR-t while its mean Faithfulness *rises* slightly above the clean baseline (0.919 vs 0.875); the poisoned context literally contains a statement supporting the attacker's false answer, so the RAGAS Faithfulness scorer (which asks "is every claim in the answer supported by the context?") rates the answer as faithful even though it is factually wrong.

`ipi` and `qInj` show the expected pattern — large Faithfulness drops (0.262 and 0.593) and high integrity-degraded rates (47% and 73%). The Answer-Relevance panel separates the two: both `ipi` and `qInj` collapse AR to ~0.03 because the injected text is unrelated to the original query, while `poiA`/`poiJ` keep AR high (~0.78) because their answers are still topically relevant, just wrong. The Context-Relevance panel adds one further finding: `qInj`'s query-rewrite leaves the retrieved documents tangentially relevant at best (0.58 vs clean 0.85) — the attacker shifted the conversation away from the user's original intent, not just the answer.

**This decoupling is the Chapter 7 hook.** RAGAS is a useful diagnostic *of generator behaviour given the context provided*, not a sufficient detector of *adversarial content in the context*. A defence based on RAGAS Faithfulness alone would miss `poiA`-style attacks entirely.

## 6.7 Retrieval impact: rank-shift@5 distributions

`rank_shift@5` is the change in rank of the originally top-1 clean document under the attack. F6 reports the empirical CDF per cell, which preserves the tail in a way the per-cell means and stacked-bar charts do not.

![Figure F6 — rank_shift@5 distribution per cell](../results/figures/rank_shift_ecdf.pdf)

**Reading F6.** `ipi` and `poiJ`/`poiA` have tight rank-shift distributions: >88% of `poiA` runs and >70% of `ipi` runs leave the original top-1 unmoved (CDF reaches 1.0 at shift = 1). `qInj` has a markedly heavier tail extending to shift = 5: rewriting the query causes much larger swings in retrieval order than the corpus-side attacks. Per `summary_by_cell.csv`: `mean_rank_shift_at_k` is 0.30 (`ipi`), 0.88 (`poiA`), 0.92 (`poiJ`), and 1.82 (`qInj`).

## 6.8 Planner adaptation: does ε-greedy learn?

F7 is a two-panel view of the planner sidecar log. Left panel: running success rate per attack family vs the query-selection index (1..50), averaged across the 3 seeds with shaded 95% bootstrap CIs. Right panel: arm-pull histogram — total selections per family per seed.

![Figure F7 — Planner adaptation](../results/figures/planner_adaptation.pdf)

**Reading F7.** Within ~30 selections, the planner has identified prompt-injection and query-injection as the high-yield families (running ASR-t ~ 0.95). Corpus-poisoning settles around 0.55 — reflecting that the family-default mapped to in the sidecar is `answer_replacement`, which fails on ~20% of trials. The arm-pull histogram confirms the exploitation step is dominant: prompt-injection was selected 40 times in seed 42, vs ~4 selections of corpus-poisoning. The ε = 0.3 exploration rate ensures every family is still pulled enough that the success-rate estimates remain unbiased.

This answers RQ2 affirmatively: the ε-greedy planner does converge on the higher-success arms within the 50-query horizon, and the convergence is reproducible across the three seeds.

## 6.9 Paired comparisons vs IPI

`paired_differences_vs_ipi.csv` (paired by seed × query_id, n=150 pairs each):

| Comparison | Cell rate | IPI rate | Difference | 95% CI | Cohen's h |
| --- | --- | --- | --- | --- | --- |
| `poiA - ipi` | 0.800 | 0.960 | **-0.160** | [-0.247, -0.087] | -0.524 (moderate) |
| `poiJ - ipi` | 0.000 | 0.960 | **-0.960** | [-0.987, -0.927] | -2.739 (huge)     |
| `qInj - ipi` | 0.960 | 0.960 |  0.000 | [-0.047, +0.040] |  0.000 (none)     |

**Reading the table.** Pairing within (seed, query_id) absorbs the per-query difficulty and per-seed RNG variance, producing tighter CIs than an unpaired between-cell comparison would. The three findings: query-injection is statistically indistinguishable from prompt-injection in end-to-end success, answer-replacement is a moderate effect-size weaker, and jamming is qualitatively different (the CI doesn't cross zero and the effect size is huge).

## 6.10 Mapping to research questions

| RQ | Question                                                          | Primary evidence                                  |
| -- | ------------------------------------------------------------------ | -------------------------------------------------- |
| RQ1 | Do attacks succeed end-to-end against a black-box RAG?            | F1, F2, `summary_by_cell.csv`                     |
| RQ2 | Does the ε-greedy planner adapt to attack difficulty?             | F7, planner sidecar logs                          |
| RQ3 | Where in the pipeline does each attack succeed or fail?           | F1 (ASR-r vs ASR-a gap), F3 (k-curve), F6 (rank-shift) |
| RQ4 | Does attack success correlate with measurable integrity loss?     | F5, `ragas_by_cell.csv`, `paired_differences_vs_ipi.csv` |

Chapter 7 (Discussion) interprets these findings against the comparable literature (PoisonedRAG, AgentPoison, BadRAG), and Chapter 8 (Conclusion) maps the Chapter-6 evidence onto the original research questions one final time.
