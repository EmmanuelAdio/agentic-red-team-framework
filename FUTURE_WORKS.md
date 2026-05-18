# Future Work — Agentic Red-Team Framework

Ideas considered and consciously deferred from the May 2026 deliverable. Each
entry pairs the deferral with a *why-it-matters* hook so this file can seed
the dissertation's Future Work section (Chapter 8) and structure
supervisor / demo conversations about the framework's natural extension axes.

**Status flags** used below:

- *blocked-by-deadline* — feasible architecturally; cut for time.
- *blocked-by-scope-discipline* — explicitly excluded by `PROJECT_SPEC.md` §2 to keep the deliverable focused.
- *blocked-by-threat-model* — would require relaxing the project's black-box-with-corpus-write threat model.
- *legitimate-stretch* — small enough to consider for the Day-16 buffer if everything else lands clean.
- *implemented-mid-project* — originally deferred, pulled into scope during the Day-7 buffer when the project ran ahead of schedule. Entries kept here as historical record; see `LAB_NOTEBOOK.md` Day 7.5 for the rationale and the implementation pointer.

The framework's contribution is the *methodology* — the agentic
plan → generate → execute → evaluate loop, the ASR (Attack Success Rate)
triple decomposition, and the reproducible exploit-bundle JSON schema. The
current implementation targets a single owned RAG (Retrieval-Augmented
Generation) pipeline as a **reference instantiation** so that every variable
(corpus, embedder, retriever, LLM, prompt template) is pinned and every
result is reproducible. Extensions below trade reproducibility, scope, or
threat-model fidelity for some other axis of value (deployment realism,
attack diversity, evaluator richness, etc.).

---

## 1. Deployment surface — turning the framework into a service

### 1.1 User-supplied RAG endpoint *(blocked-by-threat-model + blocked-by-deadline)*

Replace the in-process `RAGPipeline` with an HTTP adapter so users point the
framework at their own deployed RAG and have it run the attack matrix
against that endpoint.

*Why it matters:* turns the framework from a closed-loop reference
implementation into an external red-team service — a natural
production-assurance use case.

*Trade-offs:* corpus-poisoning collapses without write access (most
endpoints don't expose `add_documents`); the `index_state_hash`
reproducibility property weakens because target state can't be pinned;
authorisation surface (CFAA / Computer Misuse Act) requires either a
self-hosted reference target or an explicit user-owns-the-RAG licence on
the framework.

*Architectural seam already exists:* the executor depends on the
`RAGPipeline.run(query) → dict` contract only, so swapping the
implementation behind that interface is the work — no orchestration /
metrics / bundle changes needed.

### 1.2 Web dashboard reading exploit bundles *(blocked-by-scope-discipline)*

Streamlit / Gradio app loading `data/runs/*.json` and rendering interactive
filterable tables, ASR distributions, and per-run drill-downs.

*Why it matters:* live demo surface for supervisors and examiners; lowers
the bar to reading exploit bundles.

*Note:* `PROJECT_SPEC.md` §2 explicitly excludes a Streamlit dashboard.
`notebooks/03_results_analysis.ipynb` is the planned deadline-scope
equivalent — a Jupyter notebook that loads bundles and renders charts /
tables interactively, serving the same audience for half the cost.

### 1.3 Continuous-integration mode *(blocked-by-deadline)*

Schedule attack runs against a target endpoint on a cron and emit alerts
when ASR-t crosses a threshold (e.g. via webhook to Slack / GitHub).

*Why it matters:* converts the framework from a one-shot research artefact
into a production-assurance tool — the natural commercialisation /
open-source-tooling angle.

### 1.4 Interactive experimentation from the dashboard *(blocked-by-deadline)*

The Streamlit dashboard shipped on Day 14 / 15 is *read-only* — it
surfaces bundles that already exist under `results/runs/` and
`data/runs/`. A natural extension is to make the dashboard a **launch
surface** for new experiment cells, so the reader can answer "what
happens if I rerun this query under attack configuration X?" without
dropping back to `scripts/06_run_experiments.py` and a fresh shell.

Concrete scope:

1. **Wire the "Re-run with different seed" button on Run Detail.** It
   currently shows a toast saying "Not implemented in Build A". Replace
   with a small form that takes a seed override, queues a single-run
   batch via `BundleStore`, polls for the new bundle, and routes the
   reader to its Run Detail page when it lands.
2. **"Re-run with different config" expansion.** Extend the form with
   knobs the bundle schema already records per-run:
   - `attack_channel` ∈ `{corpus, query}`
   - `payload_source` ∈ `{template, llm}`
   - `retriever_top_k`
   - `llm_temperature`

   so the reader can isolate the contribution of each knob without
   leaving the page. The natural quick test the channel × payload-source
   matrix gap calls for (see §2.8 below) — flip a working IPI run from
   `template` to `llm` and watch the marker drop out — would become a
   two-click interaction here.
3. **Home-page "Run experiment" panel.** A small form on the Overview
   page that submits a `(family, strategy, channel, payload_source,
   n_seeds)` tuple as a new cell, with a progress card that streams
   per-run completions until the batch finishes. Effectively the
   `06_run_experiments.py --quick` flow exposed as a browser form.

*Why it matters:* today the dashboard is descriptive; this turns it into
an interactive harness. Useful for live supervisor / examiner demos
("now let's see what happens if we flip the channel to `query`"). The
notebook (`notebooks/03_results_analysis.ipynb`) already covers the
analyst's flow; this would cover the *interactive interrogator's* flow.

*Why deferred:* needs a job-runner abstraction on top of the current
synchronous batch loop (so the Streamlit thread does not block on a
5-minute LLM call), and a result-watch loop to refresh the dashboard
when bundles land. Estimate 2–3 days of work — out of dissertation
scope, but a clean follow-up.

---

## 2. Attack-family extensions (within the current threat model)

### 2.1 Query-side / direct prompt injection (GGPP-style) *(implemented-mid-project)*

Attacker modifies the user's query before retrieval (rather than poisoning
documents the retriever pulls).

*Why it matters:* second axis of attack delivery; complements the
corpus-side IPI work. `PROJECT_SPEC.md` §2 originally listed this as
"only if Day 6 finishes early".

*Implementation pointer:* `src/redteam/attacks/query_injection.py`,
LangGraph third family branch, notebook section *Day 7.5 — Query-side
injection*. Pulled in during the Day-7 buffer so Chapter 6 can present
the full input-channel-vs-corpus-channel attack-surface taxonomy rather
than just the corpus channel.

### 2.2 Multi-document corpus poisoning (PoisonedRAG 5-doc setup) *(implemented-mid-project)*

Insert *N* poisoned documents per query rather than one, drowning the gold
document in adversarial context.

*Why it matters:* the Day-4 lab note recorded that single-doc poisoning
hit ASR-r = 1.0 but ASR-a = 0.0 against gold-co-retrieval. The original
PoisonedRAG paper hits 97% ASR with 5 docs. This is the **single most
natural counterfactual** to the Day-4 cross-family asymmetry finding and
the most defensible follow-up experiment.

*Implementation pointer:* `generate_poison_payloads(n_docs=N)` in
`src/redteam/attacks/corpus_poisoning.py`, notebook section *Day 7.5 —
Multi-doc poisoning sweep* (N ∈ {1, 3, 5, 7}). Pulled in during the
Day-7 buffer because it converts the Day-4 negative finding into a
measured threshold — the strongest single Chapter 6 result available
for the time cost.

### 2.3 Jamming / blocker documents (availability attack) *(implemented-mid-project)*

Poisoned document designed to make the LLM refuse to answer or stall —
an availability attack rather than an integrity attack. Distinct success
criterion: ASR-deny (output matches a refusal pattern), not ASR-a
(output contains a target string).

*Why it matters:* demonstrates that the framework handles multiple
attack *objectives*, not only substring-match integrity. Strengthens the
C1 contribution (multi-dimensional diagnosis) by widening it from a
single integrity axis to integrity + availability.

*Implementation pointer:* `jamming` strategy in
`src/redteam/attacks/corpus_poisoning.py`, `compute_asr_deny` in
`src/redteam/metrics/asr.py`, notebook section *Day 7.5 — Jamming /
blocker documents*.

### 2.4 BadRAG-style trigger-conditioned poisoning *(blocked-by-deadline)*

Poisoned document only fires when the query contains a specific trigger
substring; otherwise the document remains dormant in the corpus.

*Why it matters:* much harder to detect via corpus spot-checks because the
poisoning is invisible until the trigger query arrives. Strengthens the
realism of the threat model.

*Specific evaluation caveat:* meaningful evaluation requires a query set
that **contains the trigger token in some queries**. Day 2's 50 NQ
queries are sampled from BeIR/NQ and don't naturally carry an attacker's
chosen trigger string. Two follow-up paths: (a) augment the query set
with a hand-authored trigger-bearing slice (artificial but cheap), or
(b) train the trigger to a frequent NQ token and report under that
constraint. Either path is a defensible Future Work paragraph rather
than a Day-9-matrix extension.

### 2.5 CorruptRAG-style stealth poisoning *(blocked-by-deadline)*

Poisoned document mimics the genuine corpus's stylistic register so manual
human review can't distinguish it from real entries.

*Why it matters:* defeats human-in-the-loop corpus auditing — the most
common defensive control today.

### 2.6 Explicit inter-context conflicts *(blocked-by-deadline)*

Stage two factually contradictory documents in the corpus (e.g. a true
NQ passage plus a fabricated counter-passage that asserts the opposite
claim) and observe how the LLM reconciles them — without an explicit
override instruction.

*Why it matters:* the current `instruction_override` IPI strategy
already produces an inter-context conflict implicitly (legitimate gold
doc + override-doc both retrieved, LLM resolves toward the override).
The *explicit-conflict* version isolates the **factual-conflict
resolution behaviour** from the *instruction-following* behaviour —
useful for separating two confounded mechanisms in Chapter 6's
discussion. Largely a relabel of single-doc poisoning with a different
framing of the success criterion (does the LLM cite the false doc, the
true doc, or hedge?), so cheap to add but moderate value.

### 2.7 Multi-turn conversational attacks *(blocked-by-scope-discipline)*

Build the attack across multiple turns of a conversation rather than
single-turn QA.

*Why it matters:* most production RAG deployments are stateful chatbots,
not single-turn QA endpoints — current scope under-represents the
operational target surface.

### 2.8 Balanced `(channel × payload_source)` experiment matrix *(blocked-by-deadline)*

The Day-9 matrix exercises four cells (`ipi`, `poiA`, `poiJ`, `qInj`),
but within each cell the `(attack_channel, payload_source)` split is
unbalanced. Measured across the 600-bundle run:

| family              | strategy             | channel | template | llm |  ASR-t (tpl) | ASR-t (llm) |
| ------------------- | -------------------- | ------- | -------: | --: | -----------: | ----------: |
| prompt_injection    | instruction_override | corpus  |      144 |   6 |        100 % |         0 % |
| query_injection     | prefix_injection     | query   |      141 |   9 |        100 % |        33 % |
| corpus_poisoning    | answer_replacement   | corpus  |        3 | 147 |        100 % |        80 % |
| corpus_poisoning    | jamming              | corpus  |        0 | 150 |            — |         0 % |

This is partly by design (the template carries the literal marker
substring that ASR-a substring-matches against — an LLM-rephrased
payload usually drops or paraphrases the marker, and the run records
ASR-t = 0 even when the attack would be considered successful by a
human judge), and partly an oversight (jamming has no template arm at
all; corpus poisoning has only n = 3 template runs).

A clean follow-up is to **promote `payload_source` to a first-class
condition** in the manifest and run the full Cartesian product
`{family × strategy × channel × payload_source}` with a balanced *n*
per cell (e.g. 50 queries × 3 seeds for every payload-source).

The current metric definition would need a small adjustment in lockstep
with this: the LLM-rephrased arm should be scored by an LLM-judge
ASR-a (§5.2) rather than the literal-marker substring, otherwise the
comparison is mostly measuring *marker preservation* rather than
*attack strength* — which is what the unbalanced ASR-t = 0 % for
LLM-source IPI in the table above is really telling us.

*Why it matters:* the current results carry a template-vs-LLM confound
the dissertation has to disclose as a limitation; a balanced matrix
would let the next iteration of this work make a clean claim about
which delivery channel actually works best.

*Why deferred:* doubles the matrix size, depends on §5.2 (LLM-judge
ASR-a), and only marginally improves the headline story for the
current deadline.

---

## 3. Attack-family extensions (broader threat model)

### 3.1 White-box attacks (GASLITE, Joint-GCG) *(blocked-by-threat-model)*

Adversary reads embedding-model weights and crafts gradient-optimised
adversarial documents.

*Why it matters:* strictly stronger threat model; meaningful upper bound
on attack success when the adversary has model access. `PROJECT_SPEC.md`
§2 explicitly defers.

### 3.2 Multimodal RAG (image / PDF / table retrieval) *(blocked-by-scope-discipline)*

Extend the target system from text-only retrieval to multimodal indexing.

*Why it matters:* production RAG systems increasingly index non-text
content (PDFs, screenshots, tabular data); the attack surface expands
correspondingly.

### 3.3 Canary leakage / data-exfiltration attacks *(blocked-by-threat-model)*

Plant canary tokens (unique synthetic strings, e.g. `CANARY_7f3a9c12`)
into the retrieval corpus, then craft attacker queries / documents that
attempt to coax the LLM into emitting the canary. ASR variant:
`ASR-leak` (substring-match the canary in the generator output).

*Why it matters:* tests **confidentiality** (data exfiltration) rather
than integrity, opening a third orthogonal axis alongside the current
integrity (false-answer) and availability (jamming) attack objectives.
Particularly compelling in the *user-supplied RAG endpoint* extension
(§1.1) where end-users index private documents — canary leakage is the
natural threat-model framing for a red-team-as-a-service product.

*Why deferred:* the current spec §3 threat model is integrity-focused
(corpus-write attacker aiming to corrupt outputs). Canary leakage
reframes the adversary as a confidentiality-breach attacker
(exfiltration of data the corpus owner intended to keep private). That
reframing is large enough to deserve its own threat-model section and
its own RQ — not a Day-9-matrix extension, but a strong centrepiece for
the deployment-surface Future Work in §1.

---

## 4. Target-system extensions

### 4.1 Second retriever (BM25 / sparse) *(legitimate-stretch)*

Add a sparse retriever alongside the dense bge-small one and re-run the
attack matrix on both.

*Why it matters:* tests whether the Day-4 cross-family asymmetry
generalises across retrieval geometries. `PROJECT_SPEC.md` §2 lists this
as conditional ("only if Days 8–9 land cleanly").

### 4.2 Custom corpus beyond NQ *(blocked-by-scope-discipline)*

Replace BeIR/NQ with a domain-specific corpus (medical, legal, code).

*Why it matters:* validates external generalisation of findings. Current
scope intentionally pins NQ for reproducibility — a known, citable corpus.

### 4.3 Alternative LLM backends as target *(blocked-by-deadline)*

Re-run the matrix against Ollama `llama3.1:8b`, Mistral, Claude, etc.

*Why it matters:* the Day-4 IPI-compliance result (`PWNED_BY_REDTEAM`
emitted by both override strategies) is gpt-4o-mini-specific. Cross-model
replication is the natural follow-up to ground the finding in something
broader than one model's safety training.

---

## 5. Evaluator extensions

### 5.1 TruLens integration alongside RAGAS *(blocked-by-scope-discipline)*

Add TruLens feedback functions as a second integrity-metric stack.

*Why it matters:* RAGAS and TruLens cover overlapping but non-identical
ground; running both gives a richer integrity-degradation signal.
`PROJECT_SPEC.md` §2 lists this as conditional ("only if RAGAS integration
finishes by Day 6").

### 5.2 LLM-judge for ASR-a (beyond substring match) *(blocked-by-deadline)*

Replace whitespace-normalised substring matching with an LLM-judge that
scores semantic equivalence between the generator output and the
attacker's target string.

*Why it matters:* substring matching misses paraphrased or semantically
equivalent compliance (the LLM might emit "I have been compromised" when
the marker is `PWNED_BY_REDTEAM`). LLM-judge captures these at the cost
of non-determinism — well-trodden trade-off in the LLM-eval literature.

### 5.3 Defence implementation + evaluation *(blocked-by-scope-discipline)*

Ship a defence (e.g. spotlighting, retrieval-time poison detection,
context sanitisation) and run it through the same exploit-bundle pipeline
to measure ASR drop.

*Why it matters:* `PROJECT_SPEC.md` §2 explicitly defers — current scope
evaluates *attacks*, not defences. A natural symmetric extension and the
most defensible angle for follow-up academic work.

---

## 6. Methodology and scale

### 6.1 Larger query set *(blocked-by-deadline)*

Move from 50 queries to the full BeIR / NQ test split (~3,500 queries),
yielding a ~21,000-run matrix at 3 seeds × 2 attacks.

*Why it matters:* current 50-query matrix gives feasibility-grade
estimates, not population-grade statistics. Full-scale demonstrates the
framework's throughput claims.

### 6.2 More than 3 seeds for bootstrap CIs *(blocked-by-deadline)*

Run 5–10 seeds per (query, attack) cell to tighten bootstrap 95%
confidence intervals.

*Why it matters:* the Day-9 3-seed matrix produces deliberately
conservative CIs — wide enough to detect large effects, too wide for
small-effect estimation. Tightening matters when comparing nearby attack
strategies (e.g. `instruction_override` vs `role_reassignment`).

### 6.3 Cross-paper replication study using exploit bundles *(blocked-by-deadline)*

Pick a published RAG-attack paper (e.g. PoisonedRAG, EchoLeak), reproduce
its setup inside the framework, archive the resulting bundles.

*Why it matters:* this is what the bundle JSON schema (spec §7) is built
for — third parties auditing or extending published attack claims.
Demonstrates the framework's reproducibility contribution in an external
setting and is the strongest possible follow-up paper.

---

## How to use this register

- **In supervisor / demo conversations:** when asked "why didn't you do X?",
  point to the relevant entry's status flag. Each flag justifies the
  deferral as a deliberate choice rather than an oversight.
- **In Chapter 8 (Conclusion / Future Work):** lift the entries verbatim,
  grouping by category. The *why-it-matters* hooks are already
  dissertation-paragraph-shaped.
- **When a new idea surfaces during implementation:** add it here rather
  than to `PROJECT_SPEC.md` or `LAB_NOTEBOOK.md`. Keeps the deferral
  register centralised.
