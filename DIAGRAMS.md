# Diagrams + Design Rationale

Mermaid-format diagrams **and design-rationale prose** for the framework.
Each diagram has three parts:

1. *What the diagram shows* — a short caption.
2. *The diagram itself* — Mermaid syntax that renders natively in GitHub or
   VS Code (with the *Markdown Preview Mermaid Support* extension).
3. *Design rationale* — why this design, what alternatives were considered,
   what triggers each control-flow edge. These rationale blocks are written
   to lift verbatim into the dissertation's Chapter 3 (Design).

A standalone §6 ("Metrics rationale") covers the three metric families
wired in on Day 7, and a closing §7 ("Cross-cutting design choices")
covers the system-wide decisions that don't belong to any single
diagram — model picks, reproducibility primitives, scope discipline, etc.

Sections marked **(placeholder)** will be populated on the day named in
their heading.

---

## 1. System architecture

### What it shows

The framework has four logical layers: the **target RAG (Retrieval-Augmented
Generation) pipeline** (the system under test), the **attack family modules**
(Days 3–4), the **agent layer** (LangGraph plan→generate→execute→evaluate
loop, Days 5–6), and **metrics + bundles** (Days 7–8). Each layer feeds the
next: attacks produce payloads, agents orchestrate which attack runs when,
the executor applies attacks against the target, and the evaluator scores
the result and writes a reproducible exploit-bundle JSON.

```mermaid
flowchart LR
    subgraph TARGET["Target RAG pipeline (Days 1-2)"]
        direction TB
        Corpus[(Chroma index<br/>1k NQ docs)]
        Retriever[Retriever<br/>bge-small, top-k=5]
        Generator[Generator<br/>gpt-4o-mini, T=0]
        Corpus --> Retriever --> Generator
    end

    subgraph ATTACKS["Attack family modules (Days 3-4)"]
        direction TB
        IPI[prompt_injection<br/>instruction_override<br/>role_reassignment]
        Poison[corpus_poisoning<br/>answer_replacement]
    end

    subgraph AGENTS["Agent layer (Days 5-6)"]
        direction TB
        Planner[Planner<br/>epsilon-greedy]
        ExploitGen[Exploit generator<br/>LLM-driven, Day 6]
        Executor[Executor<br/>add/run/remove]
        Evaluator[Evaluator<br/>RAGAS + ASR + rank-shift]
    end

    subgraph OUTPUT["Metrics and bundles (Days 7-8)"]
        direction TB
        Metrics[ASR-r / ASR-a / ASR-t<br/>rank-shift@k<br/>RAGAS triad]
        Bundle[(Exploit bundle JSON<br/>data/runs/*.json)]
        Metrics --> Bundle
    end

    Planner --> ExploitGen
    ExploitGen --> IPI
    ExploitGen --> Poison
    IPI --> Executor
    Poison --> Executor
    Executor -->|insert payload| Corpus
    Executor -->|run query| Generator
    Generator --> Evaluator
    Retriever --> Evaluator
    Evaluator --> Metrics
    Evaluator -.->|feedback| Planner
```

### Design rationale

**Why four layers and not one monolith?** Each layer is independently
testable and substitutable. The Target RAG layer can be swapped behind an
HTTP adapter (see `FUTURE_WORKS.md` §1.1) without touching agents or
metrics. Attack-family modules are pure payload generators — no I/O, no
LLM calls — so unit-tested deterministically (Day 3/4 tests). The agent
layer holds the only stateful component (the planner's success memory) and
the only LLM-driven adaptation (the exploit generator). Metrics and
bundles are pure functions over the agent layer's output. This division
enables the **closed-system-as-deliberate-methodological-choice** framing
recorded in `FUTURE_WORKS.md`.

**Why these specific components per layer?** The *target* uses Chroma
(persistent local vector store, no infrastructure dependency) +
`bge-small-en-v1.5` (CPU-friendly, top of the MTEB leaderboard for its
size) + `gpt-4o-mini` (cheap, capable, deterministic at temperature 0).
The *attack family* split (IPI vs corpus poisoning) is the spec §2 frozen
scope and follows the literature's most-cited threat-class taxonomy
(Greshake et al. for IPI; PoisonedRAG / Zou et al. for corpus poisoning).
The *agent layer* mirrors spec §4 verbatim — the four-agent decomposition
is what makes the system "agentic" in the project's RQ2 sense (does
adaptation improve attack success?). The *metrics+bundles* layer is the
project's signature contribution: every run is reproducible because every
run produces a JSON bundle pinning model, embedding, prompt template,
seed, and corpus state hash.

**Alternatives considered + rejected.** A single end-to-end script per
experiment (no agent layer) would have shipped faster but breaks RQ2
entirely. A more granular per-strategy agent decomposition (one
`InstructionOverrideAgent`, one `RoleReassignmentAgent`, etc.) was
rejected as premature abstraction — the strategy is data, not a class.

---

## 2. Threat model — what the attacker can and cannot do

### What it shows

Black-box-with-corpus-write (spec §3). Arrows from the attacker only land
on what they can influence; the retriever weights, LLM weights, and
system prompt are *not* reachable.

```mermaid
flowchart LR
    Attacker((Attacker))

    subgraph CAN["GRANTED capabilities"]
        Corpus[Corpus<br/>read + write docs]
        Query[User query<br/>can modify pre-retrieval]
    end

    subgraph CANT["DENIED capabilities"]
        EmbWeights[Embedding model weights]
        LLMWeights[LLM weights]
        SysPrompt[System prompt]
        ReTrain[Retrain or fine-tune]
    end

    Attacker -->|insert poisoned docs| Corpus
    Attacker -->|read corpus| Corpus
    Attacker -->|craft IPI documents| Corpus
    Attacker -.->|cannot read| SysPrompt
    Attacker -.->|cannot inspect| EmbWeights
    Attacker -.->|cannot modify| LLMWeights
    Attacker -.->|cannot retrain| ReTrain
```

### Design rationale

**Why this threat model and not something stronger?**
Black-box-with-corpus-write is the production-realistic threat profile —
it's what PoisonedRAG, BadRAG, and the EchoLeak (CVE-2025-32711) production
incident all assume. White-box variants (GASLITE, Joint-GCG) assume the
attacker can read embedding weights, which is a strictly stronger
adversary model and not what real deployed RAG systems face. The choice
also has a methodological pay-off: corpus-write capability is what makes
both attack families tractable on the *same* infrastructure (insert via
`Retriever.add_documents`), keeping the experiment matrix uniform.

**Why is "modify queries pre-retrieval" granted in the threat model?**
Real attackers in EchoLeak-style scenarios (compromised middleware,
prompt-laundering proxy, malicious copy-paste source) *do* control the
inbound message. As of Day 7.5 the framework also **implements** this
capability — see *§2.1 Attack channels* below.

**Alternatives considered + rejected.** A *fully* black-box threat model
(no corpus write) was rejected because it collapses corpus-poisoning
entirely. A *grey-box* model (attacker sees retrieval scores but not
embeddings) was rejected as adding complexity without research-question
payoff. White-box variants (GASLITE, Joint-GCG) — embedding-weight access
and gradient-optimised payloads — are stricter than the production-realistic
threat profile and stay deferred (`FUTURE_WORKS.md` §3.1).

### 2.1 Attack channels (Day-7.5 expansion)

The threat model splits the attacker's reach into two **delivery
channels**, both granted, both implemented:

```mermaid
flowchart LR
    Attacker((Attacker))

    subgraph CORPUS_CH["CORPUS channel"]
        IPI[Indirect prompt injection<br/>add override doc to corpus]
        Poison[Corpus poisoning<br/>add false-fact doc to corpus]
        MultiPoison[Multi-doc poisoning<br/>N docs to flood top-k]
    end

    subgraph QUERY_CH["QUERY channel"]
        QPrefix[Query prefix injection<br/>preamble before user question]
        QSuffix[Query suffix injection<br/>addendum after user question]
    end

    Attacker -->|writes to| IPI
    Attacker -->|writes to| Poison
    Attacker -->|writes to| MultiPoison
    Attacker -->|rewrites| QPrefix
    Attacker -->|rewrites| QSuffix

    IPI -->|reaches LLM via retrieval| LLM[LLM prompt]
    Poison -->|reaches LLM via retrieval| LLM
    MultiPoison -->|reaches LLM via retrieval| LLM
    QPrefix -->|reaches LLM via input| LLM
    QSuffix -->|reaches LLM via input| LLM
```

**Why two channels matter for the dissertation.** The two channels
exercise *different* LLM defences:

- **Corpus-channel** attacks (IPI + poisoning) succeed by *retrieval
  arithmetic*: getting the payload into top-k. Failure modes are
  retrieval-side (gold doc co-retrieves, payload deduplicated) more than
  LLM-side. The Day-4 negative finding (single-doc poisoning at ASR-a ≈
  0) and the Day-7.5 multi-doc threshold sweep both live in this regime.
- **Query-channel** attacks succeed by *prompt persuasion*: getting the
  LLM to follow the injection over the retrieved context. Failure modes
  are LLM-side (instruction-following defences, system-prompt priority).

Chapter 6 reports the cross-channel split: ASR-r is trivially `True` for
the query channel (no retrieval gating to bypass), so the channel
comparison is purely an ASR-a comparison. This is the cleanest split
available between *retrieval-side* and *LLM-side* attack mechanisms
within the project's black-box-with-write threat model.

**Implementation note.** Both channels go through the same
`plan → generate → execute → evaluate` LangGraph nodes; only the
executor branches on `state["attack_channel"]`. Adding a third channel
in future (e.g. system-prompt injection if/when granted) would mean one
new `_FAMILY_CHANNEL` entry, one new executor branch, and one new
evaluator special-case for ASR-r — the rest of the orchestration is
channel-agnostic by design.

---

## 3. Attack-flow sequence (shared by IPI and corpus poisoning)

### What it shows

Both attack families share the same delivery pattern: insert a payload
document via `Retriever.add_documents`, run the target query through the
pipeline, compute the ASR (Attack Success Rate) triple from the result,
and remove the payload via `Retriever.remove_documents` so the index
returns to its pre-attack state. The `try/finally` in the executor
guarantees the remove step runs even if the pipeline call raises.

```mermaid
sequenceDiagram
    autonumber
    participant Att as Attacker / Executor
    participant Ret as Retriever (Chroma)
    participant Gen as Generator (gpt-4o-mini)
    participant Eval as Evaluator

    Note over Att,Eval: pre-attack: PRE_HASH = retriever.get_state_hash()

    Att->>Ret: add_documents([payload])
    Note right of Att: payload is an<br/>IPIPayload or<br/>PoisonPayload
    Att->>Ret: query(query_text, k=5)
    Ret-->>Att: top-5 retrieved docs<br/>(payload typically appears)
    Att->>Gen: generate(query, retrieved_docs)
    Gen-->>Att: generator_output
    Att->>Eval: compute ASR triple
    Note right of Eval: ASR-r: payload in top-k?<br/>ASR-a: marker in answer?<br/>ASR-t: both
    Eval-->>Att: verdict
    Att->>Ret: remove_documents([payload.doc_id])

    Note over Att,Eval: post-attack: POST_HASH == PRE_HASH (assertion)
```

### Design rationale

**Why a shared add/run/remove sequence for both attack families?** IPI
and corpus poisoning differ in *payload character* (override instructions
vs false-fact assertions), not in *delivery mechanism*. Sharing the
delivery sequence means: (a) the executor has one path to test, not two;
(b) the topical-anchor retrieval-rank trick generalises across families
(verified in Days 3 + 4 round-trip tests); (c) the cross-family
comparison in Chapter 6 isolates payload character as the operative
variable, exactly what the AgentPoison-style ASR decomposition was
designed to surface (cross-family asymmetry is the headline Day-4
finding).

**Why `try/finally` cleanup specifically?** The Day 9 ~300-run experiment
matrix runs unattended. A pipeline crash mid-run that leaks a payload
into the index would corrupt every subsequent run's `index_state_hash`
and break reproducibility. `try/finally` guarantees `remove_documents`
fires regardless of whether the generator call raises, which is the
*local* rollback guarantee. Combined with the LangGraph conditional-edge
structure that never re-enters `execute` without a paired `add/remove`,
this gives the *global* rollback guarantee proved in
`test_graph_runs_one_iteration_round_trip`.

**Alternatives considered + rejected.** Persistent payload insertion
(skip the remove step, rely on a separate teardown phase) would have
shaved a few milliseconds per run but broken cross-run independence — a
cardinal sin in a 3-seed experiment design. A separate "poisoned"
collection (insert into a copy, not the live index) was rejected as
infrastructurally expensive (re-indexing a 1k-doc collection per run is
~10 seconds; insert+remove is sub-millisecond).

---

## 4. LangGraph workflow

### What it shows

The agentic loop is a 4-node LangGraph: `plan → generate → execute →
evaluate`, with one conditional edge back to `plan` (or to `END`) at the
bottom of the loop. The Day-6 evolution adds two adaptive elements: the
planner is ε-greedy with global success-rate memory, and the exploit
generator gates between a hand-templated path (iteration 0) and an LLM
path (iteration ≥ 1).

```mermaid
flowchart TD
    Start((Start)) --> Plan
    Plan["plan<br/>ε-greedy planner.select<br/>(family, strategy)<br/>payload_source: template iter=0, llm iter≥1"] --> Generate
    Generate{"generate<br/>dispatch on payload_source"}
    Generate -- "template" --> Tpl[hand-templated<br/>generate_ipi_payload<br/>generate_poison_payload]
    Generate -- "llm" --> LLM[LLMExploitGenerator<br/>generate_ipi / generate_poison<br/>conditioned on history]
    Tpl --> Execute
    LLM --> Execute
    Execute["execute<br/>add → run → remove<br/>try/finally cleanup"] --> Evaluate
    Evaluate["evaluate<br/>ASR-r / ASR-a / ASR-t<br/>planner.update(asr_t)<br/>verdict + history append"] --> Decide{should_continue?}
    Decide -- "verdict==success<br/>or iter≥max" --> End((End))
    Decide -- "else" --> Plan
```

### Design rationale

**Why this trigger logic for `payload_source` (template iter 0, LLM iter ≥ 1)?**
The hand-templated path is cheap, deterministic, and proven (Day-3 IPI
hits ASR-t = 1.0 on the demo query). Spending an LLM call on iteration 0
when the templates already work would be wasteful and mask the question
the multi-iteration design is built to answer: *can the agent adapt when
templates fail?* By gating the LLM behind a prior failure, the cost
profile aligns with the pedagogy: pay only when you must. The Day-4
poisoning result (template ASR-a = 0 against gold-co-retrieval) is the
canonical case where iteration 1's LLM-driven variant earns its cost — it
picks a query-specific plausible false answer that the template path
could not.

**Why `should_continue` exits early on `success`?** Continuing iterations
after a confirmed end-to-end exploit (ASR-t = true) wastes API budget and
adds nothing to the evidence base. The evaluator already wrote the
success verdict into `state["history"]`; the bundle JSON captures it.
Looping further only matters when the goal is to gather adaptation
evidence, which is what `failure` and `partial` verdicts trigger.

**Why ε-greedy with ε=0.3 and not Thompson sampling, UCB, or pure
greedy?** ε-greedy is the simplest defensible adaptive policy and was
named in spec §4.2; ε=0.3 is the spec's value (the literature default for
small action spaces). Thompson sampling and UCB would add ~50 lines of
code and require more samples to converge — wasted effort given the
50-query × 2-family × 3-seed matrix gives only ~75 (query, family) cells
total. Pure greedy (ε=0) was rejected because the planner needs at least
some exploration to escape an early miss — if iteration 0 fails for IPI
on the first query, ε=0 would lock IPI's success rate at 0 and never
retry it. Sample efficiency is logged as a refinement axis in
`FUTURE_WORKS.md` §6.

**Why a single global memory and not per-query-type buckets?** Per spec
§4.2 the planner should keep memory "per query type". With 50 queries
and ~7 plausible buckets (`who/when/where/what/why/how/other`), each
bucket would carry ~7 samples — too thin for ε-greedy to converge
meaningfully. A single global memory pools the signal at the cost of
fidelity. The per-bucket variant is logged in `FUTURE_WORKS.md` §6 as a
refinement once larger query sets are run.

**Why does `plan` come before `generate` rather than fusing into one
node?** Two reasons. First, separation of concerns: `plan` makes a
*decision* about which family to attempt; `generate` performs the
*construction* of the payload. Different node responsibilities, different
test surfaces, different cost profiles. Second, the bundle JSON (Day 8)
records the planner's choice and the exploit's content separately —
splitting them at node boundary keeps state-update merges clean.

**Alternatives considered + rejected.** A *while-loop* outside LangGraph
(plain Python state machine) would have side-stepped the LangGraph
learning curve named in spec §10 risks but lost the conditional-edge
predicate's clarity. A single *adapt* node (planner+generator fused)
was rejected for the separation-of-concerns reason. A *parallel-fanout*
graph (run both attack families simultaneously, pick the winner) was
rejected because it doubles API spend per query for a marginal
adaptation signal.

---

## 5. Exploit-bundle structure & on-disk layout

### 5.1 What this section is

Describes how every red-team run lands on disk: the per-bundle JSON
shape (spec §7 with three additive blocks), the batch-folder layout
under `data/runs/`, and the design rationale for both. This is the
operational definition of Contribution **C4** (reproducible exploit
bundles); the bundle layer is the seam between the live LangGraph
state and any downstream analysis or archival.

### 5.2 On-disk layout

Every invocation of the experiment driver (or the dry-run script,
`scripts/05_run_dryrun.py`) creates **one batch folder** and writes
every bundle from that invocation into it, plus a single batch-level
summary alongside.

```
data/runs/
├── batch_<batch_id>/                                   # one folder per invocation
│   ├── run_<query_id>_<batch_id>_bundle.json           # one per query
│   ├── run_<query_id>_<batch_id>_bundle.json
│   ├── ...
│   └── batch_<batch_id>_summary.json                   # rollup for this batch
├── batch_<batch_id>/
│   └── ...
```

`batch_<batch_id>` is a UTC timestamp (e.g. `batch_20260508T152200Z`).
The same `batch_id` is the trailing token in every bundle filename
inside, so a file is self-identifying — copying one out of its folder
is unambiguous about which batch produced it.

**Why batch-folder grouping (and not flat, and not per-run-folder):**

- *Flat* (`data/runs/<run_id>.json` for every run) makes Day 9's ~450
  bundles unmanageable in a single directory; a single `ls` becomes
  noise, and there is no natural place for batch-level metadata
  (rollback hashes, planner snapshot, wall time) to live without
  polluting the top level.
- *Per-run folders* (`data/runs/<run_id>/bundle.json`) over-fragments:
  there is rarely any per-run artefact other than the bundle itself,
  so each folder ends up containing one file plus directory overhead.
  More importantly the unit users actually reason about is "the batch
  I ran with seed 42" — so the directory split should match that unit.
- *Batch folders* group runs by the invocation that produced them.
  Every artefact for one batch — bundles plus summary — is co-located,
  which keeps the unit easy to share, archive, gzip, or delete.

**Why `batch_<id>_summary.json` lives inside the batch folder:** the
summary is a derived rollup of *its* bundles (verdict counts, ASR
totals, family distribution, planner snapshot, pre/post
`index_state_hash`). Putting it next to the bundles it describes keeps
a batch self-contained — no cross-folder lookups when you tar one up.

### 5.3 Bundle JSON shape (spec §7 with Day 6 / 7.5 / 8 extensions)

Each bundle is a single Pydantic-validated JSON document. The top-level
key order is fixed so the most-scanned blocks are at the top of the
file:

```
bundle_version          # "1.0"  — the lever for any future breaking change
summary                 # headline metrics (Day 8 addition; see §5.4)
run_id                  # globally unique within the project
timestamp_utc           # ISO-8601 with `Z` suffix
seed                    # planner + RNG seed for this run
framework_version       # the project's own version string

target_system           # what the RAG pipeline was running
  ├── embedding_model
  ├── vector_store
  ├── retriever_top_k
  ├── llm_model
  ├── llm_temperature
  └── prompt_template_hash       # SHA-256 of the verbatim prompt template

attack                  # what the planner + exploit-generator produced
  ├── family            # prompt_injection | corpus_poisoning | query_injection
  ├── strategy          # e.g. instruction_override, answer_replacement
  ├── payload           # the adversarial doc body or modified query
  ├── payload_id
  ├── injection_stage   # indexing | query        (mapped from attack_channel)
  ├── iteration
  ├── payload_source    # template | llm          (Day 6 — provenance flag)
  ├── attack_channel    # corpus | query          (Day 7.5 — channel split)
  ├── modified_query    # populated only for query-channel attacks
  └── exploit_prompt_template_hash  # SHA-256 of the LLM exploit-gen prompt
                                    # (`null` for the template path)

execution               # what actually happened on this run
  ├── query
  ├── query_id
  ├── index_state_hash  # SHA-256 of the index doc_id set at run time
  ├── retrieved_docs    # [{doc_id, rank, score, is_poisoned}] — content stripped
  ├── generator_output
  ├── generator_latency_ms
  └── baseline_top1_doc_id    # rank-1 from the *clean* baseline pass (for rs@k)

evaluation              # the reference-free metric vector (spec §6)
  ├── ragas_faithfulness         # ∈ [0, 1] | null
  ├── ragas_answer_relevance     # ∈ [0, 1] | null
  ├── ragas_context_relevance    # ∈ [0, 1] | null
  ├── asr_retrieval              # bool
  ├── asr_answer                 # bool
  ├── asr_target                 # bool   (asr_retrieval ∧ asr_answer)
  ├── asr_deny                   # bool   (Day 7.5; wired Day 8 — see §5.5)
  ├── rank_shift_at_k            # int    (sentinel = k if baseline doc fell out)
  ├── verdict                    # success | partial | failure
  ├── evaluator_notes            # RAGAS NaN / failure reasons
  └── iteration_history          # one entry per past iteration of the loop

reproducibility         # environment metadata for replay
  ├── git_commit                 # short SHA, or null if not in a repo
  ├── python_version
  └── key_dependencies           # name -> version for the pinned subset
```

### 5.4 Why a `summary` block at the top of every bundle

The Day-8 schema adds a `summary` block immediately after
`bundle_version`. Its fields duplicate values that also appear in
`attack` / `execution` / `evaluation` (verdict, ASR triple + ASR-deny,
rank-shift, family / strategy / channel, query_id, generator latency,
RAGAS triple). The duplication is deliberate:

1. **At-a-glance scanning.** A reader opening any bundle file sees the
   verdict and the metric vector before any of the verbose blocks.
   With ~50–500 bundles per batch, this is the difference between
   instant triage and parsing every file to grep one field.
2. **Diff-friendliness.** Outcome changes between two runs (success →
   failure, ASR-t flip, RAGAS drop) surface at the top of the diff
   rather than hiding deep inside the `evaluation` block.
3. **Cannot drift.** The summary is constructed *once* in
   `build_bundle` from the same fields the detail blocks read. There
   is no second authorship path, so the summary cannot fall out of
   sync with the canonical view. A schema-test
   (`test_builder_writes_top_level_summary_consistent_with_blocks`)
   pins this contract so future refactors cannot break it silently.

The trade-off is a few hundred extra bytes per bundle (typical bundles
are ~3–5 KiB; the summary adds ~250 bytes). Cheap relative to the
analysis-time saving.

### 5.5 ASR-deny — wired into `evaluate_node` from Day 8

`compute_asr_deny` was introduced on Day 7.5 alongside the jamming
attack family (in `redteam.metrics.asr`) but was only invoked from the
notebook. The Day-8 wire-up calls it from `evaluate_node` for every
run — corpus and query channel alike — so each bundle's
`evaluation.asr_deny` is always populated as a bool.

**Why it ships now rather than being deferred to Day 9:** the dry-run
script writes 50 bundles and is the first time the bundle JSON is
exercised under realistic conditions. Leaving `asr_deny` at `null` for
those 50 bundles would pollute the historical record — Day-9's
analysis would then have to special-case "pre-Day-8 bundles" when
aggregating availability outcomes. Wiring it into the evaluator on
Day 8 means *every* recorded run carries the field uniformly.

The integrity ASR triple and ASR-deny are *orthogonal*: the integrity
triple measures attacks that hijack the answer; ASR-deny measures
attacks that block the answer. A jamming attack can register
`asr_deny=True` with `asr_target=False`, which is the exact case
Day 7.5's blocker-document family was added to study. The schema
keeps `asr_deny: Optional[bool]` so any pre-Day-8-wire-up bundles
still parse, but the builder always emits a bool for fresh runs.

### 5.6 Reproducibility primitives recorded per bundle

Three SHA-256 hashes plus a git commit pin replay-fidelity:

- `target_system.prompt_template_hash` — protects against silent
  edits to the RAG generator's prompt template.
- `attack.exploit_prompt_template_hash` — protects against silent
  edits to the LLM exploit-generator's prompt template (when the
  payload took the LLM path; `null` for the template path).
- `execution.index_state_hash` — pins the retriever's doc_id set at
  run time. Lets a reviewer assert the index was in the expected
  state when the attack ran.
- `reproducibility.git_commit` — short SHA of the project at run
  time; pairs with `key_dependencies` to recover the exact code
  version.

Together these mean: given a bundle, a third party can recompute the
target_system, replay the same payload against the same index state,
and check the generator's response against the recorded one — without
relying on any out-of-band metadata.

---

## 6. Metrics rationale

### What this section is

A prose-only design block (no diagram) covering the three metric
families wired in on Day 7: the ASR triple, `rank_shift@k`, and the
RAGAS triple. Each subsection explains *why this metric*, *why this
specific definition*, and *what alternatives were rejected*. Lifts
into Chapter 3 (Design) directly.

### 6.1 Why reference-free metrics?

The dissertation's Contribution C1 is a framework that scores attack
success **without an oracle answer key** — a property most published
RAG-attack benchmarks lack. Reference-required metrics (BLEU, ROUGE,
exact-match against a gold answer) force you to ship the framework with
a labelled corpus, which constrains generalisation and makes it
impossible to red-team a third-party system whose ground truth you
don't have. The three metric families chosen here are *all
reference-free*:

- ASR triple uses an attacker-supplied marker, not a gold answer.
- `rank_shift@k` uses the system's own pre-attack output as the
  reference, not a labelled top-1 doc.
- RAGAS uses LLM-judge consistency between query, retrieved context,
  and answer, with no ground-truth answer required.

This is what makes the framework portable to scenarios where the
attacker has corpus-write but no ground truth — exactly the
EchoLeak / PoisonedRAG production threat model.

### 6.2 ASR triple — why decompose?

Spec §6.1 (and AgentPoison [9]) decomposes attack success into three
components: ASR-r (did the payload reach the LLM?), ASR-a (did the LLM
emit the marker?), ASR-t (both — end-to-end success). The decomposition
isolates *where* an attack failed:

- ASR-r = 0 → retrieval-side failure (topical anchor didn't work, or
  embedding geometry rejected the payload).
- ASR-r = 1, ASR-a = 0 → generator-side failure (LLM saw the payload
  but did not comply / preferred surviving clean evidence). This is
  exactly the Day-4 cross-family asymmetry finding.
- ASR-t = 1 → end-to-end success.

Without the decomposition, both failure modes would aggregate into a
single "0% success" number that obscures the operative variable.

**Why substring matching for ASR-a (not LLM-judge)?** Substring is
deterministic, cheap, fast, and aligned with how the published RAG-attack
literature reports ASR ([6], [9]). LLM-judge would catch paraphrased /
semantically-equivalent compliance but adds non-determinism and another
LLM call per evaluation — logged in `FUTURE_WORKS.md` §5.2 as a
refinement.

### 6.3 `rank_shift@k` — why this definition?

Spec §6.3 defines rank-shift as "the change in rank position of the
originally top-1 clean document". The implementation:

1. Run a *baseline* (clean) retrieval pass before the attacked one.
2. Identify `baseline_top1_doc_id`.
3. Look up that doc-id in the attacked top-k. If present at rank `r`,
   `rank_shift = r - 1`. If absent, treat as if it landed at rank `k+1`
   (sentinel) so `rank_shift = k`.

**Why does the metric *only* track the baseline rank-1 doc?** It's the
single most-important retrieval result — the doc the system would have
cited if no attack had occurred. Tracking it directly answers "did the
attack push the original top answer aside?", which is the production
question. Tracking *all* baseline-top-k positions (a Kendall-tau-style
metric) was rejected as a generalisation that adds complexity for
marginal interpretability gain at this scope.

**Why cache the baseline per-query inside the executor?** The Day-9
~300-run matrix iterates each query 2 times under the LangGraph loop
(plus 3 seeds). Without caching, each query would do 6 baseline
pipeline runs — wasteful when the SQLiteCache already makes them
near-free. The closure-level dict cache is per-process; cross-process
caching is unnecessary because the clean prompt's SQLiteCache hit makes
it sub-millisecond anyway.

### 6.4 RAGAS triple — why these three, why this provider?

Spec §6.2 names Faithfulness, Answer Relevance, Context Relevance —
the three RAGAS metrics most often cited in the RAG-evaluation
literature and the ones with the cleanest definitions for an attacked
condition:

- **Faithfulness** ↓ under attack means the LLM is producing claims
  not supported by retrieved context — the *integrity-degradation*
  signal. Spec §6.2 names a ≥0.2 drop as "integrity-degraded".
- **Answer Relevance** ↓ under attack means the LLM is answering a
  *different question* than the one asked — the *off-topic-pivot*
  signal.
- **Context Relevance** ↓ under attack means the retrieval context is
  drifting from the query's information need — the *topical-pollution*
  signal.

**Why our `gpt-4o-mini` and not RAGAS's default `gpt-4`?** Two reasons.
First, model-pinning consistency: the bundle JSON records exactly one
LLM model per run, and using a different evaluator model would mean
the bundle's `target_system.llm_model` would not cover the RAGAS
calls. Second, cost: `gpt-4` evaluation would 10× the per-run RAGAS
cost without a corresponding accuracy gain on these three metrics
(RAGAS's own benchmarks show convergent results down to mid-tier
models).

**Why `AsyncOpenAI` for both the LLM and the embeddings, and why we
bypass RAGAS's sync `score()` entirely.** RAGAS 0.4 has two coupled
quirks that surfaced once the wrapper was exercised inside a Jupyter
kernel:

1. *RAGAS refuses sync `.score()` when an event loop is already
   running.* `BaseMetric.score()` checks `asyncio.get_running_loop()`
   and raises `RuntimeError: Cannot call sync score() from an async
   context. Use ascore() instead.` Jupyter / IPython kernels keep a
   live event loop in the background, so every `score()` call from a
   notebook fails with this error. `nest_asyncio.apply()` does not
   help here — RAGAS isn't *trying* to nest, it's *refusing* to. The
   wrapper bypasses `.score()` and calls `metric.ascore(...)` itself,
   wrapped in our own `asyncio.run()`. With `nest_asyncio` patched at
   scorer-build time, `asyncio.run()` works inside Jupyter; outside
   Jupyter (pytest, scripts, batch runner) the patch is a no-op.

2. *RAGAS's async path requires async clients on both LLM and
   embeddings.* `ascore()` for `Faithfulness` / `ContextRelevance`
   calls `llm.agenerate(...)` which the sync `OpenAI` client refuses
   with `TypeError: Cannot use agenerate() with a synchronous
   client`. `ascore()` for `AnswerRelevancy` *additionally* calls
   `embeddings.aembed_text(...)` which has the same constraint on
   the embedding client. The wrapper passes a single `AsyncOpenAI`
   instance to both `llm_factory(..., client=async_client)` and
   `OpenAIEmbeddings(client=async_client)`.

This async-everywhere choice means: RAGAS works identically in pytest,
scripts, and Jupyter notebooks; the SQLiteCache underneath still fires
on repeated (query, context, answer) triples; and the wrapper's
`try/except` records the original error class on any new RAGAS-API
drift so future readers can diagnose without re-instrumenting.

**Why a separate embedder (`text-embedding-3-small`) for Answer
Relevance, not the project's `bge-small-en-v1.5`?** Answer Relevance
embeds the original query and the LLM-reverse-engineered question and
compares them — this is *evaluation geometry*, not retrieval geometry.
Reusing the retrieval embedder would conflate the two. RAGAS's default
(`text-embedding-3-small`) is the principled choice; the small extra
cost is recorded in §7.3 cost arithmetic.

**Why wrap every RAGAS call in `try/except`?** Spec §10's risk register
flags "RAGAS metrics return NaN on edge cases" as high-likelihood. The
wrapper records `None` per metric on failure with the reason in
`notes`, so the bundle JSON preserves *why* a score is missing. A
silent NaN-to-0 collapse (the spec's quick-fix suggestion) was rejected
because 0 is also a valid RAGAS score, and conflating "scored 0" with
"failed to score" loses important information at the analysis stage.

---

## 7. Cross-cutting design choices

These decisions span multiple components and don't belong to a single
diagram. Each is sourced for direct lift into Chapter 3.

### 7.1 Model choices

**Embedding model: `BAAI/bge-small-en-v1.5`.** ~33M parameters, runs
CPU-only, top of the MTEB leaderboard for its size class. Picked for
speed of iteration on a laptop without GPU; production RAG systems
overwhelmingly use models in this size class. Alternatives considered:
`all-MiniLM-L6-v2` (smaller, slightly worse retrieval quality) —
rejected because bge-small's retrieval geometry is closer to what
PoisonedRAG used.

**Target LLM: `gpt-4o-mini-2024-07-18`.** Cost cap (~$0.15 per million
input tokens) lets the full 300-run experiment matrix fit under the spec
§2 hard cap of $50. Capable enough that hand-templated overrides matter
(if the model were too weak, every attack would succeed trivially — no
adaptation signal). Alternatives: `gpt-4o` (10× cost — overkill for the
experiment matrix); `llama3.1:8b` via Ollama (free but slower iteration,
non-deterministic batching, kept as the spec §9 cost tripwire fallback).

**Temperature = 0.** Deterministic outputs make exploit bundles
re-executable. RAGAS metrics computed on a non-deterministic generator
output would have different values across re-runs, breaking the
reproducibility contribution.

### 7.2 Reproducibility primitives

**`SQLiteCache` on every LLM call.** Set globally in
`redteam.target.generator` via LangChain's `set_llm_cache`. Re-runs hit
the cache; identical (model, prompt) pairs return cached completions.
Cost arithmetic: one full 300-run matrix costs ~$X uncached, ~$0
re-cached. The Day-9 experiment can be re-run cheaply during writeup.

**`index_state_hash` (SHA-256 over sorted doc_ids).** Recorded in every
bundle's `execution` block. Two bundles with the same `index_state_hash`
were generated against identical corpus state, which is what makes
cross-bundle comparisons meaningful. The hash is computed in
`Retriever.get_state_hash`.

**`prompt_template_hash` (SHA-256 of the un-rendered template).**
Recorded in every bundle. If the template changes between runs, the hash
changes and bundles produced before/after the change are distinguishable
without comparing prompt strings byte-by-byte.

**Fixed seeds.** Every random source — corpus sampling
(`load_nq_slice(seed=42)`), planner RNG (`Planner(seed=...)`),
payload-id hashing (`generate_*_payload(seed=...)`) — takes an explicit
seed argument. Default 42 across the project; the 3-seed Day-9 matrix
uses 42 / 7 / 1729.

### 7.3 Caching + cost discipline

**Cache hit rule:** for the cache to fire, the *prompt string* must be
byte-identical. Re-runs of the same experiment matrix are full cache
hits (zero new spend); re-runs after editing the prompt template invalidate
all entries (rebuild from scratch). This is why the prompt template is
declared as a module-level constant with its hash baked in — accidental
edits surface immediately as a test failure
(`test_prompt_template_hash_matches_template`).

**Tripwire from spec §9:** if API spend hits $30 (60% of cap), switch
the LLM constant to Ollama `llama3.1:8b` and re-run cached. Recorded as
a Day-9 fallback rather than a Day-1 build choice because development
iteration is much faster on hosted gpt-4o-mini, and the cache means
final-experiment cost is amortised.

### 7.4 Scope discipline

**Two attack families, not three.** Spec §2 "drop the third" rule. The
two-family choice is what enables a *cross-family* comparison axis in
Chapter 6 (the most natural finding-shape for the 50-query × 2-family
matrix). Three families would have given a three-way comparison without
adding a methodologically distinct axis.

**One LangGraph, not many.** Spec §2. A second graph would have meant a
duplicated node-set for marginal payoff. The conditional-edge structure
already supports adaptation within one graph.

**One evaluator stack (RAGAS, no TruLens).** Spec §2. TruLens is named
as a Future Work conditional. Day 7 builds RAGAS only; if it lands fast,
spec allows TruLens — but the timing is tight and `FUTURE_WORKS.md` §5.1
records the deferral.

**No defences.** Spec §2 explicitly evaluates *attacks*, not defences.
Adding a defence layer would have doubled the design surface for marginal
research-question payoff. Defence implementation is `FUTURE_WORKS.md`
§5.3 — the natural symmetric extension.

### 7.5 What's deliberately NOT in this codebase

- No HTTP layer: the framework targets a single in-process pipeline
  (`FUTURE_WORKS.md` §1.1).
- No web dashboard: the planned `notebooks/03_results_analysis.ipynb`
  serves the same audience (`FUTURE_WORKS.md` §1.2).
- No authentication, no multi-user state: this is a research artefact,
  not a service.
- No persistence across `build_graph` calls: planner memory and exploit
  bundles are in-memory or per-process. Day-8 bundle JSON is the only
  cross-run persistence layer.

These are *consciously* missing — recorded in `FUTURE_WORKS.md` so
supervisors and examiners can see the deferrals are deliberate
methodological choices, not oversights.
