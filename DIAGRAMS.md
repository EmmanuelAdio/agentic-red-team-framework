# Diagrams

Mermaid-format diagrams for the framework. Render natively in GitHub or VS Code (with the
*Markdown Preview Mermaid Support* extension); export to PDF/PNG for Chapter 3 (Design)
during the writeup phase.

Sections marked **(placeholder)** will be populated on the day named in their heading.

---

## 1. System architecture

The framework has four logical layers: the **target RAG pipeline** (the system under test),
the **attack family modules** (Days 3–4), the **agent layer** (LangGraph plan→generate→execute→evaluate
loop, Days 5–6), and **metrics + bundles** (Days 7–8). Each layer feeds the next:
attacks produce payloads, agents orchestrate which attack runs when, the executor
applies attacks against the target, and the evaluator scores the result and writes
a reproducible exploit-bundle JSON.

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

---

## 2. Threat model — what the attacker can and cannot do

Black-box-with-corpus-write (spec §3). Arrows from the attacker only land on what they
can influence; the retriever weights, LLM weights, and system prompt are *not* reachable.

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

This matches PoisonedRAG and EchoLeak's threat profile exactly: the adversary writes
content that the retriever later pulls, but cannot touch any model internals.

---

## 3. Attack-flow sequence (shared by IPI and corpus poisoning)

Both attack families share the same delivery pattern: insert a payload document via
`Retriever.add_documents`, run the target query through the pipeline, compute the ASR
triple from the result, and remove the payload via `Retriever.remove_documents` so the
index returns to its pre-attack state. The `try/finally` in the executor guarantees the
remove step runs even if the pipeline call raises.

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

---

## 4. LangGraph workflow

The agentic loop is a 4-node LangGraph: `plan → generate → execute → evaluate`,
with one conditional edge back to `plan` (or to `END`) at the bottom of the loop.
Day 5's planner is a deterministic round-robin over the two attack families;
Day 6 swaps in an ε-greedy planner with success-rate memory. The evaluator
computes only the ASR (Attack Success Rate) triple inline — RAGAS scoring
arrives in Day 7's `metrics` module.

The conditional-edge predicate (`should_continue`) ends the loop when the
verdict is `"success"` (no point retrying a working exploit) **or** when the
iteration counter has reached `max_iterations`. Otherwise it loops back to
`plan` for another attack family.

```mermaid
flowchart TD
    Start((Start)) --> Plan
    Plan[plan<br/>round-robin: PI ↔ poisoning] --> Generate
    Generate[generate<br/>build IPI / poison payload] --> Execute
    Execute[execute<br/>add → run → remove<br/>try/finally cleanup] --> Evaluate
    Evaluate[evaluate<br/>ASR-r / ASR-a / ASR-t<br/>verdict + history] --> Decide{should_continue?}
    Decide -- "verdict==success<br/>or iter≥max" --> End((End))
    Decide -- "else" --> Plan
```

The `try/finally` block inside `execute` guarantees that the payload's
`doc_id` is removed from the Chroma index even if `RAGPipeline.run` raises;
this is what keeps the `index_state_hash` invariant across the ~300-run
experiment matrix on Day 9.

---

## 5. Exploit-bundle structure (placeholder — Day 8)

Visual of the JSON schema from spec §7 (`target_system`, `attack`, `execution`,
`evaluation`, `reproducibility` blocks). Diagram added on Day 8.
