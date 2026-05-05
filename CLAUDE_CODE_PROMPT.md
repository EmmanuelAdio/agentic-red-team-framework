# Claude Code Bootstrap Prompt — Agentic Red-Team Framework

Copy everything below the divider into a fresh Claude Code session in an empty directory. Claude Code will scaffold the entire project, ask you to commit, then walk through Day 1 implementation with you.

---

You are helping me build my MSci dissertation project at Loughborough University. I have **16 days** until submission (today is Mon 4 May 2026, deadline Tue 20 May 2026). Failure is not abstract — it costs my degree classification.

## Project context

I am building an **Agentic Red-Team Framework for LLM-Augmented Search Systems**. The framework autonomously plans, executes, and evaluates adversarial attacks against a Retrieval-Augmented Generation (RAG) pipeline at both retrieval and generation phases, producing reproducible exploit traces scored by reference-free integrity metrics.

The full specification is in `PROJECT_SPEC.md` in this directory. **Read it first.** Then read it again. Do not deviate from the scope it defines.

## Hard scope rules — these are not suggestions

1. **One** RAG pipeline only: LangChain + Chroma + `BAAI/bge-small-en-v1.5` + `gpt-4o-mini-2024-07-18` (or `llama3.1:8b` via Ollama as a free fallback).
2. **One** dataset: a 1k-document slice of Natural Questions from BEIR.
3. **Two** attack families only: prompt injection and PoisonedRAG-style corpus poisoning. Do **not** propose a third unless I explicitly ask.
4. **One** orchestration: a 4-node LangGraph (`plan → generate → execute → evaluate → loop`).
5. **One** evaluator stack: RAGAS Faithfulness/Answer Relevance/Context Relevance + custom ASR-r/ASR-a/ASR-t + rank-shift@k.
6. **No** dashboard. Plots only.
7. **No** TruLens unless I tell you Day 6 is done.
8. **No** GASLITE, no Joint-GCG, no defences, no second retriever, no second LLM.
9. Every LLM call goes through `langchain.cache.SQLiteCache`. Every run produces an exploit-bundle JSON conforming to the schema in `PROJECT_SPEC.md` §7.
10. `temperature=0`, fixed seeds, n=3 per experimental cell.

## Engineering principles

- **Single-author, 16 days.** Optimise for *working* over *clever*. Boring code that runs beats elegant code that doesn't.
- **No premature abstraction.** No abstract base classes for two concrete subclasses. No plugin systems. No config DSLs. Hardcode now, refactor never.
- **Type hints everywhere.** `from __future__ import annotations` at the top of every file. Use `TypedDict` for state, `dataclass` for value objects, `Pydantic BaseModel` only where it touches I/O.
- **Tests are minimal but real.** One unit test per attack family, one per metric, one end-to-end smoke test that runs `<60s`. No mocking the LLM in the smoke test — it must hit the real API or local Ollama.
- **Commit at the end of every day.** Even if the day went badly. Especially if the day went badly.
- **Lab notebook discipline.** Append a 5–10 line entry to `LAB_NOTEBOOK.md` after each work session: *what I did, what worked, what broke, what's next.*

## What I want you to do RIGHT NOW (Day 1)

Do these in order. Pause between each numbered task and let me confirm before continuing. Do not try to do all of Day 1 in one go.

1. **Bootstrap the repository.** Create the directory structure exactly as specified in `PROJECT_SPEC.md` §8. Empty `__init__.py` files where needed. Create:
   - `pyproject.toml` with `redteam` package, Python ≥3.11, deps: `langchain>=0.2`, `langgraph>=0.2`, `langchain-openai`, `langchain-community`, `chromadb>=0.5`, `ragas>=0.2`, `sentence-transformers`, `datasets` (HuggingFace), `pydantic>=2`, `python-dotenv`, `tenacity`, `numpy`, `pandas`, `matplotlib`, `pytest`.
   - `requirements.txt` (pinned).
   - `.env.example` listing `OPENAI_API_KEY=` only.
   - `.gitignore` covering `.env`, `__pycache__`, `data/corpus/`, `data/runs/`, `.chroma/`, `*.sqlite`, `.pytest_cache`, `.venv/`.
   - `LICENSE` (MIT, my name: Emmanuel Adio, 2026).
   - `README.md` stub with the project title, supervisor, 3-line description, and a "Setup" section with the four commands needed to reproduce.
   - `LAB_NOTEBOOK.md` stub with today's date as the first heading.

   Show me the directory tree when done. Wait for my "go" before continuing.

2. **Implement `src/redteam/config.py`.** A single module with:
   - `PROJECT_ROOT` resolved from `__file__`.
   - All paths (corpus dir, runs dir, results dir) as `Path` objects.
   - Model name constants (`EMBEDDING_MODEL`, `LLM_MODEL`, `LLM_TEMPERATURE=0.0`, `RETRIEVER_TOP_K=5`).
   - A `load_env()` function that calls `dotenv.load_dotenv()` and validates `OPENAI_API_KEY` is set.
   - **No** dependency on any other `redteam` module. This is the leaf of the dependency graph.

   Wait for my "go".

3. **Implement `src/redteam/target/corpus.py`.** Functions:
   - `load_nq_slice(n_docs: int = 1000, seed: int = 42) -> list[Document]`: pulls Natural Questions from `BeIR/nq` on HuggingFace, deterministically samples `n_docs` documents, returns a list of LangChain `Document` objects with `page_content` and `metadata={"doc_id": str, "source": "nq"}`.
   - `chunk_documents(docs, chunk_size=512, chunk_overlap=64) -> list[Document]`: standard `RecursiveCharacterTextSplitter`. Preserve `doc_id` in metadata.

   Include a `__main__` block that prints `len()` of the loaded slice and the first 200 chars of doc 0. I will run it manually and confirm.

4. **Implement `src/redteam/target/retriever.py`.** A `Retriever` class wrapping Chroma:
   - `__init__(persist_dir: Path, embedding_model_name: str)`: loads or creates a Chroma collection with the bge-small embedding.
   - `index(docs: list[Document]) -> None`: idempotent — skips re-indexing if the collection size matches.
   - `query(text: str, k: int = 5) -> list[RetrievedDoc]`: returns a list of `RetrievedDoc` (dataclass: `doc_id`, `content`, `score`, `rank`).
   - `add_documents(docs)` and `remove_documents(doc_ids)`: for runtime poisoning experiments.
   - `get_state_hash() -> str`: SHA-256 of sorted `doc_id` list — captures index state for the exploit bundle.

   Wait for my "go" before tasks 5+.

5. **Implement `src/redteam/target/generator.py`.** An `LLMGenerator` class:
   - Wraps `ChatOpenAI` with `temperature=0`, `cache=SQLiteCache(database_path=PROJECT_ROOT/".cache.sqlite")`.
   - One method: `generate(query: str, retrieved_docs: list[RetrievedDoc]) -> GeneratorOutput` where `GeneratorOutput` is a dataclass with `text`, `latency_ms`, `prompt_template_hash`.
   - Use the prompt template from `PROJECT_SPEC.md` §4.1 verbatim.
   - Hash the rendered prompt template (without the variable values) and store it.

6. **Implement `src/redteam/target/pipeline.py`.** A thin `RAGPipeline` class composing `Retriever` and `LLMGenerator`. One method `run(query: str) -> dict` that returns everything the executor needs to populate the exploit bundle.

7. **Write `scripts/01_build_corpus.py`** that calls `load_nq_slice`, chunks, and indexes into Chroma. **Write `scripts/02_run_baseline.py`** that loads 5 hardcoded test queries and prints the answers.

8. **Stop.** Tell me to run `python scripts/01_build_corpus.py` and then `python scripts/02_run_baseline.py`. Help me debug anything that breaks. Day 1 is done when both scripts run cleanly end-to-end.

## After Day 1, expect me to ask for

- Day 2: 50-query test set construction with expected-answer ground truth from NQ.
- Day 3: `attacks/prompt_injection.py` — generates instruction-override payloads, demonstrates one successful IPI.
- Day 4: `attacks/corpus_poisoning.py` — generates poisoned documents that contain target answers and pass topical relevance.
- Day 5: LangGraph 4-node skeleton in `orchestration/graph.py`.
- Day 6: Planner agent (ε-greedy over attack family) + LLM-driven exploit generator prompts.
- Day 7: RAGAS integration + ASR-r/ASR-a/ASR-t + rank-shift@k.
- Day 8: Exploit bundle JSON I/O + 50-run dry test.
- Day 9: Full experiments, ~300 runs.

Throughout: **always show me the smallest possible working version first**, then layer complexity. If you find yourself writing more than 100 lines without me running anything, stop and ask me to run what you have.

## How to disagree with me

If you think I'm making a mistake — picking the wrong abstraction, fighting LangGraph, optimising prematurely — tell me. Don't just comply. But once I've made a final decision (especially scope decisions), implement it without re-litigating.

## What success looks like on Day 1 EOD

I run `python scripts/02_run_baseline.py` and see five questions answered by the RAG pipeline using retrieved NQ documents. The answers don't have to be correct. They just have to come out of the pipeline. The Chroma index is persistent. The cache works. We commit.

Begin with task 1.
