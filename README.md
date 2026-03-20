# Agentic Red-Team Framework (Step 1 Baseline)

This repository currently implements **Step 1 only** of the thesis project:

> Controlled corpus + baseline RAG testbed for later agentic red-team evaluation.

It does **not** yet implement planner/attack-generator/evaluator/dashboard logic.

## What This Baseline Includes

- Local corpus ingestion (`.txt` and `.json`)
- Versioned corpus snapshots (`corpus_version` per ingest run)
- Document/chunk metadata with `attack_label` (`benign`, `poisoned`, `misleading`)
- Chunking via LangChain text splitter
- Embedding abstraction with deterministic stub implementation
- Dual retrieval backend abstraction:
  - `local`: in-process cosine retrieval over stored vectors
  - `mongo`: Mongo-backed retrieval abstraction (portable baseline)
- Baseline RAG service with full query trace logging
- FastAPI endpoints for ingest/list/query/trace
- MongoDB persistence for documents, chunks, corpus versions, query traces, and responses
- Pytest suite covering loading, chunking, retrieval, and response contract

Default query mode is `retrieve_only` (no external LLM required).

## Project Structure

```text
app/
  api/
  core/
  corpus/
  retrieval/
  rag/
  db/
  main.py
data/sample_corpus/
tests/
.env.template
requirements.txt
```

## Setup

1. Create and activate a Python 3.11+ virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Create `.env` from template:

```bash
cp .env.template .env
```

On Windows PowerShell:

```powershell
Copy-Item .env.template .env
```

## Run MongoDB

Run local MongoDB (example with Docker):

```bash
docker run -d --name rag-mongo -p 27017:27017 mongo:7
```

If you already have MongoDB installed locally, keep `MONGODB_URI=mongodb://localhost:27017`.

## Run the API

```bash
uvicorn app.main:app --reload
```

Health check:

```bash
curl http://localhost:8000/health
```

## Ingest Sample Corpus

```bash
curl -X POST http://localhost:8000/api/v1/corpus/ingest \
  -H "Content-Type: application/json" \
  -d '{}'
```

This ingests `data/sample_corpus` by default and returns a new `corpus_version`.

## Query Baseline RAG

```bash
curl -X POST http://localhost:8000/api/v1/rag/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is retrieval-augmented generation?"}'
```

Response includes:

- `final_answer` (`null` in retrieve-only mode)
- `retrieved_chunks` and similarity scores
- `prompt_context` and `prompt_used`
- `model_metadata`
- `trace_id`

## Inspect Trace

```bash
curl http://localhost:8000/api/v1/traces/<trace_id>
```

## Configuration Notes

Important `.env` keys:

- `VECTOR_BACKEND=local|mongo`
- `EMBEDDING_PROVIDER=deterministic_stub`
- `QUERY_MODE=retrieve_only|generate`
- `LLM_PROVIDER=none|stub`

Provider-specific adapters are intentionally deferred and marked with `TODO(provider)` comments.

## Run Tests

```bash
pytest -q
```

Tests use `mongomock` via environment overrides for isolated local execution.

## Extension Points for Later Steps

This baseline is designed to be called programmatically by future modules:

- RAG orchestration entrypoint: `BaselineRAGService.answer_query(...)`
- Retrieval abstraction: `Retriever` + `VectorIndex` implementations
- Ingestion pipeline: `CorpusIngestionService`
- Trace persistence: `query_traces` and `rag_responses` collections

These hooks are ready for adding planner, attack generation, evaluation, and experiment logging in later steps.
