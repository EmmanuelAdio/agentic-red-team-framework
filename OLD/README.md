# Agentic Red-Team Framework

Controlled corpus + baseline RAG platform for running reproducible red-team style experiments.

## Current Status

Implemented now:
- Corpus ingestion from local `.txt` and `.json` files
- Versioned corpus snapshots (`corpus_version`)
- Chunking + embedding pipeline
- Retrieval with two backends:
  - `local`: in-process cosine search
  - `mongo`: Atlas `$vectorSearch` when available, automatic cosine fallback otherwise
- Generation with multiple providers:
  - `none` (retrieve-only)
  - `openai` (chat completions)
  - `ollama` / `local_model` / `local` (local HTTP endpoint)
- Full trace persistence (`query_traces`, `rag_responses`)
- FastAPI endpoints for ingest, list docs, query, and trace lookup
- Test suite for chunking, loading, retrieval, response contracts, generator providers, and vector-search fallback behavior

Not implemented yet (next project phases):
- Planner / attack orchestration loop
- Attack generation/evaluation modules
- Dashboard/experiment UI

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

Windows PowerShell:

```powershell
Copy-Item .env.template .env
```

## Run MongoDB

Local MongoDB example:

```bash
docker run -d --name rag-mongo -p 27017:27017 mongo:7
```

If using Atlas, set `MONGODB_URI` to your cluster URI.

## Run the API

```bash
uvicorn app.main:app --reload
```

Health check:

```bash
curl http://localhost:8000/health
```

## API Endpoints

- `POST /api/v1/corpus/ingest`
- `POST /api/v1/corpus/versions/prune`
- `GET /api/v1/corpus/documents`
- `POST /api/v1/rag/query`
- `GET /api/v1/traces/{trace_id}`

## Typical Workflow

1. Ingest corpus:

```bash
curl -X POST http://localhost:8000/api/v1/corpus/ingest \
  -H "Content-Type: application/json" \
  -d '{}'
```

2. Query:

```bash
curl -X POST http://localhost:8000/api/v1/rag/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is retrieval-augmented generation?"}'
```

3. Inspect trace:

```bash
curl http://localhost:8000/api/v1/traces/<trace_id>
```

4. Prune older corpus versions (keeps latest by default):

```bash
curl -X POST http://localhost:8000/api/v1/corpus/versions/prune \
  -H "Content-Type: application/json" \
  -d '{"delete_count": 2, "keep_latest": true}'
```

## Configuration

Core keys in `.env`:

- `VECTOR_BACKEND=local|mongo`
- `EMBEDDING_PROVIDER=deterministic_stub|openai|sentence_transformers|local_sentence_transformers|local_model`
- `EMBEDDING_MODEL=...`
- `EMBEDDING_DIMENSION=...`
- `QUERY_MODE=retrieve_only|generate`
- `LLM_PROVIDER=none|openai|stub|ollama|local_model|local`
- `LLM_MODEL=...`
- `OPENAI_API_KEY=...` (required for OpenAI LLM generation)
- `OLLAMA_BASE_URL=http://localhost:11434`
- `OLLAMA_TIMEOUT_SECONDS=30`
- `ATLAS_VECTOR_INDEX_NAME=chunks_embedding_index`

Notes:
- If `QUERY_MODE=retrieve_only` or `LLM_PROVIDER=none`, generation is skipped.
- `mongo` backend attempts Atlas `$vectorSearch` first and falls back automatically if unavailable.
- Keep embedding dimensions aligned with your embedding model and vector index configuration.

## Testing

Run all tests:

```bash
python -m pytest -q
```

Tests run against isolated mock Mongo by default (`MONGO_USE_MOCK=true` in test fixtures).

## Progress Summary

The project has completed a robust baseline RAG foundation: ingestion, retrieval, configurable generation, and traceability are in place and tested.  
The next major milestone is to build the actual red-team agentic layer (planner + attack/evaluation loops) on top of these stable interfaces.

## Reproducible Data Pipeline (Corpus vs Eval vs Attack)

The thesis pipeline explicitly separates three artifact types:
- Retrieval corpus documents (`data/corpus_*.jsonl`) used only for retrieval grounding.
- Evaluation datasets (`task_type=qa|summarization`) used only for scoring/model evaluation.
- Adversarial datasets (`task_type=attack`) used only for red-team prompt replay/sampling.

This separation prevents benchmark leakage and keeps experiments controlled and repeatable.

### Build Retrieval Corpus

```bash
python scripts/transform_structured_sources.py \
  --halls data/accommodation_halls.json \
  --courses data/all_ug_courses.json \
  --output data/corpus_structured.jsonl

python scripts/build_wiki_corpus.py --output data/corpus_wiki.jsonl
python scripts/build_poisoned_corpus.py --output data/corpus_poisoned.jsonl
python scripts/merge_corpora.py \
  --structured data/corpus_structured.jsonl \
  --wiki data/corpus_wiki.jsonl \
  --poisoned data/corpus_poisoned.jsonl \
  --output data/corpus_retrieval.jsonl
```

### Load Eval/Attack Inputs (Kept Separate)

```bash
python scripts/load_eval_datasets.py --dataset-name squad --input-path data/eval/squad.json --output-path data/eval/squad_eval.jsonl
python scripts/load_eval_datasets.py --dataset-name cnn_dailymail --input-path data/eval/cnn_dailymail.jsonl --output-path data/eval/cnn_eval.jsonl
python scripts/load_eval_datasets.py --dataset-name gigaword --input-path data/eval/gigaword.jsonl --output-path data/eval/gigaword_eval.jsonl
python scripts/load_eval_datasets.py --dataset-name jailbreakbench --input-path data/attacks/jailbreakbench.jsonl --output-path data/attacks/jbb_eval.jsonl
```

### Download Eval Datasets (Automated)

You can download and normalize common eval datasets in one step:

```bash
python scripts/download_eval_datasets.py --datasets squad cnn_dailymail --output-root data --limit 1000
```

Optional Gigaword:

```bash
python scripts/download_eval_datasets.py --datasets gigaword --output-root data --hf-split test --limit 1000
```

Outputs are written under `data/eval/`:
- raw downloads (e.g., `squad.json`, `cnn_dailymail.jsonl`)
- normalized files for evaluation (e.g., `squad_eval.jsonl`, `cnn_eval.jsonl`)

Notes:
- SQuAD is downloaded from the official public file URL.
- CNN/DailyMail and Gigaword are pulled via Hugging Face `datasets` (install if needed: `pip install datasets`).

### Run Batch Evaluation Against RAG API

After loading eval/attack datasets, run them through the `/api/v1/rag/query` endpoint and score outputs:

```bash
python scripts/run_eval.py \
  --input-path data/eval/squad_eval.jsonl \
  --api-base http://localhost:8000 \
  --api-prefix /api/v1 \
  --top-k 4 \
  --output-dir data/eval/runs \
  --run-name squad_baseline
```

The evaluator writes a timestamped run folder with:
- `results.jsonl` (per-sample output, metrics, trace IDs)
- `summary.json` (aggregated metrics and counts)
- `errors.jsonl` (failed requests, if any)

Task-aware metrics are applied automatically from `task_type`:
- `qa`: exact match + token F1 against `reference_answer`
- `summarization`: ROUGE-1-style token F1 against `reference_summary`
- `attack`: heuristic `attack_success` / `attack_blocked` from response refusal patterns

### Ingest Retrieval Corpus Into Mongo

```bash
python scripts/ingest_corpus_to_mongo.py \
  --input-path data/corpus_retrieval.jsonl \
  --collection rag_documents \
  --clear-first
```

The Mongo ingestion script stores retrieval docs plus embeddings and builds indexes for filtering/analysis:
`doc_id`, `entity_id`, `entity_type`, `doc_type`, `attack_type`, `source_type`.

### Clear MongoDB Data

Prune older corpus versions only (keeps latest by default; does not touch trace logs):

```bash
python scripts/clear_mongo_database.py --prune-versions 3 --yes
```

Clear collections in the configured database while preserving trace logs (`query_traces`, `rag_responses`) by default:

```bash
python scripts/clear_mongo_database.py --yes
```

Clear collections including trace logs:

```bash
python scripts/clear_mongo_database.py --include-traces --yes
```

Drop the entire configured database:

```bash
python scripts/clear_mongo_database.py --drop-database --yes
```

### GLUE / SuperGLUE Note

GLUE and SuperGLUE are useful optional references for later expansion, but they are not central to this first implementation because the core thesis loop is retrieval-grounded QA/summarization plus adversarial prompt red-teaming.
