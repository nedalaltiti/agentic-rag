# Agentic RAG (OpenWebUI + pgvector + Ollama)

This repo is an end-to-end **agentic RAG** system: ingest PDFs, store embeddings in **Postgres/pgvector**, answer questions through an **OpenAI-compatible** API for **OpenWebUI**, and trace activity in **Arize Phoenix**.

## What’s included

* **Indexer CLI** (Docling → chunk → embed → store in pgvector)
* **Backend API** (FastAPI + `/v1/chat/completions` + agent tools)
* **Evaluator CLI** (RAGAS metrics over a generated/curated test set)
* **Observability** (Phoenix traces for retrieval + tool calls)

## Design choices (at a glance)

| Area | Choice | Where |
|------|--------|-------|
| REST API contract | OpenAI-compatible (`/v1/chat/completions`, `/v1/models`) | `src/agentic_rag/backend/api/v1/` |
| Chunking strategy | Heading-first contextual chunking with optional LLM context | `src/agentic_rag/indexer/chunking.py` |
| Embedding model | `qwen3-embedding:0.6b` (Ollama) | `.env.example` |
| LLM model | `qwen3:1.7b` (Ollama) | `.env.example` |
| Retrieval | Hybrid (pgvector + Postgres full-text) with RRF fusion | `src/agentic_rag/backend/rag/retriever.py` |
| Re-ranking | LLM reranker (Ollama) | `src/agentic_rag/backend/rag/reranker.py` |
| Agent prompts | Jinja2 prompts synced to Phoenix | `src/agentic_rag/prompts/` |

## Demo flow

1. Start the stack
2. Drop PDFs into `data/raw/`
3. Run the indexer
4. Open OpenWebUI and chat
5. Open Phoenix and inspect traces
6. Run evaluation and review RAGAS scores

## Quick start

```bash
git clone <repo-url>
cd agentic-rag
cp .env.example .env

docker compose up -d
curl http://localhost:8000/health
```

On first launch the `ollama-init` service automatically pulls the models
defined in `.env` (`LLM_MODEL` and `EMBEDDING_MODEL`), and the backend
applies SQL migrations on startup — no manual steps required.

> **Mac with host Ollama (Metal GPU):** Use the compose override to skip the
> containerised Ollama and its init job:
> ```bash
> docker compose -f docker-compose.yml -f docker-compose.mac.yml up -d
> ```
> You must pull the models yourself: `ollama pull qwen3:1.7b && ollama pull qwen3-embedding:0.6b`

> **Full reset:** To wipe all data and start fresh:
> ```bash
> docker compose down -v          # removes containers + volumes
> docker compose up -d            # recreates everything
> ```

> **Local Development (outside Docker):** The `.env.example` uses Docker service names
> (`postgres`, `ollama`, `phoenix`). If running locally without Docker, update these to
> `localhost` in your `.env` file. Note: if Docker Compose is running, Postgres is on
> host port **5433**, not 5432:
> ```
> DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5433/ragdb
> OLLAMA_BASE_URL=http://localhost:11434
> PHOENIX_COLLECTOR_ENDPOINT=http://localhost:6006/v1/traces
> ```

### Index documents

Put PDFs in `data/raw/` then:

```bash
agentic-index --source data/raw/
```

**Index versioning:** If you change the embedding model, tokenizer, or chunking settings, bump
`INDEX_VERSION` (in `.env`) and re-run the indexer. This keeps retrieval aligned to the
correct embedding space.

**Chunking modes:**

| Mode | Command | What it does | When to use |
|------|---------|-------------|-------------|
| `fast` | `agentic-index --source data/raw/` | Structured prefix only (`[Doc: ...][Section: ...]`) | Default. Fast, deterministic, good for most documents |
| `llm` | `agentic-index --source data/raw/ --mode llm` | Prefix + LLM-generated context summary per chunk | When embedding quality matters more than indexing speed. Uses first 6000 chars of the document for context, so works best with focused documents. Non-deterministic. |

For PDFs, Docling extracts page count and the chunker estimates page numbers per chunk based on character offsets. Markdown files don't have page numbers.

### Chat (OpenWebUI)

* OpenWebUI: `http://localhost:3000`
* Backend API: `http://localhost:8000`
* Ollama: `http://localhost:11434`

The backend exposes:

* `GET /v1/models`
* `POST /v1/chat/completions`
* `GET /docs` — interactive Swagger UI

### Ollama modes

1. **Default (portable):** Use the Ollama container. `OLLAMA_BASE_URL=http://ollama:11434`
2. **Optional (Mac speed):** Use host Ollama with Metal acceleration:
```bash
docker compose -f docker-compose.yml -f docker-compose.mac.yml up -d
```

### Evaluate (RAGAS)

```bash
# Make sure evaluator model is pulled
ollama pull qwen3:4b

# 1. Generate a synthetic test set from indexed chunks
agentic-eval generate --num-samples 10 --output eval_testset.json

# 2. Run retrieval + answer pipeline and compute RAGAS metrics
agentic-eval evaluate --testset eval_testset.json --output eval_results.json

# 3. Pretty-print the results
agentic-eval report --results eval_results.json
```

### Continuous evaluation (monitoring)

Run evaluations on a schedule to monitor retrieval quality over time:

```bash
agentic-eval monitor --testset eval_testset.json --output-dir eval_runs --interval-seconds 3600
```

Set `--skip-ragas` for faster retrieval-only monitoring.

**Note on evaluation data:** `agentic-eval generate` creates a synthetic Q/A dataset from random chunks. If you need curated ground-truth, provide a JSON file in the same format (`question`, `ground_truth`, and optional metadata) and pass it to `agentic-eval evaluate`.

RAGAS evaluation uses a **separate evaluator model** (`EVAL_MODEL`, default: `qwen3:4b`) to avoid self-evaluation bias — the chat model does not judge its own output. Pull it before running evaluation:

```bash
# If using Docker:
docker compose exec ollama ollama pull qwen3:4b

# If running Ollama locally:
ollama pull qwen3:4b
```

Override the evaluator model via `EVAL_MODEL` in `.env` if needed.

### Traces (Phoenix)

Phoenix UI: `http://localhost:6006`

What to check:
* retrieved chunks and scores
* tool call sequence (retriever → rerank → response)

### Prompt management (Phoenix)

Prompts are stored as Jinja2 templates in `src/agentic_rag/prompts/`. Some are only used in optional modes (agent mode, LLM chunking, or eval generation).

| Template | Used by | Purpose |
|----------|---------|---------|
| `system_prompt.j2` | Chat endpoint | System instructions for the chat model |
| `user_prompt.j2` | Chat endpoint, evaluator | Main RAG prompt: injects query + retrieved context |
| `context_generation_template.j2` | Indexer (`--mode llm`) | Generates contextual summaries per chunk (Anthropic-style) |
| `reranker_template.j2` | LLM reranker | Scores chunk relevance to a query |
| `researcher_backstory.j2` | CrewAI researcher agent | Agent persona and instructions |
| `writer_backstory.j2` | CrewAI writer agent | Agent persona and instructions |
| `qa_generation_template.j2` | Evaluator (testset generation) | Generates synthetic Q/A pairs from chunks |
| `scope_anchors.txt` | Scope gate | Anchor phrases used to classify in-scope queries |

**Phoenix sync:** When `PHOENIX_PROMPT_SYNC=true` (default in `.env.example`), the backend and CLI tools push all templates to Phoenix on startup and tag them with `PHOENIX_PROMPT_TAG` (default: `development`). In production, set the tag to `production` to version prompts in the Phoenix UI.

When `PHOENIX_PROMPT_SYNC=false`, prompts are served from the local `.j2` files only. Disable sync during local development to avoid unnecessary Phoenix calls.

In production (`ENVIRONMENT=prod`), `PromptRegistry.render()` and `get_template()` fetch the tagged prompt from Phoenix first and fall back to local if Phoenix is unreachable.

**Phoenix checklist:**
1. Set `ENVIRONMENT=prod` and `PHOENIX_PROMPT_TAG=demo` (in `.env` or your shell)
2. Start backend or CLI
3. In Phoenix UI, confirm prompts exist under the tag
4. Edit a prompt, re-run a query, and confirm the response changes

### Retrieval & reranker tuning

**RRF weights:** Configure in `.env` with `RRF_WEIGHT_VECTOR` and `RRF_WEIGHT_KEYWORD`.

**Reranker settings:**

| Setting | Default | Notes |
|---------|---------|-------|
| `TOP_K_RERANK` | 5 | Final number of chunks returned after reranking |
| `RERANKER_TIMEOUT` | 30s | Total timeout; falls back to retrieval order on expiry |
| `TOP_K_RETRIEVAL` | 10 | Candidates from hybrid search before reranking |

The reranker is only active in agent mode (CrewAI path). Fast RAG skips it entirely.

## Citation format

Each response includes structured citations with complete source metadata. The backend returns an `AgentResponse` with a `citations` array containing:

**Citation Schema:**
```json
{
  "document_id": "uuid",
  "chunk_id": "uuid", 
  "file_name": "document.pdf",
  "page_number": 12,
  "section_path": "Introduction > Overview",
  "chunk_text": "Retrieved text snippet...",
  "score": 0.92
}
```

**Fields:**
- `document_id`: UUID of source document
- `chunk_id`: UUID of specific chunk
- `file_name`: Original filename
- `page_number`: Page number (null if unavailable)
- `section_path`: Hierarchical section location (e.g., "Chapter 1 > Section 1.2")
- `chunk_text`: Actual retrieved text
- `score`: Relevance score (0.0-1.0)

The agent's text response typically includes inline citations.

## Services

|     Service |  Port | Notes                            |
| ----------: | :---: | -------------------------------- |
| Backend API |  8000 | FastAPI (`/v1/chat/completions`) |
|   OpenWebUI |  3000 | Chat frontend                    |
|  PostgreSQL |  5432 | pgvector store                   |
|      Ollama | 11434 | local LLM + embeddings           |
|     Phoenix |  6006 | tracing dashboard                |

**Port notes:**
- PostgreSQL is mapped to **host port 5433** (not 5432) to avoid conflicts with a local Postgres. When connecting from outside Docker, use `localhost:5433`. Inside Docker, services use `postgres:5432`.
- OpenWebUI is mapped to **host port 3000** (container port 8080).

**OpenWebUI integration:** Configure `OPENAI_API_BASE_URL=http://backend:8000/v1` and `OPENAI_API_KEY=dummy`. OpenWebUI will discover models via `/v1/models`.

**Session persistence:** The API returns an `X-Session-Id` header. Reuse it on subsequent requests to keep conversation memory.

**Health & service status:** `GET /health` checks database, Ollama, and Phoenix. If DB or Ollama are down, status is `unhealthy`. If Phoenix is down, status is `degraded`.

## Known limitations (current)

* PDFs with complex tables/scans depend heavily on Docling parsing quality.
* Retrieval quality depends on chunking + embedding model choice.
* First launch may take several minutes while Ollama models are downloaded.

## Troubleshooting

**Backend can’t reach Ollama**

* Check `OLLAMA_BASE_URL` and that the `ollama` service is up.

**Mac GPU Ollama (optional override)**

Use the compose override:
```bash
docker compose -f docker-compose.yml -f docker-compose.mac.yml up -d
```

**No results retrieved**

* Confirm the indexer ran successfully and vectors are in Postgres.
* Check DB connection string and schema migration ran.

## Local development (without Docker)

```bash
# 1. Install the project in editable mode
pip install -e ".[dev,eval]"

# 2. Start Postgres (pgvector), Ollama, and Phoenix however you prefer,
#    then point your .env at localhost (see Quick start note above).

# 3. Run the database migrations manually
psql "$DATABASE_URL" -f migrations/001_init_extensions.sql
psql "$DATABASE_URL" -f migrations/002_create_tables.sql
psql "$DATABASE_URL" -f migrations/003_create_indexes.sql

# 4. Pull the required Ollama models
ollama pull qwen3:1.7b
ollama pull qwen3-embedding:0.6b

# 5. Start the backend
agentic-api
```

API docs are available at `http://localhost:8000/docs` (Swagger UI).

## Development

```bash
pip install -e ".[dev,eval]"
ruff check src/ tests/
pytest -v
mypy src/agentic_rag
```

## Testing prerequisites

Install test dependencies before running `pytest`:
```bash
pip install -e ".[dev,eval]"
```

## License

MIT
