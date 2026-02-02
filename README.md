# Agentic RAG (OpenWebUI + pgvector + Ollama)

This repo is a small, end-to-end **agentic RAG** system built for a case study: ingest PDFs, store embeddings in **Postgres/pgvector**, answer questions through an **OpenAI-compatible** API so it plugs into **OpenWebUI**, and keep the whole thing observable in **Arize Phoenix**.

It’s intentionally “boring” in the good way: reproducible Docker setup, clear module boundaries (indexer / backend / evaluator), and enough instrumentation to debug retrieval and prompt issues.

## What’s included

* **Indexer CLI** (Docling → chunk → embed → store in pgvector)
* **Backend API** (FastAPI + `/v1/chat/completions` + agent tools)
* **Evaluator CLI** (RAGAS metrics over a generated/curated test set)
* **Observability** (Phoenix traces for retrieval + tool calls)

## Demo flow (the one reviewers can follow)

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

### Evaluate (RAGAS)

```bash
# 1. Generate a synthetic test set from indexed chunks
agentic-eval generate --num-samples 10 --output eval_testset.json

# 2. Run retrieval + answer pipeline and compute RAGAS metrics
agentic-eval evaluate --testset eval_testset.json --output eval_results.json

# 3. Pretty-print the results
agentic-eval report --results eval_results.json
```

RAGAS evaluation uses a **separate evaluator model** (`EVAL_MODEL`, default: `prometheus2:7b`) to avoid self-evaluation bias — the chat model does not judge its own output. Pull it before running evaluation:

```bash
# If using Docker:
docker compose exec ollama ollama pull prometheus2:7b

# If running Ollama locally:
ollama pull prometheus2:7b
```

Override the evaluator model via `EVAL_MODEL` in `.env` if needed.

### Traces (Phoenix)

Phoenix UI: `http://localhost:6006`

What I usually look at:

* which chunks were retrieved (and their scores)
* tool call sequence (retriever → rerank → final response)
* prompt inputs/outputs when the answer looks off

### Prompt management (Phoenix)

Prompts are stored as Jinja2 templates in `src/agentic_rag/prompts/`:

| Template | Used by | Purpose |
|----------|---------|---------|
| `user_prompt.j2` | Chat endpoint, evaluator | Main RAG prompt: injects query + retrieved context |
| `context_generation_template.j2` | Indexer (`--mode llm`) | Generates contextual summaries per chunk (Anthropic-style) |
| `reranker_template.j2` | LLM reranker | Scores chunk relevance to a query |
| `researcher_backstory.j2` | CrewAI researcher agent | Agent persona and instructions |
| `writer_backstory.j2` | CrewAI writer agent | Agent persona and instructions |

**Phoenix sync:** When `PHOENIX_PROMPT_SYNC=true` (default in `.env.example`), the backend pushes all templates to Phoenix on startup and tags them with `PHOENIX_PROMPT_TAG` (default: `development`). In production, set the tag to `production` so you can version and roll back prompts from the Phoenix UI without redeploying.

When `PHOENIX_PROMPT_SYNC=false`, prompts are served from the local `.j2` files only. Disable sync during local development to avoid unnecessary Phoenix calls.

In production (`ENVIRONMENT=prod`), `get_template()` fetches the tagged prompt from Phoenix first and falls back to local if Phoenix is unreachable.

### Retrieval & reranker tuning

**RRF weights:** The hybrid retriever fuses semantic (pgvector) and keyword (full-text) results using Reciprocal Rank Fusion. Weights are configurable via `RRF_WEIGHT_VECTOR` and `RRF_WEIGHT_KEYWORD` in `.env`. Equal weights (1.0/1.0) work well for most cases. Increase `RRF_WEIGHT_VECTOR` if your embedding model is strong; increase `RRF_WEIGHT_KEYWORD` for terminology-heavy domains.

**Reranker:** The LLM reranker scores each candidate chunk via an Ollama LLM call. This improves precision but adds latency proportional to the number of candidates. Key settings:

| Setting | Default | Notes |
|---------|---------|-------|
| `TOP_K_RERANK` | 5 | Final number of chunks returned after reranking |
| `RERANKER_TIMEOUT` | 30s | Total timeout; falls back to retrieval order on expiry |
| `TOP_K_RETRIEVAL` | 10 | Candidates from hybrid search before reranking |

The reranker is only active in agent mode (CrewAI path). Fast RAG skips it entirely.

For local GPU (e.g. Apple Silicon / 8GB VRAM), `qwen3:1.7b` with `TOP_K_RERANK=5` keeps reranking under 10 seconds. On CPU-only setups, consider setting `USE_CREWAI=false` to disable agent mode or reducing `TOP_K_RERANK` to 3.

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

**Example Response:**
```json
{
  "answer": "The company's revenue grew by 25% in Q4...",
  "citations": [
    {
      "document_id": "a1b2c3d4-...",
      "chunk_id": "e5f6g7h8-...",
      "file_name": "Annual Report 2024.pdf",
      "page_number": 12,
      "section_path": "Financial Results > Revenue",
      "chunk_text": "Q4 revenue increased 25% year-over-year...",
      "score": 0.92
    }
  ],
  "trace_id": "trace-xyz",
  "usage": {
    "prompt_tokens": 120,
    "completion_tokens": 85,
    "total_tokens": 205
  }
}
```

The agent's text response typically includes inline citations like: `"According to the Annual Report (p.12), revenue grew..."`

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

**OpenWebUI integration:** OpenWebUI is configured to use this backend as an OpenAI-compatible provider via `OPENAI_API_BASE_URL=http://backend:8000/v1`. The backend implements `/v1/chat/completions` and `/v1/models`, which is the minimum OpenWebUI needs. The API key is set to `dummy` since the backend doesn't require authentication. No additional OpenWebUI configuration is needed — it auto-discovers models and routes chat through the backend.

**Health & service status:** The `GET /health` endpoint checks database, Ollama, and Phoenix. Database and Ollama are critical — if either is down, overall status is `unhealthy`. Phoenix is non-critical — if unreachable, status reports `degraded` (not `unhealthy`). The system continues to function without Phoenix, but traces and prompt sync are unavailable. For assessment purposes, keep Phoenix running to demonstrate full observability.

## Known limitations (current)

* PDFs with complex tables/scans depend heavily on Docling parsing quality.
* Retrieval quality depends on chunking + embedding model choice.
* If Ollama doesn’t have the models pulled yet, first run will be slow.

## Troubleshooting

**Backend can’t reach Ollama**

* Check `OLLAMA_HOST` and that the `ollama` service is up.

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

## License

MIT
