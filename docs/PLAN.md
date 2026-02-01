# Implementation Plan: Agentic RAG System

> This document outlines the complete implementation strategy for the agentic RAG system.
> Each phase will be committed separately for clear tracking.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Design Decisions](#design-decisions)
3. [Implementation Phases](#implementation-phases)
4. [Commit Strategy](#commit-strategy)
5. [Important Warnings](#important-warnings)

---

## Architecture Overview

### System Components

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              DOCKER COMPOSE                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐                │
│  │   OpenWebUI  │────▶│   Backend    │────▶│   Ollama     │                │
│  │   (Chat UI)  │     │   (FastAPI)  │     │   (LLM)      │                │
│  │   :3000      │     │   :8000      │     │   :11434     │                │
│  └──────────────┘     └──────┬───────┘     └──────────────┘                │
│                              │                                              │
│         ┌────────────────────┼────────────────────┐                        │
│         │                    │                    │                        │
│         │ queries            │ traces             │ prompts                │
│         ▼                    ▼                    ▼                        │
│  ┌──────────────┐     ┌──────────────┐                                     │
│  │  PostgreSQL  │     │   Phoenix    │                                     │
│  │  + pgvector  │     │ (Observ.)    │                                     │
│  │   :5432      │     │   :6006      │                                     │
│  └──────────────┘     └──────────────┘                                     │
│                                                                             │
│  Backend → PostgreSQL: vector storage, memory, documents                   │
│  Backend → Phoenix: OpenTelemetry/OpenInference traces + prompt fetch      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Data Flow: Indexing

```
┌─────────┐    ┌─────────┐    ┌───────────────┐    ┌──────────┐    ┌─────────┐
│  PDF    │───▶│ Docling │───▶│  Contextual   │───▶│ Ollama   │───▶│pgvector │
│  Files  │    │ Parser  │    │   Chunking    │    │ Embedder │    │  Store  │
└─────────┘    └─────────┘    └───────────────┘    └──────────┘    └─────────┘
                                     │
                                     ▼
                              Add document context
                              to each chunk (Anthropic-style)
```

### Data Flow: Query Processing

```
┌────────────┐
│  User      │
│  Query     │
└─────┬──────┘
      │
      ▼
┌─────────────────────────────────────────────────────────────────┐
│                        CrewAI Orchestration                      │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────┐  │
│  │  Planner    │───▶│  Retriever  │───▶│    Synthesizer      │  │
│  │  Agent      │    │  Agent      │    │    Agent            │  │
│  │             │    │             │    │                     │  │
│  │ - Analyze   │    │ - Vector    │    │ - Generate answer   │  │
│  │   query     │    │   search    │    │ - Add citations     │  │
│  │ - Plan      │    │ - Re-rank   │    │ - Format response   │  │
│  │   strategy  │    │ - Filter    │    │                     │  │
│  └─────────────┘    └─────────────┘    └─────────────────────┘  │
│         │                  │                      │              │
│         └──────────────────┴──────────────────────┘              │
│                            │                                     │
│              All calls traced via OpenInference                  │
│                     Backend ──────▶ Phoenix                      │
└─────────────────────────────────────────────────────────────────┘
      │
      ▼
┌────────────────────────────────────────┐
│  Response with Citations               │
│  ---                                   │
│  **Sources:**                          │
│  1. [doc.pdf] Page 5, Chunk 3 (0.92)  │
└────────────────────────────────────────┘
```

---

## Design Decisions

### 1. Chunking Strategy: Contextual Chunking (Anthropic-style)

**What**: Each chunk is prepended with document-level context before embedding.

**Why**: Improves retrieval accuracy by giving each chunk awareness of its source.

**Two Modes**:

#### Fast Mode (Default for Demo)
Metadata-based context prefix — **no LLM calls required**.

```
Context prefix built from:
- Document title (from filename or metadata)
- Section headings (extracted heuristically by Docling)
- Page number
- Short doc synopsis (first paragraph or abstract)

Example:
"Document: Annual Report 2024.pdf | Section: Financial Highlights | Page: 12

Revenue increased by 15%..."
```

#### Slow Mode (Optional)
LLM-generated context summaries per chunk — **cached to DB**.

```
Each chunk sent to LLM with prompt:
"Summarize the context of this chunk in 1-2 sentences."

Result cached so subsequent re-indexing skips LLM calls.
```

**Parameters**:
- Chunk size: 512 tokens
- Overlap: 50 tokens
- Sentence-aware splitting
- Mode: configurable via `--context-mode fast|slow`

---

### 2. Embedding Model: `nomic-embed-text`

**Why chosen**:
- Runs locally via Ollama (100% open source requirement)
- 768 dimensions (good balance of quality/speed)
- Strong performance on retrieval benchmarks
- Supports long context (8192 tokens)

**Alternative**: `mxbai-embed-large` if higher quality needed

---

### 3. LLM Model: `llama3.2:3b` (default)

**Why chosen**:
- Fast inference for interactive chat
- Good instruction following
- Small memory footprint

**Configurable**: Can switch to `llama3.1:8b` or `mistral:7b` via env var

---

### 4. Retrieval Mechanism: Hybrid Search

**Components**:
1. **Semantic search**: HNSW index on embeddings (cosine similarity)
2. **Keyword search**: PostgreSQL full-text search (GIN index)
3. **Fusion**: Reciprocal Rank Fusion (RRF) to combine results

**Parameters**:
- Top-k: 10 candidates
- Final results: 5 after re-ranking

**Pragmatic Fallback**:
> If FTS adds complexity during implementation, we ship **semantic-only first**, 
> then add FTS + RRF as an upgrade. This keeps the demo working while showing 
> awareness of the full solution.

---

### 5. Re-ranking Mechanism: LLM-based Re-ranking

**Approach**: Use Ollama LLM to score relevance of each candidate.

**Why not cross-encoder**:
- Keeps stack 100% Ollama-based
- More flexible for different query types
- Can explain relevance reasoning

**Implementation**:
```python
prompt = f"""Score relevance (0-10) of this passage to the query.
Query: {query}
Passage: {passage}
Score:"""
```

---

### 6. CrewAI Agent Design

#### Agent 1: Query Planner
- **Role**: Understand user intent, plan retrieval strategy
- **Tools**: None (reasoning only)
- **Output**: Search parameters (filters, keywords, intent)

#### Agent 2: Knowledge Retriever  
- **Role**: Execute search, gather relevant chunks
- **Tools**: `vector_search`, `memory_lookup`
- **Output**: Ranked list of relevant passages

#### Agent 3: Response Synthesizer
- **Role**: Generate answer with citations
- **Tools**: None (synthesis only)
- **Output**: Final response with source attribution

---

### 7. Conversation Memory

**Storage**: PostgreSQL table for persistence

**Structure**:
```sql
conversations (
    id UUID,
    session_id VARCHAR,
    role VARCHAR,        -- 'user' | 'assistant'
    content TEXT,
    metadata JSONB,
    created_at TIMESTAMP
)
```

**Context Window**: Last 10 messages included in prompt

**Index**: On `(session_id, created_at)` (already in migrations).

---

### 8. Phoenix Integration

**Tracing** (via OpenInference):
- Auto-instrument with `openinference-instrumentation-llama-index`
- Auto-instrument with `openinference-instrumentation-fastapi`
- Backend emits traces → Phoenix collects via OTLP
- All LLM calls, retrievals, and API requests traced

**Prompt Management**:
- Store prompts in Phoenix at startup with **version tags**
- Development: fetch by name (latest)
- Production: fetch by **version identifier** for stability

```python
# Development (mutable)
prompt = phoenix.get_prompt("retriever_system")

# Production (stable)
prompt = phoenix.get_prompt("retriever_system", version="v1.0.0")
```

**Metrics Captured**:
- Latency per component
- Token usage
- Retrieval scores
- Error rates

---

### 9. OpenWebUI Integration

**Method**: Configure as OpenAI-compatible endpoint

**CRITICAL**: OpenWebUI expects **strict OpenAI API compliance**. No custom JSON structures.

**Setup in OpenWebUI**:
```
Settings → Connections → OpenAI API
Base URL: http://backend:8000/v1
API Key: (any string, we don't validate)
```

**Endpoints exposed** (must match OpenAI spec exactly):
- `GET /v1/models` - List available models
- `POST /v1/chat/completions` - Chat endpoint (strict OpenAI schema)
- `GET /health` - Health check

**Common OpenWebUI Gotchas** (must get these right):

| Field | Requirement |
|-------|-------------|
| `choices[0].message.role` | Must be `"assistant"` |
| `object` | Must be `"chat.completion"` (non-streaming) or `"chat.completion.chunk"` (streaming) |
| `created` | Must be **int** unix timestamp (not string, not float) |
| `model` | Must be included in response |
| `id` | Must be present (e.g., `"chatcmpl-{uuid}"`) |

**Streaming format** (SSE):
```
data: {"id":"chatcmpl-xxx","object":"chat.completion.chunk","created":1234567890,"model":"llama3.2:3b","choices":[{"index":0,"delta":{"role":"assistant","content":"Hello"},"finish_reason":null}]}

data: {"id":"chatcmpl-xxx","object":"chat.completion.chunk","created":1234567890,"model":"llama3.2:3b","choices":[{"index":0,"delta":{"content":" world"},"finish_reason":null}]}

data: {"id":"chatcmpl-xxx","object":"chat.completion.chunk","created":1234567890,"model":"llama3.2:3b","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}

data: [DONE]

```

**Note**: Each `data:` line must end with `\n\n`. Final message is literally `data: [DONE]\n\n`.

---

### 10. Citation Format

**Output Structure**:
```
[Generated answer text...]

---
**Sources:**
1. [Annual Report 2024.pdf] Page 12, Chunk 3 (score: 0.92)
   "Revenue increased by 15% compared to the previous fiscal..."

2. [Q4 Earnings.pdf] Page 5, Chunk 1 (score: 0.87)
   "The company reported strong performance in Q4..."
```

**Metadata stored per chunk**:
- `document_id`: UUID
- `file_name`: Original filename
- `page_number`: Page in source PDF
- `chunk_index`: Position in document
- `section_title`: Extracted heading (if available)

---

## Implementation Phases

### Phase 1: Core Infrastructure
**Files**: `shared/config.py`, `shared/database.py`, `shared/logging.py`, `shared/observability.py`

**Tasks**:
- [ ] Pydantic settings with env var loading
- [ ] Async PostgreSQL connection pool (asyncpg + SQLAlchemy)
- [ ] Structlog JSON logging setup
- [ ] Phoenix tracing initialization (OpenInference)
- [ ] Base schemas (Pydantic models)

**Commit**: `feat: add core infrastructure (config, database, logging, observability)`

---

### Phase 2: Database Migrations
**Files**: `migrations/001_init_extensions.sql`, `migrations/002_create_tables.sql`, `migrations/003_create_indexes.sql`

**Tasks**:
- [ ] Enable pgvector extension
- [ ] Create documents table
- [ ] Create chunks table with vector column
- [ ] Create conversations table
- [ ] HNSW indexes for fast retrieval
- [ ] GIN index for full-text search

**Migration Mechanism**:
Mount SQL files into PostgreSQL's init directory:
```yaml
# docker-compose.yml
postgres:
  image: pgvector/pgvector:pg16
  volumes:
    - ./migrations:/docker-entrypoint-initdb.d:ro
```

PostgreSQL automatically runs `.sql` files in `/docker-entrypoint-initdb.d/`
in alphabetical order on first container start. This is the simplest and most
robust approach for an assessment.

Use `scripts/reset_db.sh` (`down -v`) to force migrations to rerun.

**One command to apply** (fresh setup):
```bash
docker compose up -d postgres
# Migrations run automatically on first start
```

**Commit**: `feat: add database schema and migrations`

**Note**: Must run BEFORE indexer or backend to ensure tables exist.

---

### Phase 3: Document Indexer CLI
**Files**: `indexer/cli.py`, `indexer/parser.py`, `indexer/chunking.py`, `indexer/pipeline.py`

**Tasks**:
- [ ] Docling PDF parser integration
- [ ] Contextual chunking implementation (fast mode default)
- [ ] Embedding generation via Ollama
- [ ] PGVector storage with metadata
- [ ] Typer CLI with progress bars

**Commit**: `feat: implement document indexer with contextual chunking`

---

### Phase 4: RAG Retrieval Layer
**Files**: `backend/rag/retriever.py`, `backend/rag/reranker.py`, `shared/citations.py`

**Tasks**:
- [ ] LlamaIndex VectorStoreIndex setup
- [ ] Semantic search (ship first)
- [ ] Hybrid search with FTS + RRF (upgrade)
- [ ] LLM-based re-ranking
- [ ] Citation extraction and formatting

**Commit**: `feat: implement RAG retrieval with hybrid search and re-ranking`

---

### Phase 5: Conversation Memory
**Files**: `shared/memory.py`, `shared/schemas.py`

**Tasks**:
- [ ] Memory storage in PostgreSQL
- [ ] Session management
- [ ] Context window truncation
- [ ] Memory retrieval for agents

**Commit**: `feat: add conversation memory persistence`

---

### Phase 6: Agentic Layer (CrewAI)
**Files**: `backend/crew/agents.py`, `backend/crew/tools.py`, `backend/crew/runner.py`

**Tasks**:
- [ ] Define agent roles and prompts
- [ ] Implement vector_search tool
- [ ] Implement memory_lookup tool
- [ ] Crew orchestration logic
- [ ] Integration with memory

**Commit**: `feat: implement CrewAI agents with retrieval tools`

---

### Phase 7: Phoenix Prompts
> **Note**: Tracing is in Phase 1, prompts are Phase 7 (separate concerns).

**Files**: `shared/prompts.py`, `config/prompts/`

**Tasks**:
- [ ] Define prompt templates (Jinja2)
- [ ] Phoenix prompt initialization with version tags
- [ ] Prompt retrieval helper (dev: by name, prod: by version)
- [ ] Startup hook to sync prompts

**Commit**: `feat: add Phoenix prompt management with versioning`

---

### Phase 8: FastAPI Backend
**Files**: `backend/main.py`, `backend/api/v1/chat.py`, `backend/api/v1/health.py`

**Tasks**:
- [ ] FastAPI app with lifespan events
- [ ] **Strict OpenAI-compatible** /v1/chat/completions
- [ ] Streaming support (SSE with proper format)
- [ ] Health check with service status
- [ ] CORS for OpenWebUI

**Commit**: `feat: implement FastAPI backend with OpenAI-compatible API`

---

### Phase 9: Docker Deployment
**Files**: `docker-compose.yml`, `docker/backend.Dockerfile`, `.env.example`

**Tasks**:
- [ ] Multi-stage Dockerfile for backend
- [ ] Docker Compose with all services
- [ ] Environment variable configuration
- [ ] Health check configurations
- [ ] Volume mounts for persistence

**Commit**: `feat: add Docker deployment configuration`

---

### Phase 10: RAG Evaluator CLI
**Files**: `evaluator/cli.py`, `evaluator/generation.py`, `evaluator/metrics.py`

**Tasks**:
- [ ] Synthetic test set generation
- [ ] RAGAS metrics implementation
- [ ] Evaluation report generation
- [ ] Typer CLI with options

**Commit**: `feat: implement RAG evaluator with RAGAS metrics`

---

### Phase 11: Testing & Documentation
**Files**: `tests/`, `README.md`, `docs/setup_guide.md`

**Tasks**:
- [ ] Unit tests for core functions
- [ ] Integration tests for API
- [ ] Complete README with setup instructions
- [ ] Architecture documentation

**Commit**: `docs: add tests and documentation`

---

## Commit Strategy

Each phase = 1 focused commit. This makes review easy and shows clear progress.

```
main
  │
  └── dev
        │
        ├── feat: add core infrastructure
        ├── feat: add database schema and migrations    ← BEFORE indexer
        ├── feat: implement document indexer
        ├── feat: implement RAG retrieval
        ├── feat: add conversation memory
        ├── feat: implement CrewAI agents
        ├── feat: add Phoenix prompt management
        ├── feat: implement FastAPI backend
        ├── feat: add Docker deployment
        ├── feat: implement RAG evaluator
        └── docs: add tests and documentation
              │
              └── PR to main (final merge)
```

---

## Important Warnings

### 1. Contextual Chunking Modes

**Fast Mode (Default)**: Uses metadata-based context prefix. No LLM calls. Indexes 50-page PDF in ~1 minute.

**Slow Mode (Optional)**: Uses LLM to generate context summaries. Indexes 50-page PDF in 10+ minutes with local Ollama.

**Recommendation**: 
- Use **fast mode** for demos (3-5 page dense PDFs)
- Document the `--context-mode slow` flag for users who want higher quality

---

### 2. OpenWebUI API Compliance

**The Trap**: OpenWebUI is strict. It expects the API to look **exactly** like OpenAI's.

**Critical Fields**:
- `choices[0].message.role` = `"assistant"`
- `object` = `"chat.completion"` or `"chat.completion.chunk"`
- `created` = int unix timestamp
- `model` = must be present
- Streaming: `data: {...}\n\n` format, terminate with `data: [DONE]\n\n`

**Reference schema (non-streaming)**:
```json
{
  "id": "chatcmpl-xxx",
  "object": "chat.completion",
  "created": 1234567890,
  "model": "llama3.2:3b",
  "choices": [{
    "index": 0,
    "message": {
      "role": "assistant",
      "content": "Response text..."
    },
    "finish_reason": "stop"
  }],
  "usage": {
    "prompt_tokens": 10,
    "completion_tokens": 20,
    "total_tokens": 30
  }
}
```

---

### 3. Phoenix Prompt Versioning

**The Trap**: `phoenix.get_prompt("name")` returns latest version, which can shift unexpectedly.

**The Fix**:
- Development: fetch by name (convenient)
- Production/Demo: fetch by version tag
- Tag prompts on Phoenix initialization: `v1.0.0`, `v1.0.1`, etc.

---

### 4. Database Must Exist Before Code Runs

**The Trap**: Running indexer or backend before migrations = crashes.

**The Fix**:
- Phase 2 is migrations (before Phase 3 indexer)
- SQL files mounted to `/docker-entrypoint-initdb.d/`
- PostgreSQL runs them automatically on first start
- `docker compose up -d postgres` → migrations applied

---

## File-to-Requirement Mapping

| Requirement | Files |
|------------|-------|
| Docling | `indexer/parser.py` |
| LlamaIndex + PGVector | `backend/rag/retriever.py`, `shared/database.py` |
| Contextual RAG | `indexer/chunking.py` |
| Re-ranking | `backend/rag/reranker.py` |
| Conversation Memory | `shared/memory.py` |
| Citations | `shared/citations.py` |
| Ollama | `shared/llm_factory.py` |
| CrewAI | `backend/crew/agents.py`, `backend/crew/runner.py` |
| Phoenix (tracing) | `shared/observability.py` |
| Phoenix (prompts) | `shared/prompts.py` |
| RAGAS | `evaluator/metrics.py` |
| OpenWebUI | `backend/api/v1/chat.py`, `docker-compose.yml` |

---

## Next Steps

1. **Review this plan** - confirm design decisions
2. **Confirm phase order** - migrations before indexer
3. **Start Phase 1** - Core Infrastructure

**Ready to proceed?**
