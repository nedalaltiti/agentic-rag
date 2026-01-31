# Agentic RAG (OpenWebUI + pgvector + Ollama)

This repo is a small, end-to-end **agentic RAG** system built for a case study: ingest PDFs, store embeddings in **Postgres/pgvector**, answer questions through an **OpenAI-compatible** API so it plugs into **OpenWebUI**, and keep the whole thing observable in **Arize Phoenix**.

It’s intentionally “boring” in the good way: reproducible Docker setup, clear module boundaries (indexer / backend / evaluator), and enough instrumentation to debug retrieval and prompt issues.

## What’s included

* **Indexer CLI** (Docling → chunk → embed → store in pgvector)
* **Backend API** (FastAPI + `/v1/chat` + agent tools)
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

### Index documents

Put PDFs in `data/raw/` then:

```bash
agentic-index --input-dir data/raw/ --chunk-size 512 --overlap 50
```

### Chat (OpenWebUI)

* OpenWebUI: `http://localhost:3000`
* Backend API: `http://localhost:8000`
* Ollama: `http://localhost:11434`

The backend exposes:

* `GET /v1/models`
* `POST /v1/chat`

### Traces (Phoenix)

Phoenix UI: `http://localhost:6006`

What I usually look at:

* which chunks were retrieved (and their scores)
* tool call sequence (retriever → rerank → final response)
* prompt inputs/outputs when the answer looks off

## Citation format

Each response includes a Sources block derived from chunk metadata:

* `document_name`
* `page_number` (when available)
* `chunk_index`
* `score`
* short preview

Example:

```
…answer…

Sources:
1) Annual Report 2024.pdf — p.12 — chunk 3 — score 0.92
2) Q4 Earnings.pdf — p.5 — chunk 1 — score 0.87
```

## Services

|     Service |  Port | Notes                            |
| ----------: | :---: | -------------------------------- |
| Backend API |  8000 | FastAPI (`/v1/chat`) |
|   OpenWebUI |  3000 | Chat frontend                    |
|  PostgreSQL |  5432 | pgvector store                   |
|      Ollama | 11434 | local LLM + embeddings           |
|     Phoenix |  6006 | tracing dashboard                |

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

## Development

```bash
pip install -e ".[dev,eval]"
ruff check src/ tests/
pytest -v
mypy src/agentic_rag
```

## License

MIT
