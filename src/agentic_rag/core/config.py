"""Application configuration using Pydantic Settings.

Loads configuration from environment variables and .env file.
All settings are validated at startup.
"""

from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings with environment variable support."""

    # App Config
    APP_NAME: str = "Agentic RAG"
    APP_VERSION: str = "0.1.0"
    ENVIRONMENT: Literal["dev", "prod"] = "dev"
    LOG_LEVEL: str = "INFO"

    # Domain configuration
    DOMAIN_NAME: str = "PDPL"
    DOMAIN_FULL_NAME: str = "Personal Data Protection Law"
    DOMAIN_REGION: str = "Saudi Arabia"
    DOMAIN_TOPICS: str = "PDPL, privacy, and data protection"
    DOMAIN_CLOSING: str = (
        "Is there anything else I can help you with regarding PDPL compliance?"
    )

    # API Config
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    FORCE_STREAMING: bool = False

    # Database (Postgres + PGVector)
    # Typed as str to avoid Pydantic validation issues with 'postgresql+asyncpg' scheme
    DATABASE_URL: str = Field(default="postgresql+asyncpg://postgres:postgres@postgres:5432/ragdb")
    DB_POOL_SIZE: int = 20
    DB_MAX_OVERFLOW: int = 10

    # Observability (Arize Phoenix)
    PHOENIX_COLLECTOR_ENDPOINT: str = "http://phoenix:6006/v1/traces"
    # Phoenix REST API base URL (prompt management)
    PHOENIX_API_URL: str = "http://phoenix:6006"
    PHOENIX_PROJECT_NAME: str = "agentic-rag-v1"

    # Prompt management controls
    PHOENIX_PROMPT_SYNC: bool = True
    PHOENIX_PROMPT_TAG: str = "development"  # Use "production" in prod

    # LLM & Embedding (Ollama)
    OLLAMA_BASE_URL: str = "http://ollama:11434"
    LLM_MODEL: str = "qwen3:1.7b"
    EMBEDDING_MODEL: str = "qwen3-embedding:0.6b"
    # Must match the output dimension of EMBEDDING_MODEL and the DB schema
    # (migrations/002_create_tables.sql: vector(1024)).
    EMBEDDING_DIMENSION: int = 1024
    # Index versioning
    INDEX_VERSION: str = "v1"

    # Chunking tokenizer: "default" (LlamaIndex/tiktoken cl100k_base) or
    # "hf:<repo>" to load a local HuggingFace tokenizer (e.g. "hf:Qwen/Qwen3-1.7B").
    # "default" is approximate for non-English text; use "hf:" for exact token sizing.
    TOKENIZER: str = "default"

    # Contextual RAG Settings
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 50
    TOP_K_RETRIEVAL: int = 10
    TOP_K_RERANK: int = 5
    # Optional runtime HNSW tuning (pgvector). None = use DB default.
    # Higher values improve recall but increase latency.
    HNSW_EF_SEARCH: int | None = None
    USE_CREWAI: bool = True
    EVAL_MODEL: str = "qwen3:4b"
    RRF_WEIGHT_VECTOR: float = 1.0
    RRF_WEIGHT_KEYWORD: float = 1.5
    RERANKER_TIMEOUT: float = 30.0
    # Reranker cache (reduces repeat LLM calls on similar queries)
    RERANK_CACHE_TTL: int = 900  # seconds
    RERANK_CACHE_MAX: int = 512
    QUERY_EMBED_CACHE_TTL: int = 900
    # Prompt caching (system prompt + context assembly)
    PROMPT_CACHE_TTL_SECONDS: int = 900
    PROMPT_CACHE_MAX: int = 512
    # Semantic cache (answer + citations)
    SEMANTIC_CACHE_ENABLED: bool = True
    SEMANTIC_CACHE_TTL_SECONDS: int = 7 * 24 * 60 * 60  # 7 days
    SEMANTIC_CACHE_MIN_SIMILARITY: float = 0.92
    # Optional retrieval cutoffs (precision tuning). None disables.
    # VECTOR_MIN_SIMILARITY uses cosine similarity (0-1); higher = stricter.
    VECTOR_MIN_SIMILARITY: float | None = None
    # KEYWORD_MIN_SCORE uses Postgres ts_rank; higher = stricter.
    KEYWORD_MIN_SCORE: float | None = None
    # Evaluation controls
    EVAL_SAMPLE_SIZE: int = 20
    EVAL_RETRIEVAL_K: int = 5
    EVAL_MAX_WORKERS: int = 1
    EVAL_TIMEOUT: int = 600
    EVAL_INTERVAL_SECONDS: int = 3600

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
