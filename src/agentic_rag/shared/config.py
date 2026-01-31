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

    # API Config
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000

    # Database (Postgres + PGVector)
    # Typed as str to avoid Pydantic validation issues with 'postgresql+asyncpg' scheme
    DATABASE_URL: str = Field(
        default="postgresql+asyncpg://postgres:postgres@postgres:5432/ragdb"
    )
    DB_POOL_SIZE: int = 20
    DB_MAX_OVERFLOW: int = 10

    # Observability (Arize Phoenix)
    PHOENIX_COLLECTOR_ENDPOINT: str = "http://phoenix:6006/v1/traces"
    # Phoenix REST API base URL (prompt management)
    PHOENIX_API_URL: str = "http://phoenix:6006"
    PHOENIX_PROJECT_NAME: str = "agentic-rag-v1"

    # Prompt management controls
    PHOENIX_PROMPT_SYNC: bool = True  # disable during hot reload if you want
    # If set, registry will prefer this tag in prod (e.g., "production" or "v1.0.0")
    PHOENIX_PROMPT_TAG: str = Field(default_factory=lambda: "production" if Settings().ENVIRONMENT == "prod" else "development")

    # LLM & Embedding (Ollama)
    OLLAMA_BASE_URL: str = "http://ollama:11434"
    LLM_MODEL: str = "qwen3:1.7b"
    EMBEDDING_MODEL: str = "qwen3-embedding:0.6b"

    # Contextual RAG Settings
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 50
    TOP_K_RETRIEVAL: int = 10
    TOP_K_RERANK: int = 5

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
