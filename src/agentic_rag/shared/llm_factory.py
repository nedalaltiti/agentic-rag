"""LLM and Embedding model factory using Ollama.

Provides centralized configuration for LlamaIndex models.
"""

import structlog
from llama_index.core import Settings as LlamaIndexSettings
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama

from .config import settings

logger = structlog.get_logger()


def get_llm() -> Ollama:
    """Return the configured Ollama LLM instance."""
    return Ollama(
        model=settings.LLM_MODEL,
        base_url=settings.OLLAMA_BASE_URL,
        request_timeout=300.0,
        temperature=0.1,
    )


def get_embedding_model() -> OllamaEmbedding:
    """Return the configured Ollama Embedding instance."""
    return OllamaEmbedding(
        model_name=settings.EMBEDDING_MODEL,
        base_url=settings.OLLAMA_BASE_URL,
    )


def configure_global_settings() -> None:
    """
    Set LlamaIndex global defaults.

    Should be called ONCE at application startup.
    """
    logger.info(
        "Configuring LlamaIndex Global Settings",
        llm=settings.LLM_MODEL,
        embed=settings.EMBEDDING_MODEL,
    )

    LlamaIndexSettings.llm = get_llm()
    LlamaIndexSettings.embed_model = get_embedding_model()
    LlamaIndexSettings.chunk_size = settings.CHUNK_SIZE
    LlamaIndexSettings.chunk_overlap = settings.CHUNK_OVERLAP
