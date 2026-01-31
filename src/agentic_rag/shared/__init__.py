"""Shared utilities and configuration for the agentic RAG system."""

from .config import settings
from .constants import (
    AGENT_PLANNER,
    AGENT_RETRIEVER,
    AGENT_SYNTHESIZER,
    CHUNK_INDEX,
    COLLECTION_NAME,
    CONTEXT_PREFIX,
    CREATED_AT,
    DOC_HASH,
    DOC_ID,
    DOC_NAME,
    EMBEDDING_DIMENSION,
    PAGE_NUMBER,
    SECTION_TITLE,
)
from .database import AsyncSessionLocal, Base, engine, get_db
from .exceptions import (
    AgentError,
    AgenticRAGError,
    ConfigError,
    DependencyUnavailable,
    DocumentParsingError,
    EmbeddingError,
    LLMError,
    RetrievalError,
    VectorStoreError,
)
from .health import (
    check_all_services,
    check_database,
    check_ollama,
    check_phoenix,
    get_overall_status,
)
from .llm_factory import configure_global_settings, get_embedding_model, get_llm
from .logging import setup_logging
from .models import Chunk, Conversation, Document
from .observability import setup_observability
from .schemas import AgentResponse, ChatMessage, Citation, TokenUsage

__all__ = [
    # Config
    "settings",
    # Constants
    "DOC_ID",
    "DOC_NAME",
    "DOC_HASH",
    "PAGE_NUMBER",
    "CHUNK_INDEX",
    "SECTION_TITLE",
    "CREATED_AT",
    "CONTEXT_PREFIX",
    "EMBEDDING_DIMENSION",
    "COLLECTION_NAME",
    "AGENT_PLANNER",
    "AGENT_RETRIEVER",
    "AGENT_SYNTHESIZER",
    # Database
    "engine",
    "AsyncSessionLocal",
    "Base",
    "get_db",
    # Models
    "Document",
    "Chunk",
    "Conversation",
    # Exceptions
    "AgenticRAGError",
    "ConfigError",
    "DependencyUnavailable",
    "VectorStoreError",
    "EmbeddingError",
    "LLMError",
    "DocumentParsingError",
    "RetrievalError",
    "AgentError",
    # Health Checks
    "check_database",
    "check_ollama",
    "check_phoenix",
    "check_all_services",
    "get_overall_status",
    # Logging
    "setup_logging",
    # Observability
    "setup_observability",
    # LLM
    "get_llm",
    "get_embedding_model",
    "configure_global_settings",
    # Schemas
    "ChatMessage",
    "Citation",
    "TokenUsage",
    "AgentResponse",
]
