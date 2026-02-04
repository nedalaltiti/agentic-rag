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
    PAGE_NUMBER,
    SECTION_TITLE,
)
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
    "COLLECTION_NAME",
    "AGENT_PLANNER",
    "AGENT_RETRIEVER",
    "AGENT_SYNTHESIZER",
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
    # Schemas
    "ChatMessage",
    "Citation",
    "TokenUsage",
    "AgentResponse",
]
