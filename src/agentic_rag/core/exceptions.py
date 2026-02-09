"""Custom exceptions for the agentic RAG application."""


class AgenticRAGError(Exception):
    """Base exception for all agentic RAG errors."""

    def __init__(self, message: str, details: dict | None = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)


class ConfigError(AgenticRAGError):
    """Raised when configuration is invalid or missing."""

    pass


class DependencyUnavailable(AgenticRAGError):
    """Raised when a required external service is unavailable."""

    def __init__(self, service: str, message: str, details: dict | None = None):
        self.service = service
        super().__init__(f"{service} unavailable: {message}", details)


class VectorStoreError(AgenticRAGError):
    """Raised when vector store operations fail."""

    pass


class EmbeddingError(AgenticRAGError):
    """Raised when embedding generation fails."""

    pass


class LLMError(AgenticRAGError):
    """Raised when LLM inference fails."""

    pass


class DocumentParsingError(AgenticRAGError):
    """Raised when document parsing fails."""

    def __init__(self, file_name: str, message: str, details: dict | None = None):
        self.file_name = file_name
        super().__init__(f"Failed to parse '{file_name}': {message}", details)


class RetrievalError(AgenticRAGError):
    """Raised when document retrieval fails."""

    pass


class IndexMismatchError(AgenticRAGError):
    """Raised when the configured embedding/index does not match stored data."""

    pass


class AgentError(AgenticRAGError):
    """Raised when CrewAI agent execution fails."""

    def __init__(self, agent_name: str, message: str, details: dict | None = None):
        self.agent_name = agent_name
        super().__init__(f"Agent '{agent_name}' failed: {message}", details)
