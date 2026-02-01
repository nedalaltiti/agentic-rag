"""Pydantic schemas for API and internal data models.

These schemas are used for request/response validation and serialization.
"""

from typing import Literal

from pydantic import UUID4, BaseModel, ConfigDict, Field


class ChatMessage(BaseModel):
    """A single message in a chat conversation."""

    role: Literal["user", "assistant", "system"]
    content: str


class Citation(BaseModel):
    """Citation metadata for a retrieved chunk."""

    document_id: UUID4
    chunk_id: UUID4
    file_name: str
    page_number: int | None = None
    section_path: str | None = None
    chunk_text: str
    score: float


class TokenUsage(BaseModel):
    """Token usage statistics for LLM calls."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class AgentResponse(BaseModel):
    """Response from the agentic RAG pipeline."""

    answer: str
    citations: list[Citation] = Field(default_factory=list)
    trace_id: str | None = None
    usage: TokenUsage = Field(default_factory=TokenUsage)


class OpenAIChatMessage(BaseModel):
    """OpenAI chat message format."""

    role: Literal["user", "assistant", "system"]
    content: str


class OpenAIChatRequest(BaseModel):
    """OpenAI /v1/chat/completions request."""

    model_config = ConfigDict(extra="ignore")

    model: str | None = None  # Optional: default to settings.LLM_MODEL
    messages: list[OpenAIChatMessage]
    stream: bool = False
    temperature: float = 0.7
    max_tokens: int | None = None
    top_p: float = 1.0


class OpenAIChatChoice(BaseModel):
    """A single choice in OpenAI chat response."""

    index: int
    message: OpenAIChatMessage
    finish_reason: str | None = None


class OpenAIChatResponse(BaseModel):
    """OpenAI /v1/chat/completions response."""

    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[OpenAIChatChoice]
    usage: TokenUsage


class OpenAIChatStreamDelta(BaseModel):
    """Delta content for streaming response."""

    role: Literal["assistant"] | None = None
    content: str | None = None


class OpenAIChatStreamChoice(BaseModel):
    """A single choice in streaming response."""

    index: int
    delta: OpenAIChatStreamDelta
    finish_reason: str | None = None


class OpenAIChatStreamChunk(BaseModel):
    """A single chunk in streaming response."""

    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: list[OpenAIChatStreamChoice]


class ModelInfo(BaseModel):
    """Model information for /v1/models response."""

    id: str
    object: str = "model"
    created: int
    owned_by: str = "agentic-rag"


class ModelsListResponse(BaseModel):
    """Response for /v1/models endpoint."""

    object: str = "list"
    data: list[ModelInfo]
