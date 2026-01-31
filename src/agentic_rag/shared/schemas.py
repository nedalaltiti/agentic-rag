"""Pydantic schemas for API and internal data models.

These schemas are used for request/response validation and serialization.
"""

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import UUID4, BaseModel, Field


class ChatMessage(BaseModel):
    """A single message in a chat conversation."""

    role: Literal["user", "assistant", "system"]
    content: str


class Citation(BaseModel):
    """Citation metadata for a retrieved chunk."""

    document_id: UUID4
    chunk_id: UUID4
    file_name: str
    page_number: Optional[int] = None
    section_path: Optional[str] = None
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
    citations: List[Citation] = Field(default_factory=list)
    trace_id: Optional[str] = None
    usage: TokenUsage = Field(default_factory=TokenUsage)
