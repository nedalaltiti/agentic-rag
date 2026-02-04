"""SQLAlchemy ORM models for documents, chunks, and conversations."""

import uuid
from datetime import datetime
from typing import Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import DateTime, ForeignKey, Integer, String, Text, func, text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .config import settings
from .database import Base


class Document(Base):
    """Document metadata and file information."""

    __tablename__ = "documents"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    file_name: Mapped[str] = mapped_column(String(255), nullable=False)
    file_path: Mapped[str | None] = mapped_column(String(1024))

    # Partial unique index handles constraints; removed unique=True to match DB
    file_hash: Mapped[str | None] = mapped_column(String(64))

    # Index versioning metadata
    index_version: Mapped[str] = mapped_column(
        String(32), nullable=False, default=settings.INDEX_VERSION
    )
    embedding_model: Mapped[str] = mapped_column(
        String(255), nullable=False, default=settings.EMBEDDING_MODEL
    )
    embedding_dimension: Mapped[int] = mapped_column(
        Integer, nullable=False, default=settings.EMBEDDING_DIMENSION
    )

    page_count: Mapped[int | None] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )

    chunks: Mapped[list["Chunk"]] = relationship(
        "Chunk", back_populates="document", cascade="all, delete-orphan"
    )


class Chunk(Base):
    """Document chunk with vector embedding and contextual content."""

    __tablename__ = "chunks"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), nullable=False
    )

    content: Mapped[str] = mapped_column(Text, nullable=False)
    contextual_content: Mapped[str | None] = mapped_column(Text)
    chunk_hash: Mapped[str | None] = mapped_column(String(64))

    # Typed metadata
    metadata_: Mapped[dict[str, Any]] = mapped_column(
        "metadata", JSONB, server_default=text("'{}'::jsonb"), nullable=False
    )

    embedding: Mapped[Any] = mapped_column(Vector(settings.EMBEDDING_DIMENSION), nullable=False)

    # Index versioning metadata
    index_version: Mapped[str] = mapped_column(
        String(32), nullable=False, default=settings.INDEX_VERSION
    )
    embedding_model: Mapped[str] = mapped_column(
        String(255), nullable=False, default=settings.EMBEDDING_MODEL
    )
    embedding_dimension: Mapped[int] = mapped_column(
        Integer, nullable=False, default=settings.EMBEDDING_DIMENSION
    )

    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    document: Mapped["Document"] = relationship("Document", back_populates="chunks")


class Conversation(Base):
    """Conversation history for agent memory."""

    __tablename__ = "conversations"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id: Mapped[str] = mapped_column(String(255), nullable=False)
    role: Mapped[str] = mapped_column(String(50), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    metadata_: Mapped[dict[str, Any]] = mapped_column(
        "metadata", JSONB, server_default=text("'{}'::jsonb"), nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class SemanticCache(Base):
    """Semantic cache for answers and citations."""

    __tablename__ = "semantic_cache"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    query_text: Mapped[str] = mapped_column(Text, nullable=False)
    query_embedding: Mapped[Any] = mapped_column(
        Vector(settings.EMBEDDING_DIMENSION), nullable=False
    )
    answer: Mapped[str] = mapped_column(Text, nullable=False)
    citations: Mapped[Any] = mapped_column(
        JSONB, server_default=text("'[]'::jsonb"), nullable=False
    )
    embedding_model: Mapped[str] = mapped_column(String(255), nullable=False)
    embedding_dimension: Mapped[int] = mapped_column(Integer, nullable=False)
    index_version: Mapped[str] = mapped_column(String(32), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
