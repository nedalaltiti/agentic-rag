"""Persistent conversation memory backed by PostgreSQL.

Stores and retrieves session history as LlamaIndex ChatMessage objects,
with optional session injection for transaction control.
"""

from typing import Literal

import structlog
from llama_index.core.llms import ChatMessage, MessageRole
from sqlalchemy import delete, desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from agentic_rag.shared.database import AsyncSessionLocal
from agentic_rag.shared.models import Conversation

logger = structlog.get_logger()


class ConversationMemory:
    """Manages persistent conversation history in PostgreSQL."""

    def __init__(self, session_id: str):
        self.session_id = session_id

    async def add_message(
        self,
        role: Literal["user", "assistant", "system"],
        content: str,
        metadata: dict | None = None,
        session: AsyncSession | None = None,
    ):
        """
        Persists a message.
        Supports external session injection for transaction grouping.
        """
        if session:
            await self._save_internal(session, role, content, metadata)
        else:
            async with AsyncSessionLocal() as local_session:
                await self._save_internal(local_session, role, content, metadata)
                await local_session.commit()

    async def _save_internal(
        self, session: AsyncSession, role: str, content: str, metadata: dict | None
    ):
        msg = Conversation(
            session_id=self.session_id,
            role=role,
            content=content,
            metadata_=metadata or {},
        )
        session.add(msg)

    async def get_history(self, limit: int = 10) -> list[ChatMessage]:
        """
        Retrieves recent messages as LlamaIndex ChatMessage objects.
        Chronological order: oldest -> newest.
        """
        try:
            async with AsyncSessionLocal() as session:
                stmt = (
                    select(Conversation)
                    .where(Conversation.session_id == self.session_id)
                    .order_by(desc(Conversation.created_at))
                    .limit(limit)
                )
                result = await session.execute(stmt)
                rows = result.scalars().all()

                history = []
                for row in reversed(rows):
                    if row.role == "user":
                        role_enum = MessageRole.USER
                    elif row.role == "assistant":
                        role_enum = MessageRole.ASSISTANT
                    else:
                        role_enum = MessageRole.SYSTEM

                    history.append(
                        ChatMessage(
                            role=role_enum,
                            content=row.content,
                            additional_kwargs=row.metadata_ or {},
                        )
                    )

                return history
        except Exception as e:
            logger.error(
                "Failed to retrieve history",
                error=str(e),
                session_id=self.session_id,
            )
            return []

    async def clear(self):
        """Clears memory for this session. Crucial for RAG evaluation."""
        try:
            async with AsyncSessionLocal() as session:
                stmt = delete(Conversation).where(Conversation.session_id == self.session_id)
                await session.execute(stmt)
                await session.commit()
                logger.info("Memory cleared", session_id=self.session_id)
        except Exception as e:
            logger.error("Failed to clear memory", error=str(e))
