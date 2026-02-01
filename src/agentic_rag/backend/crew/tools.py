"""CrewAI tools for vector search and conversation memory.

Provides sync-safe wrappers around async retrieval and memory backends
using an AnyIO-first bridge with asyncio.run fallback.
"""

import asyncio

from crewai.tools import BaseTool
from pydantic import Field, PrivateAttr
import structlog

from agentic_rag.backend.rag.retriever import HybridRetriever
from agentic_rag.backend.rag.reranker import LLMReranker
from agentic_rag.shared.citations import format_citations
from agentic_rag.shared.memory import ConversationMemory

logger = structlog.get_logger()


# --- Sync Bridge Helper ---


def run_async_safely(async_fn):
    """
    Execute an async function from within a synchronous CrewAI tool.

    Handles being called from a standard thread (via asyncio.to_thread)
    or an AnyIO worker thread.

    Args:
        async_fn: A zero-argument async function (not a coroutine object).
    """
    try:
        import anyio

        try:
            return anyio.from_thread.run(async_fn)
        except (RuntimeError, LookupError):
            return asyncio.run(async_fn())
    except ImportError:
        return asyncio.run(async_fn())


# --- Tool 1: Vector Search ---


class DatabaseSearchTool(BaseTool):
    """Search the knowledge base using hybrid retrieval and LLM re-ranking."""

    name: str = "Vector Database Search"
    description: str = (
        "Search the knowledge base for documents. "
        "Input should be a specific, natural language query string."
    )

    _retriever: HybridRetriever = PrivateAttr(default_factory=HybridRetriever)
    _reranker: LLMReranker = PrivateAttr(default_factory=LLMReranker)

    def _run(self, query: str) -> str:
        try:
            logger.info("Agent executing search", query=query)

            async def _execute():
                nodes = await self._retriever.aretrieve(query)
                # reranked = await self._reranker.rerank(query, nodes)
                return format_citations(nodes[:5])

            citations = run_async_safely(_execute)

            if not citations:
                return "No relevant documents found."

            result_text = ""
            for i, cit in enumerate(citations[:8]):
                snippet = cit.chunk_text[:1000] + (
                    "..." if len(cit.chunk_text) > 1000 else ""
                )
                page_str = (
                    f"Page {cit.page_number}" if cit.page_number is not None else "Page ?"
                )

                section_str = (
                    f"Section: {cit.section_path}\n" if cit.section_path else ""
                )

                result_text += (
                    f"--- Source {i + 1} ---\n"
                    f"ID: {cit.document_id}\n"
                    f"File: {cit.file_name} ({page_str})\n"
                    f"{section_str}"
                    f"Content: {snippet}\n\n"
                )
            return result_text

        except Exception as e:
            logger.error("Search tool failed", error=str(e))
            return f"Error executing search: {str(e)}"


# --- Tool 2: Memory Lookup ---


class MemoryLookupTool(BaseTool):
    """Retrieve previous conversation history for context-aware responses."""

    name: str = "Conversation Memory"
    description: str = (
        "Retrieve previous conversation history. "
        "Input is ignored, it always fetches the last 5 messages."
    )
    session_id: str = Field(..., description="The current session ID")

    def _run(self, query: str = "") -> str:
        try:
            logger.info("Agent accessing memory", session_id=self.session_id)

            async def _execute():
                memory = ConversationMemory(self.session_id)
                return await memory.get_history(limit=5)

            history = run_async_safely(_execute)

            if not history:
                return "No previous conversation history."

            result_text = "Chat History:\n"
            for msg in history:
                result_text += f"{msg.role.value}: {msg.content}\n"
            return result_text

        except Exception as e:
            return f"Error retrieving memory: {str(e)}"
