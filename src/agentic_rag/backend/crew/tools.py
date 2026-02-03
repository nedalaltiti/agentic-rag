"""CrewAI tools for vector search and conversation memory."""

import asyncio

import structlog
from crewai.tools import BaseTool
from pydantic import Field, PrivateAttr

from agentic_rag.backend.rag.reranker import LLMReranker
from agentic_rag.backend.rag.retriever import HybridRetriever
from agentic_rag.core.citations import format_citations
from agentic_rag.core.memory import ConversationMemory
from agentic_rag.core.schemas import Citation

logger = structlog.get_logger()


def run_async_safely(async_fn):
    """Execute an async function from a sync CrewAI tool context."""
    try:
        import anyio

        try:
            return anyio.from_thread.run(async_fn)
        except (RuntimeError, LookupError):
            return asyncio.run(async_fn())
    except ImportError:
        return asyncio.run(async_fn())


class DatabaseSearchTool(BaseTool):
    """Search the knowledge base using hybrid retrieval and LLM re-ranking."""

    name: str = "Vector Database Search"
    description: str = (
        "Search the knowledge base for documents. "
        "Input should be a specific, natural language query string."
    )

    _retriever: HybridRetriever = PrivateAttr(default_factory=HybridRetriever)
    _reranker: LLMReranker = PrivateAttr(default_factory=LLMReranker)
    _last_citations: list[Citation] = PrivateAttr(default_factory=list)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._reranker._semaphore = asyncio.Semaphore(1)

    def get_last_citations(self) -> list[Citation]:
        """Return the most recent citations produced by this tool."""
        return list(self._last_citations)

    def _run(self, query: str) -> str:
        try:
            logger.info("DatabaseSearchTool invoked", query=query)

            async def _execute():
                nodes = await self._retriever.aretrieve(query)
                reranked = await self._reranker.rerank(query, nodes)
                return format_citations(reranked)

            citations = run_async_safely(_execute)
            self._last_citations = citations

            if not citations:
                logger.info("DatabaseSearchTool found no citations")
                return "No relevant PDPL information found."

            logger.info("DatabaseSearchTool citations", count=len(citations))
            result_text = ""
            for i, cit in enumerate(citations[:8]):
                snippet = cit.chunk_text[:1000] + ("..." if len(cit.chunk_text) > 1000 else "")
                page_str = f"Page {cit.page_number}" if cit.page_number is not None else "Page ?"

                section_str = f"Section: {cit.section_path}\n" if cit.section_path else ""

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
