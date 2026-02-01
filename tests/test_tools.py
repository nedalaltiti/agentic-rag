"""Tests for agentic_rag.backend.crew.tools."""

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

from agentic_rag.backend.crew.tools import (
    DatabaseSearchTool,
    MemoryLookupTool,
    run_async_safely,
)
from agentic_rag.shared.schemas import Citation


class TestRunAsyncSafely:
    def test_returns_result(self):
        async def async_fn():
            return 42

        result = run_async_safely(async_fn)
        assert result == 42


class TestDatabaseSearchTool:
    @patch("agentic_rag.backend.crew.tools.HybridRetriever")
    def test_no_results(self, mock_retriever_cls):
        mock_instance = MagicMock()
        mock_instance.aretrieve = AsyncMock(return_value=[])
        mock_retriever_cls.return_value = mock_instance

        tool = DatabaseSearchTool()
        tool._retriever = mock_instance
        result = tool._run("some query")
        assert result == "No relevant documents found."

    @patch("agentic_rag.backend.crew.tools.format_citations")
    @patch("agentic_rag.backend.crew.tools.HybridRetriever")
    def test_with_results(self, mock_retriever_cls, mock_format):
        mock_retriever = MagicMock()
        mock_retriever.aretrieve = AsyncMock(return_value=["node1", "node2"])
        mock_retriever_cls.return_value = mock_retriever

        mock_reranker = MagicMock()
        mock_reranker.rerank = AsyncMock(return_value=["node1", "node2"])

        cit = Citation(
            document_id=uuid.uuid4(),
            chunk_id=uuid.uuid4(),
            file_name="test.md",
            page_number=1,
            section_path="Intro",
            chunk_text="Sample content for testing.",
            score=0.9,
        )
        mock_format.return_value = [cit]

        tool = DatabaseSearchTool()
        tool._retriever = mock_retriever
        tool._reranker = mock_reranker
        result = tool._run("PDPL query")

        assert "Source 1" in result
        assert "test.md" in result
        assert "Sample content" in result


class TestMemoryLookupTool:
    @patch("agentic_rag.backend.crew.tools.ConversationMemory")
    def test_no_history(self, mock_mem_cls):
        mock_instance = MagicMock()
        mock_instance.get_history = AsyncMock(return_value=[])
        mock_mem_cls.return_value = mock_instance

        tool = MemoryLookupTool(session_id="test-session")
        result = tool._run()
        assert result == "No previous conversation history."
