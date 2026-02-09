"""Tests for agentic_rag.backend.api.v1.chat helpers and endpoint."""

from unittest.mock import AsyncMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from agentic_rag.backend.api.v1.chat import (
    _generate_followup_questions,
    _get_session_id,
    _should_use_agent_mode,
    _stream_with_thinking,
    router,
)
from agentic_rag.backend.api.v1.chat_service import (
    RagPayload,
    _format_context_for_llm,
    _is_conversational,
    _is_openwebui_internal_request,
    _route_decision,
)
from agentic_rag.core.config import settings
from agentic_rag.core.schemas import OpenAIChatMessage


class TestIsConversational:
    @pytest.mark.parametrize("msg", ["hello", "Hi", "hey!", "thanks", "bye", "ok"])
    def test_greetings_and_farewells(self, msg):
        assert _is_conversational(msg) is True

    @pytest.mark.parametrize(
        "msg",
        [
            "What is the PDPL?",
            "Explain data processing requirements",
            "How does PDPL affect my company?",
        ],
    )
    def test_real_queries(self, msg):
        assert _is_conversational(msg) is False

    def test_short_strings(self):
        assert _is_conversational("hi") is True
        assert _is_conversational("ok") is True
        assert _is_conversational("ab") is False
        assert _is_conversational("DPO") is False


class TestOpenWebuiInternalRequest:
    def test_title_request(self):
        query = "### Task: Generate a title with emoji for this conversation"
        is_internal, resp = _is_openwebui_internal_request(query)
        assert is_internal is True
        assert "title" in resp

    @pytest.mark.asyncio
    async def test_generate_followup_default(self):
        """No assistant message → returns default questions."""
        messages = [OpenAIChatMessage(role="user", content="hello")]
        result = await _generate_followup_questions(messages)
        import json

        parsed = json.loads(result)
        assert "questions" in parsed
        assert len(parsed["questions"]) == 3

    def test_normal_query(self):
        is_internal, resp = _is_openwebui_internal_request("What is PDPL?")
        assert is_internal is False
        assert resp == ""


class TestShouldUseAgentMode:
    def test_agent_mode_header(self):
        assert _should_use_agent_mode("What is PDPL?", "true") is True

    def test_agent_mode_keyword(self):
        assert _should_use_agent_mode("compare PDPL with GDPR", None) is True

    def test_internal_request_skip(self):
        assert _should_use_agent_mode("### Task: generate title", None) is False


class TestRouteDecision:
    @pytest.mark.asyncio
    async def test_internal_request(self):
        route = await _route_decision("### Task: generate title", "sess-1", False)
        assert route.kind == "internal"
        assert route.internal_response
        assert route.memory is None

    @pytest.mark.asyncio
    @patch("agentic_rag.backend.api.v1.chat_service.ConversationMemory")
    async def test_conversational(self, mock_memory):
        mem_instance = mock_memory.return_value
        mem_instance.add_message = AsyncMock()

        route = await _route_decision("hello", "sess-1", False)

        assert route.kind == "conversational"
        assert route.memory is mem_instance
        mem_instance.add_message.assert_awaited_once_with("user", "hello")

    @pytest.mark.asyncio
    @patch("agentic_rag.backend.api.v1.chat_service.ScopeGate.is_in_scope", new_callable=AsyncMock)
    @patch("agentic_rag.backend.api.v1.chat_service.ConversationMemory")
    async def test_scope_refusal(self, mock_memory, mock_scope):
        mem_instance = mock_memory.return_value
        mem_instance.add_message = AsyncMock()
        mock_scope.return_value = (False, 0.2)

        route = await _route_decision("What is PDPL?", "sess-1", False)

        assert route.kind == "scope_refusal"
        assert route.memory is mem_instance

    @pytest.mark.asyncio
    @patch("agentic_rag.backend.api.v1.chat_service.ScopeGate.is_in_scope", new_callable=AsyncMock)
    @patch("agentic_rag.backend.api.v1.chat_service.ConversationMemory")
    async def test_scope_error_fails_closed(self, mock_memory, mock_scope):
        mem_instance = mock_memory.return_value
        mem_instance.add_message = AsyncMock()
        mock_scope.side_effect = RuntimeError("scope gate down")

        route = await _route_decision("What is PDPL?", "sess-1", False)

        assert route.kind == "scope_refusal"
        assert route.memory is mem_instance

    @pytest.mark.asyncio
    @patch("agentic_rag.backend.api.v1.chat_service.ScopeGate.is_in_scope", new_callable=AsyncMock)
    @patch("agentic_rag.backend.api.v1.chat_service.ConversationMemory")
    async def test_agent_mode(self, mock_memory, mock_scope):
        mem_instance = mock_memory.return_value
        mem_instance.add_message = AsyncMock()
        mock_scope.return_value = (True, 0.9)

        route = await _route_decision("What is PDPL?", "sess-1", True)

        assert route.kind == "agent"
        assert route.memory is mem_instance

    @pytest.mark.asyncio
    @patch("agentic_rag.backend.api.v1.chat_service.ScopeGate.is_in_scope", new_callable=AsyncMock)
    @patch("agentic_rag.backend.api.v1.chat_service.ConversationMemory")
    async def test_rag_route(self, mock_memory, mock_scope):
        mem_instance = mock_memory.return_value
        mem_instance.add_message = AsyncMock()
        mock_scope.return_value = (True, 0.9)

        route = await _route_decision("What is PDPL?", "sess-1", False)

        assert route.kind == "rag"
        assert route.memory is mem_instance


class TestFormatContextForLLM:
    def test_with_citations(self, sample_citations):
        result = _format_context_for_llm(sample_citations)
        assert "Source [1]:" in result
        assert "Source [2]:" in result
        assert "PDPL applies" in result

    def test_empty(self):
        result = _format_context_for_llm([])
        assert "No relevant information found" in result


class TestGetSessionId:
    def test_header_priority(self):
        messages = [OpenAIChatMessage(role="user", content="hello")]
        sid = _get_session_id(messages, session_header="custom-session-123")
        assert sid == "custom-session-123"

    def test_falls_back_to_uuid(self):
        messages = [OpenAIChatMessage(role="user", content="hello")]
        sid1 = _get_session_id(messages)
        sid2 = _get_session_id(messages)
        assert sid1 != sid2
        assert len(sid1) == 16


class TestChatCompletionsEndpoint:
    @pytest.fixture()
    def client(self):
        """Build a TestClient with a minimal app to avoid startup side effects."""
        test_app = FastAPI()
        test_app.include_router(router)
        return TestClient(test_app)

    @patch(
        "agentic_rag.backend.api.v1.chat._process_query",
        new_callable=AsyncMock,
        return_value=(
            "PDPL is a data protection law.",
            [],
            {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        ),
    )
    def test_chat_completions_ok(self, mock_pq, client):
        resp = client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "What is PDPL?"}],
                "stream": False,
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["choices"][0]["message"]["content"] == "PDPL is a data protection law."
        mock_pq.assert_awaited_once()

    def test_chat_completions_empty_messages(self, client):
        resp = client.post(
            "/v1/chat/completions",
            json={"messages": [], "stream": False},
        )
        assert resp.status_code == 400

    @patch(
        "agentic_rag.backend.api.v1.chat._process_query",
        new_callable=AsyncMock,
        return_value=(
            "PDPL is a data protection law.",
            [],
            {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        ),
    )
    def test_model_respected_in_response(self, mock_pq, client):
        """request.model is reflected in the response."""
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": "What is PDPL?"}],
                "stream": False,
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["model"] == "gpt-4o"


class TestShortDomainQueries:
    """Short domain-relevant queries must not bypass RAG."""

    @pytest.mark.parametrize("query", ["DPO", "law", "act", "SCC"])
    def test_short_domain_terms_not_conversational(self, query):
        assert _is_conversational(query) is False

    @pytest.mark.parametrize("query", ["hi", "ok", "yo", "bye", "cool"])
    def test_short_greetings_still_conversational(self, query):
        assert _is_conversational(query) is True

    @pytest.mark.parametrize("query", ["hellooo", "heyyy", "hiiii", "byeee"])
    def test_repeated_chars_still_conversational(self, query):
        assert _is_conversational(query) is True


class TestStreamingCitationsAfterStop:
    """Citations must arrive before finish_reason='stop'."""

    @pytest.mark.asyncio
    @patch("agentic_rag.backend.api.v1.chat_service.ConversationMemory")
    @patch("agentic_rag.backend.api.v1.chat_service.ScopeGate.is_in_scope", new_callable=AsyncMock)
    @patch("agentic_rag.backend.api.v1.streaming._prepare_rag", new_callable=AsyncMock)
    @patch("agentic_rag.backend.api.v1.streaming.ollama_chat_stream")
    async def test_citations_before_stop(
        self,
        mock_stream,
        mock_prepare,
        mock_scope,
        mock_memory,
        sample_citations,
        monkeypatch,
    ):
        monkeypatch.setattr(settings, "SEMANTIC_CACHE_ENABLED", False)
        mock_scope.return_value = (True, 0.8)
        mock_prepare.return_value = RagPayload(
            citations=sample_citations,
            history=[],
            context="ctx",
            system_prompt="sys",
            user_prompt="usr",
        )

        async def fake_stream(*args, **kwargs):
            yield {
                "thinking": None,
                "content": "PDPL applies to data [1] and officers [2].",
                "done": False,
            }
            yield {"thinking": None, "content": None, "done": True}

        mock_stream.return_value = fake_stream()
        mem_instance = mock_memory.return_value
        mem_instance.add_message = AsyncMock()
        mem_instance.get_history = AsyncMock(return_value=[])

        chunks = []
        async for chunk in _stream_with_thinking(
            "test-id", "qwen3:1.7b", "What is PDPL?", "sess-1", False, 1234567890
        ):
            chunks.append(chunk)

        # Find positions of citations and stop
        citations_idx = None
        stop_idx = None
        for i, c in enumerate(chunks):
            if '"citations"' in c and '"document_id"' in c:
                citations_idx = i
            if '"stop"' in c and '"finish_reason"' in c:
                stop_idx = i

        assert citations_idx is not None, "Citations chunk not found"
        assert stop_idx is not None, "Stop chunk not found"
        assert citations_idx < stop_idx, "Citations must come before finish_reason=stop"


class TestStreamingFirstChunk:
    """Ensure the first RAG status chunk is emitted before retrieval begins."""

    @pytest.mark.asyncio
    @patch("agentic_rag.backend.api.v1.chat_service.ConversationMemory")
    @patch("agentic_rag.backend.api.v1.chat_service.ScopeGate.is_in_scope", new_callable=AsyncMock)
    @patch("agentic_rag.backend.api.v1.streaming._prepare_rag", new_callable=AsyncMock)
    async def test_search_chunk_before_prepare(
        self, mock_prepare, mock_scope, mock_memory, sample_citations, monkeypatch
    ):
        monkeypatch.setattr(settings, "SEMANTIC_CACHE_ENABLED", False)
        mock_scope.return_value = (True, 0.8)
        mock_prepare.return_value = RagPayload(
            citations=sample_citations,
            history=[],
            context="ctx",
            system_prompt="sys",
            user_prompt="usr",
        )
        mem_instance = mock_memory.return_value
        mem_instance.add_message = AsyncMock()

        gen = _stream_with_thinking(
            "test-id", "qwen3:1.7b", "What is PDPL?", "sess-1", False, 1234567890
        )

        first = await gen.__anext__()
        second = await gen.__anext__()

        assert '"role"' in first
        assert "<think>Searching documents..." in second
        assert mock_prepare.await_count == 0

        await gen.aclose()


class TestStreamingErrorClosesThinkTag:
    """If streaming errors after <think> opens, the tag must still close."""

    @pytest.mark.asyncio
    @patch("agentic_rag.backend.api.v1.chat_service.ConversationMemory")
    @patch("agentic_rag.backend.api.v1.chat_service.ScopeGate.is_in_scope", new_callable=AsyncMock)
    @patch("agentic_rag.backend.api.v1.streaming._prepare_rag", new_callable=AsyncMock)
    @patch("agentic_rag.backend.api.v1.streaming.ollama_chat_stream")
    async def test_think_tag_closed_on_error(
        self, mock_stream, mock_prepare, mock_scope, mock_memory, monkeypatch
    ):
        monkeypatch.setattr(settings, "SEMANTIC_CACHE_ENABLED", False)
        mock_scope.return_value = (True, 0.8)
        mock_prepare.return_value = RagPayload(
            citations=[],
            history=[],
            context="ctx",
            system_prompt="sys",
            user_prompt="usr",
        )

        async def exploding_stream(*args, **kwargs):
            yield {"thinking": "reasoning...", "content": None, "done": False}
            raise ConnectionError("Ollama crashed")

        mock_stream.return_value = exploding_stream()
        mem_instance = mock_memory.return_value
        mem_instance.add_message = AsyncMock()
        mem_instance.get_history = AsyncMock(return_value=[])

        raw = ""
        async for chunk in _stream_with_thinking(
            "test-id", "qwen3:1.7b", "What is PDPL?", "sess-1", False, 1234567890
        ):
            raw += chunk

        # <think> was opened by status message, must be closed despite error
        assert raw.count("<think>") == raw.count("</think>"), (
            "Unclosed <think> tag after streaming error"
        )
