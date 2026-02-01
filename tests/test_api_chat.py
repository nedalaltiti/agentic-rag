"""Tests for agentic_rag.backend.api.v1.chat helpers and endpoint."""

from unittest.mock import AsyncMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from agentic_rag.backend.api.v1.chat import (
    _format_context_for_llm,
    _format_sources_section,
    _get_session_id,
    _is_conversational,
    _is_openwebui_internal_request,
    _should_use_agent_mode,
    router,
)
from agentic_rag.shared.schemas import OpenAIChatMessage


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
        assert _is_conversational("ab") is True
        assert _is_conversational("x") is True


class TestOpenWebuiInternalRequest:
    def test_title_request(self):
        query = "### Task: Generate a title with emoji for this conversation"
        is_internal, resp = _is_openwebui_internal_request(query)
        assert is_internal is True
        assert "title" in resp

    def test_followup_request(self):
        query = "### Task: Generate follow-up questions for the conversation"
        is_internal, resp = _is_openwebui_internal_request(query)
        assert is_internal is True
        assert "questions" in resp

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


class TestFormatSourcesSection:
    def test_with_citations(self, sample_citations):
        result = _format_sources_section(sample_citations)
        assert "**Sources:**" in result
        assert "pdpl overview" in result  # + replaced with space, .md stripped

    def test_empty(self):
        assert _format_sources_section([]) == ""


class TestFormatContextForLLM:
    def test_with_citations(self, sample_citations):
        result = _format_context_for_llm(sample_citations)
        assert "Source [1]:" in result
        assert "Source [2]:" in result
        assert "PDPL applies" in result

    def test_empty(self):
        result = _format_context_for_llm([])
        assert "No relevant documents found" in result


class TestGetSessionId:
    def test_header_priority(self):
        messages = [OpenAIChatMessage(role="user", content="hello")]
        sid = _get_session_id(messages, session_header="custom-session-123")
        assert sid == "custom-session-123"

    def test_deterministic_hash(self):
        messages = [OpenAIChatMessage(role="user", content="hello")]
        sid1 = _get_session_id(messages)
        sid2 = _get_session_id(messages)
        assert sid1 == sid2
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
        return_value=("PDPL is a data protection law.", []),
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
