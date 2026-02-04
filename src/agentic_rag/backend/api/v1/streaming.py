"""Streaming response renderer for chat completions."""

from __future__ import annotations

import json
from collections.abc import AsyncGenerator
from typing import Literal

import structlog

from agentic_rag.backend.api.v1.chat_service import (
    CLOSING_LINE,
    SCOPE_REFUSAL,
    RouteDecision,
    _agent_mode_response,
    _fallback_rag_answer,
    _format_sources_footer,
    _prepare_rag,
)
from agentic_rag.core.exceptions import DependencyUnavailable, IndexMismatchError
from agentic_rag.backend.rag.semantic_cache import lookup_cache, store_cache
from agentic_rag.core.llm_factory import ollama_chat_stream
from agentic_rag.core.prompts import PromptRegistry
from agentic_rag.core.schemas import (
    OpenAIChatStreamChoice,
    OpenAIChatStreamChunk,
    OpenAIChatStreamDelta,
)

logger = structlog.get_logger()


class StreamingRenderer:
    """Render streaming SSE responses based on a routing decision."""

    def __init__(self, request_id: str, model: str, created_at: int) -> None:
        self._request_id = request_id
        self._model = model
        self._created_at = created_at

    def _sse(
        self,
        content: str = "",
        role: Literal["assistant"] | None = None,
        finish: str | None = None,
    ) -> str:
        chunk = OpenAIChatStreamChunk(
            id=self._request_id,
            created=self._created_at,
            model=self._model,
            choices=[
                OpenAIChatStreamChoice(
                    index=0,
                    delta=OpenAIChatStreamDelta(
                        role=role,
                        content=content or None,
                    ),
                    finish_reason=finish,
                )
            ],
        )
        return f"data: {chunk.model_dump_json()}\n\n"

    def _sse_stop(self) -> str:
        return self._sse(finish="stop") + "data: [DONE]\n\n"

    async def stream_response(
        self,
        route: RouteDecision,
        query: str,
    ) -> AsyncGenerator[str, None]:
        """Yield SSE chunks for the given routing result."""
        yield self._sse(role="assistant")

        if route.kind == "internal":
            yield self._sse(route.internal_response) + self._sse_stop()
            return

        memory = route.memory
        if memory is None:
            yield self._sse("Sorry, I encountered an error.")
            yield self._sse_stop()
            return

        if route.kind == "conversational":
            conv_system = PromptRegistry.render("conversational_prompt")
            conv_answer = ""
            try:
                async for chunk in ollama_chat_stream(
                    conv_system, query, think=False, model=self._model,
                ):
                    if chunk.get("content"):
                        conv_answer += chunk["content"]
                        yield self._sse(chunk["content"])
                    if chunk.get("done"):
                        break
            except Exception:
                if not conv_answer:
                    conv_answer = (
                        "Welcome! I'm here to help with Saudi Arabia's Personal Data "
                        "Protection Law (PDPL). What would you like to know?"
                    )
                    yield self._sse(conv_answer)
            await memory.add_message("assistant", conv_answer)
            yield self._sse_stop()
            return

        if route.kind == "scope_refusal":
            await memory.add_message("assistant", SCOPE_REFUSAL)
            yield self._sse(SCOPE_REFUSAL) + self._sse_stop()
            return

        if route.kind == "agent":
            try:
                answer, citations = await _agent_mode_response(
                    query, route.session_id, model=self._model,
                )
                await memory.add_message("assistant", answer)
                for idx, line in enumerate(answer.split("\n")):
                    yield self._sse(line if idx == 0 else "\n" + line)
                if citations:
                    data = [c.model_dump(mode="json") for c in citations]
                    yield f"data: {json.dumps({'citations': data})}\n\n"
            except DependencyUnavailable as exc:
                msg = (
                    "Search service is temporarily unavailable. "
                    "Please try again in a moment."
                )
                logger.exception(
                    "Dependency unavailable", session_id=route.session_id, service=exc.service
                )
                yield self._sse(msg)
            except Exception:
                yield self._sse("Sorry, I encountered an error.")
            yield self._sse_stop()
            return

        # --- RAG mode: real-time streaming from Ollama with thinking ---

        cached = await lookup_cache(query)
        if cached is not None:
            await memory.add_message("assistant", cached.answer)
            for idx, line in enumerate(cached.answer.split("\n")):
                yield self._sse(line if idx == 0 else "\n" + line)
            if cached.citations:
                data = [c.model_dump(mode="json") for c in cached.citations]
                yield f"data: {json.dumps({'citations': data})}\n\n"
            yield self._sse_stop()
            return

        yield self._sse("<think>Searching documents...")

        try:
            rag_payload = await _prepare_rag(memory, query)
        except IndexMismatchError:
            msg = (
                "Index embedding mismatch. Reindex documents or update "
                "INDEX_VERSION/EMBEDDING_MODEL to match the existing index."
            )
            logger.exception("Index mismatch during retrieval", session_id=route.session_id)
            yield self._sse("</think>")
            yield self._sse(msg)
            yield self._sse_stop()
            return
        except DependencyUnavailable as exc:
            msg = (
                "Search service is temporarily unavailable. "
                "Please try again in a moment."
            )
            logger.exception(
                "Dependency unavailable", session_id=route.session_id, service=exc.service
            )
            yield self._sse("</think>")
            yield self._sse(msg)
            yield self._sse_stop()
            return
        except Exception:
            logger.exception("Retrieval failed", session_id=route.session_id)
            rag_payload = None

        if rag_payload is None:
            yield self._sse("</think>")
            yield self._sse("Sorry, I encountered an error. Please try again.")
            yield self._sse_stop()
            return

        full_answer = ""
        in_thinking = True

        try:
            async for chunk in ollama_chat_stream(
                rag_payload.system_prompt,
                rag_payload.user_prompt,
                think=True,
                model=self._model,
            ):
                if chunk.get("thinking"):
                    yield self._sse(chunk["thinking"])

                if chunk.get("content"):
                    if in_thinking:
                        yield self._sse("</think>")
                        in_thinking = False
                    full_answer += chunk["content"]
                    yield self._sse(chunk["content"])

            if chunk.get("done"):
                if in_thinking:
                    yield self._sse("</think>")
                break

            # No-op; we handle sources after stream ends

        sources_footer = _format_sources_footer(rag_payload.citations)
        if sources_footer:
            yield self._sse(sources_footer)
            full_answer += sources_footer

        yield self._sse(CLOSING_LINE)
        full_answer += CLOSING_LINE
        await memory.add_message("assistant", full_answer)
            await store_cache(query, full_answer, rag_payload.citations)
        except Exception:
            logger.exception("Streaming failed", session_id=route.session_id)
            if in_thinking:
                yield self._sse("</think>")
            fallback = _fallback_rag_answer(rag_payload.citations)
            yield self._sse(fallback)
            await memory.add_message("assistant", fallback)
            if rag_payload.citations:
                data = [c.model_dump(mode="json") for c in rag_payload.citations]
                yield f"data: {json.dumps({'citations': data})}\n\n"
            yield self._sse_stop()
            return

        if rag_payload.citations:
            data = [c.model_dump(mode="json") for c in rag_payload.citations]
            yield f"data: {json.dumps({'citations': data})}\n\n"

        yield self._sse(finish="stop")
        yield "data: [DONE]\n\n"
