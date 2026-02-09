"""OpenAI-compatible chat completions endpoint."""

import json
import re
import time
import uuid
from collections.abc import AsyncGenerator

import structlog
from fastapi import APIRouter, Header, HTTPException, Response
from fastapi.responses import StreamingResponse

from agentic_rag.backend.api.v1.chat_service import (
    SCOPE_REFUSAL,
    RouteDecision,
    _agent_mode_response,
    _conversational_response,
    _fallback_rag_answer,
    _fast_rag_response,
    _prepare_rag,
    _route_decision,
)
from agentic_rag.backend.rag.semantic_cache import lookup_cache, store_cache
from agentic_rag.core.config import settings
from agentic_rag.core.exceptions import DependencyUnavailable, IndexMismatchError
from agentic_rag.core.llm_factory import ollama_chat_with_thinking
from agentic_rag.core.schemas import (
    Citation,
    OpenAIChatChoice,
    OpenAIChatMessage,
    OpenAIChatRequest,
    OpenAIChatResponse,
    TokenUsage,
)

logger = structlog.get_logger()

router = APIRouter(tags=["chat"])

# Patterns that trigger agent (multi-step) mode.  Each is compiled as a
# word-boundary regex so "compare" won't match inside "incomparable".
_AGENT_MODE_PATTERNS = [
    re.compile(r"\b" + p + r"\b", re.IGNORECASE)
    for p in [
        r"compare",
        r"analyze in detail",
        r"based on our previous",
        r"use agent",
        r"multi-step",
        r"research and",
        r"investigate",
        r"step[- ]by[- ]step",
    ]
]


def _get_session_id(
    messages: list,
    session_header: str | None = None,
) -> str:
    """Get session ID from header, or generate a new one."""
    if session_header:
        return session_header

    return str(uuid.uuid4())[:16]


def _should_use_agent_mode(query: str, agent_header: str | None) -> bool:
    """Check header and keyword triggers; skip for OpenWebUI internal requests."""
    if not settings.USE_CREWAI:
        return False

    stripped = query.strip()
    if stripped.startswith("### Task:") or stripped.startswith("###Task:"):
        return False

    if agent_header and agent_header.lower() == "true":
        return True

    return any(p.search(query) for p in _AGENT_MODE_PATTERNS)


async def _generate_followup_questions(
    messages: list[OpenAIChatMessage],
) -> str:
    """Generate contextual follow-up questions based on conversation history."""
    # Extract last assistant message for context
    last_assistant = ""
    for msg in reversed(messages):
        if msg.role == "assistant" and msg.content:
            last_assistant = msg.content[:1000]
            break

    domain = settings.DOMAIN_NAME
    default_questions = [
        f"What are the key obligations under {domain}?",
        f"How does {domain} handle cross-border data transfers?",
        f"What are the penalties for non-compliance with {domain}?",
    ]

    if not last_assistant:
        return json.dumps({"questions": default_questions})

    prompt = (
        "Based on this assistant response, generate exactly 3 short follow-up "
        f"questions the user might ask next about {domain}. Return ONLY a JSON object "
        'in this format: {"questions": ["q1", "q2", "q3"]}\n\n'
        f"Assistant response:\n{last_assistant}"
    )

    try:
        _, content, _ = await ollama_chat_with_thinking(
            system_prompt="You generate follow-up questions. Reply with JSON only.",
            user_message=prompt,
            think=False,
        )
        # Validate it's parseable JSON
        parsed = json.loads(content.strip())
        if "questions" in parsed and isinstance(parsed["questions"], list):
            return json.dumps(parsed)
    except Exception:
        logger.debug("Follow-up generation failed, using defaults")

    return json.dumps({"questions": default_questions})


_ZERO_USAGE: dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


async def _process_query(
    query: str,
    session_id: str,
    use_agent_mode: bool = False,
    model: str | None = None,
) -> tuple[str, list[Citation], dict[str, int]]:
    """Route query once, then render response for non-streaming clients."""
    route = await _route_decision(query, session_id, use_agent_mode)

    if route.kind == "internal":
        return route.internal_response, [], _ZERO_USAGE

    memory = route.memory
    if memory is None:
        raise HTTPException(status_code=500, detail="Failed to initialize memory")

    if route.kind == "conversational":
        answer = await _conversational_response(query, model=model)
        await memory.add_message("assistant", answer)
        return answer, [], _ZERO_USAGE

    if route.kind == "scope_refusal":
        await memory.add_message("assistant", SCOPE_REFUSAL)
        return SCOPE_REFUSAL, [], _ZERO_USAGE

    usage = _ZERO_USAGE
    try:
        if route.kind == "agent":
            answer, citations = await _agent_mode_response(
                query,
                session_id,
                model=model,
            )
        else:
            used_fallback = False
            cached = await lookup_cache(query)
            if cached is not None:
                await memory.add_message("assistant", cached.answer)
                return cached.answer, cached.citations, _ZERO_USAGE

            rag_payload = await _prepare_rag(memory, query)
            citations = rag_payload.citations
            try:
                answer, usage = await _fast_rag_response(
                    query,
                    citations,
                    history=rag_payload.history,
                    model=model,
                    system_prompt=rag_payload.system_prompt,
                    user_prompt=rag_payload.user_prompt,
                )
            except Exception:
                logger.exception("LLM generation failed; using fallback response")
                answer = _fallback_rag_answer(citations)
                used_fallback = True
    except IndexMismatchError as exc:
        logger.exception("Index mismatch during retrieval", details=exc.details)
        raise HTTPException(
            status_code=409,
            detail=(
                "Index embedding mismatch. Reindex documents or update INDEX_VERSION/"
                "EMBEDDING_MODEL to match the existing index."
            ),
        ) from None
    except DependencyUnavailable as exc:
        logger.exception("Dependency unavailable", service=getattr(exc, "service", None))
        answer = "Search service is temporarily unavailable. Please try again in a moment."
        await memory.add_message("assistant", answer)
        return answer, [], _ZERO_USAGE
    except Exception:
        logger.exception("Response generation failed", session_id=session_id)
        raise HTTPException(status_code=500, detail="Failed to generate response") from None

    await memory.add_message("assistant", answer)
    if route.kind == "rag" and not used_fallback and citations:
        await store_cache(query, answer, citations)

    return answer, citations, usage


async def _stream_with_thinking(
    request_id: str,
    model: str,
    query: str,
    session_id: str,
    use_agent_mode: bool,
    created_at: int,
) -> AsyncGenerator[str, None]:
    """Stream the response as SSE chunks with real-time thinking."""
    from agentic_rag.backend.api.v1.streaming import StreamingRenderer

    route = await _route_decision(query, session_id, use_agent_mode)
    renderer = StreamingRenderer(request_id, model, created_at)

    async for chunk in renderer.stream_response(route, query):
        yield chunk


@router.post("/v1/chat/completions")
async def chat_completions(
    request: OpenAIChatRequest,
    response: Response,
    x_session_id: str | None = Header(None, alias="X-Session-Id"),
    x_agent_mode: str | None = Header(None, alias="X-Agent-Mode"),
):
    """OpenAI-compatible chat completions. Supports X-Session-Id and X-Agent-Mode headers."""
    if not request.messages:
        raise HTTPException(status_code=400, detail="Messages list cannot be empty")

    user_messages = [m for m in request.messages if m.role == "user"]
    if not user_messages:
        raise HTTPException(status_code=400, detail="No user message found")

    query = user_messages[-1].content
    session_id = _get_session_id(request.messages, x_session_id)
    if not x_session_id:
        logger.warning(
            "X-Session-Id missing; generated a new session id for this request",
            session_id=session_id,
        )
    # request.model is the logical name (e.g. "agentic-rag") shown in OpenWebUI;
    # always use settings.LLM_MODEL for actual Ollama calls.
    display_model = request.model or settings.LLM_MODEL
    model = settings.LLM_MODEL
    request_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
    created_at = int(time.time())
    should_stream = bool(request.stream) or settings.FORCE_STREAMING

    # Handle follow-up questions request from Open WebUI
    query_lower = query.strip().lower()
    if query_lower.startswith("### task:") and "follow-up" in query_lower:
        followup_json = await _generate_followup_questions(request.messages)
        if should_stream:
            from agentic_rag.backend.api.v1.streaming import StreamingRenderer

            route = RouteDecision(
                kind="internal",
                session_id=session_id,
                internal_response=followup_json,
            )
            renderer = StreamingRenderer(request_id, display_model, created_at)
            return StreamingResponse(
                renderer.stream_response(route, query),
                headers={"X-Session-Id": session_id},
                media_type="text/event-stream",
            )
        return OpenAIChatResponse(
            id=request_id,
            created=created_at,
            model=display_model,
            choices=[
                OpenAIChatChoice(
                    index=0,
                    message=OpenAIChatMessage(
                        role="assistant",
                        content=followup_json,
                    ),
                    finish_reason="stop",
                )
            ],
            usage=TokenUsage(
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
            ),
        )
    use_agent_mode = _should_use_agent_mode(query, x_agent_mode)

    logger.info(
        "Chat request received",
        model=model,
        stream=should_stream,
        session_id=session_id,
        agent_mode=use_agent_mode,
    )

    if should_stream:
        return StreamingResponse(
            _stream_with_thinking(
                request_id,
                display_model,
                query,
                session_id,
                use_agent_mode,
                created_at,
            ),
            headers={"X-Session-Id": session_id},
            media_type="text/event-stream",
        )
    response.headers["X-Session-Id"] = session_id

    try:
        answer, citations, usage = await _process_query(
            query,
            session_id,
            use_agent_mode,
            model=model,
        )
    except HTTPException:
        raise
    except Exception:
        logger.exception("Query processing failed", session_id=session_id)
        raise HTTPException(status_code=500, detail="Query processing failed") from None

    return OpenAIChatResponse(
        id=request_id,
        created=created_at,
        model=display_model,
        choices=[
            OpenAIChatChoice(
                index=0,
                message=OpenAIChatMessage(role="assistant", content=answer),
                finish_reason="stop",
            )
        ],
        usage=TokenUsage(
            prompt_tokens=usage["prompt_tokens"],
            completion_tokens=usage["completion_tokens"],
            total_tokens=usage["total_tokens"],
        ),
        citations=citations,
    )
