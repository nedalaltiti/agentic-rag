"""OpenAI-compatible chat completions endpoint."""

import asyncio
import hashlib
import json
import time
import uuid
from collections.abc import AsyncGenerator
from typing import Literal

import structlog
from fastapi import APIRouter, Header, HTTPException
from fastapi.responses import StreamingResponse

from agentic_rag.backend.rag.reranker import LLMReranker
from agentic_rag.backend.rag.retriever import HybridRetriever
from agentic_rag.core.citations import format_citations
from agentic_rag.core.config import settings
from agentic_rag.core.llm_factory import ollama_chat_stream, ollama_chat_with_thinking
from agentic_rag.core.memory import ConversationMemory
from agentic_rag.core.prompts import PromptRegistry
from agentic_rag.core.schemas import (
    Citation,
    OpenAIChatChoice,
    OpenAIChatMessage,
    OpenAIChatRequest,
    OpenAIChatResponse,
    OpenAIChatStreamChoice,
    OpenAIChatStreamChunk,
    OpenAIChatStreamDelta,
    TokenUsage,
)
from agentic_rag.core.scope_gate import ScopeGate

logger = structlog.get_logger()

router = APIRouter(tags=["chat"])

SCOPE_REFUSAL = (
    "I specialize in PDPL and data protection topics only. "
    "I'm not able to help with that question. "
    "Feel free to ask me anything about Saudi Arabia's Personal Data Protection Law!"
)

AGENT_MODE_KEYWORDS = {
    "compare",
    "plan",
    "steps",
    "analyze in detail",
    "based on our previous",
    "use agent",
    "multi-step",
    "research and",
    "investigate",
}


def _get_session_id(
    messages: list,
    session_header: str | None = None,
) -> str:
    """Get session ID from header, or derive deterministically from the first user message."""
    if session_header:
        return session_header

    if messages:
        first_user = next((m for m in messages if m.role == "user"), None)
        if first_user:
            return hashlib.sha256(first_user.content.encode()).hexdigest()[:16]

    return str(uuid.uuid4())[:16]


def _should_use_agent_mode(query: str, agent_header: str | None) -> bool:
    """Check header and keyword triggers; skip for OpenWebUI internal requests."""
    if not settings.USE_CREWAI:
        return False

    if query.strip().startswith("### Task:") or query.strip().startswith("###"):
        return False

    if agent_header and agent_header.lower() == "true":
        return True

    query_lower = query.lower()
    return any(kw in query_lower for kw in AGENT_MODE_KEYWORDS)


def _is_conversational(query: str) -> bool:
    """Check if query is conversational (greeting, thanks, farewell) - no RAG needed."""
    conversational = {
        "hello",
        "hi",
        "hey",
        "hii",
        "helloo",
        "hiii",
        "heyyy",
        "yo",
        "thanks",
        "thank you",
        "thx",
        "ty",
        "appreciated",
        "great",
        "awesome",
        "ok",
        "okay",
        "sure",
        "yes",
        "no",
        "no thanks",
        "nothing",
        "got it",
        "bye",
        "goodbye",
        "see you",
        "later",
        "take care",
    }
    cleaned = query.strip().lower().rstrip("!.,?")
    return cleaned in conversational or len(cleaned) <= 3


async def _conversational_response(query: str) -> str:
    """Quick LLM response for greetings/thanks."""
    system_prompt = (
        "You are a friendly PDPL (Personal Data Protection Law) compliance assistant "
        "for Saudi Arabia. Respond naturally and briefly to the user's message. "
        "Always end by offering help with PDPL and data protection topics. "
        "Keep responses to 1-2 sentences max. Be warm but professional. No emojis."
    )

    try:
        _, content = await ollama_chat_with_thinking(
            system_prompt=system_prompt,
            user_message=query,
            think=False,
        )
        return content.strip()
    except Exception:
        return (
            "Welcome! I'm here to help with Saudi Arabia's Personal Data Protection Law (PDPL). "
            "What would you like to know?"
        )


def _is_openwebui_internal_request(query: str) -> tuple[bool, str]:
    """Detect and short-circuit OpenWebUI internal requests (title gen, follow-ups, tags)."""
    query_lower = query.strip().lower()
    app_name = settings.APP_NAME

    if "### task:" in query_lower and "title" in query_lower and "emoji" in query_lower:
        return True, f'{{"title": "{app_name} Chat"}}'

    if "### task:" in query_lower and "follow-up questions" in query_lower:
        return True, (
            '{"questions": ['
            '"What are the key obligations for data controllers under PDPL?", '
            '"How does the PDPL handle cross-border data transfers?", '
            '"What are the penalties for non-compliance with PDPL?"]}'
        )

    if "### task:" in query_lower and ("tags" in query_lower or "classify" in query_lower):
        return True, '{"tags": ["rag", "knowledge-base", "assistant"]}'

    if query.strip().startswith("### Task:") or query.strip().startswith("###Task:"):
        return True, "OK"

    return False, ""


def _format_context_for_llm(citations: list[Citation]) -> str:
    """Format citations as context for the LLM prompt."""
    if not citations:
        return "No relevant documents found in the knowledge base."

    context_parts = []
    for i, cit in enumerate(citations, 1):
        doc_name = cit.file_name.replace("+", " ").replace(".md", "").replace("_", " ")

        header_parts = [f"Source [{i}]: {doc_name}"]
        if cit.section_path and cit.section_path != "N/A":
            header_parts.append(f"Section: {cit.section_path}")
        if cit.page_number:
            header_parts.append(f"Page: {cit.page_number}")

        header = " | ".join(header_parts)

        content = cit.chunk_text[:1500].strip()

        context_parts.append(f"{header}\n{content}")

    return "\n\n---\n\n".join(context_parts)


async def _retrieve_and_rerank(
    query: str,
    use_reranker: bool = False,
) -> list[Citation]:
    """Retrieve documents and optionally rerank them."""
    retriever = HybridRetriever(include_toc=False)

    try:
        nodes = await retriever.aretrieve(query)

        if use_reranker and nodes:
            reranker = LLMReranker()
            reranker._semaphore = asyncio.Semaphore(1)
            nodes = await reranker.rerank(query, nodes[:5])
        else:
            nodes = nodes[:5]

        citations = format_citations(nodes)

        logger.info(
            "Retrieval completed",
            query=query[:50],
            documents_found=len(citations),
            reranked=use_reranker,
        )

        return citations
    except Exception:
        logger.exception("Retrieval failed")
        return []


def _format_sources_section(citations: list[Citation]) -> str:
    """Format citations as a visible Sources section for the answer."""
    if not citations:
        return ""

    sources = []
    seen = set()  # Deduplicate sources

    for i, c in enumerate(citations[:5], 1):
        doc_name = c.file_name.replace("+", " ").replace(".md", "").replace("_", " ")

        section = c.section_path if c.section_path and c.section_path != "N/A" else ""
        source_key = f"{doc_name}|{section}"

        if source_key not in seen:
            seen.add(source_key)
            if section:
                sources.append(f"- [{i}] {doc_name} — {section}")
            else:
                sources.append(f"- [{i}] {doc_name}")

    if not sources:
        return ""

    return "\n\n---\n\n**Sources:**\n\n" + "\n".join(sources)


def _format_history(messages: list) -> str:
    """Format conversation history for prompt injection."""
    if not messages:
        return ""
    lines = []
    for msg in messages:
        role = msg.role.value.capitalize() if hasattr(msg.role, "value") else str(msg.role)
        content = msg.content[:500] if msg.content else ""
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


async def _fast_rag_response(
    query: str,
    citations: list[Citation],
    history: list | None = None,
) -> str:
    """Single LLM call with retrieved context (default path)."""
    context = _format_context_for_llm(citations)
    history_text = _format_history(history or [])

    system_prompt = PromptRegistry.render("system_prompt")
    user_prompt = PromptRegistry.render(
        "user_prompt",
        query=query,
        context=context,
        history=history_text,
    )

    thinking, content = await ollama_chat_with_thinking(
        system_prompt=system_prompt,
        user_message=user_prompt,
        think=True,
    )

    answer = ""
    if thinking:
        answer = f"<think>\n{thinking}\n</think>\n\n"
    answer += content.strip()
    answer += _format_sources_section(citations)

    return answer


async def _agent_mode_response(
    query: str,
    session_id: str,
    citations: list[Citation],
) -> str:
    """Multi-step response via CrewAI agent pipeline."""
    from agentic_rag.backend.crew.runner import CrewRunner

    context = _format_context_for_llm(citations)

    runner = CrewRunner(session_id)
    answer = await asyncio.to_thread(runner.kickoff_with_context, query, context)

    if citations and "Sources:" not in answer and "References" not in answer:
        answer += _format_sources_section(citations)

    return answer


async def _process_query(
    query: str,
    session_id: str,
    use_agent_mode: bool = False,
) -> tuple[str, list[Citation]]:
    """Route query through internal-request check, conversational check, or RAG pipeline."""
    is_internal, internal_response = _is_openwebui_internal_request(query)
    if is_internal:
        logger.info("Bypassing RAG for OpenWebUI internal request", session_id=session_id)
        return internal_response, []

    memory = ConversationMemory(session_id)

    await memory.add_message("user", query)

    if _is_conversational(query):
        logger.info("Handling conversational message", session_id=session_id)
        answer = await _conversational_response(query)
        await memory.add_message("assistant", answer)
        return answer, []

    # Semantic scope gate
    try:
        in_scope, _ = await ScopeGate.is_in_scope(query)
    except Exception:
        in_scope = True
    if not in_scope:
        logger.info("Out-of-scope query refused (non-stream)", session_id=session_id)
        await memory.add_message("assistant", SCOPE_REFUSAL)
        return SCOPE_REFUSAL, []

    citations = await _retrieve_and_rerank(query, use_reranker=use_agent_mode)

    try:
        if use_agent_mode:
            logger.info("Using Agent Mode (CrewAI)", session_id=session_id)
            answer = await _agent_mode_response(query, session_id, citations)
        else:
            logger.info("Using Fast RAG Mode", session_id=session_id)
            history = await memory.get_history(limit=5)
            answer = await _fast_rag_response(query, citations, history=history)
    except Exception:
        logger.exception("Response generation failed", session_id=session_id)
        raise HTTPException(status_code=500, detail="Failed to generate response") from None

    await memory.add_message("assistant", answer)

    return answer, citations


async def _stream_with_thinking(
    request_id: str,
    model: str,
    query: str,
    session_id: str,
    use_agent_mode: bool,
    created_at: int,
) -> AsyncGenerator[str, None]:
    """Stream the response as SSE chunks with real-time thinking."""

    def sse(
        content: str = "",
        role: Literal["assistant"] | None = None,
        finish: str | None = None,
    ) -> str:
        chunk = OpenAIChatStreamChunk(
            id=request_id,
            created=created_at,
            model=model,
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

    def sse_stop() -> str:
        return sse(finish="stop") + "data: [DONE]\n\n"

    yield sse(role="assistant")

    # Internal OpenWebUI requests (title gen, follow-ups, tags)
    is_internal, internal_response = _is_openwebui_internal_request(query)
    if is_internal:
        yield sse(internal_response) + sse_stop()
        return

    memory = ConversationMemory(session_id)
    await memory.add_message("user", query)

    # Conversational (greetings, thanks)
    if _is_conversational(query):
        answer = await _conversational_response(query)
        await memory.add_message("assistant", answer)
        yield sse(answer) + sse_stop()
        return

    # Scope gate: refuse off-topic before retrieval
    try:
        in_scope, _ = await ScopeGate.is_in_scope(query)
    except Exception:
        in_scope = True

    if not in_scope:
        await memory.add_message("assistant", SCOPE_REFUSAL)
        yield sse(SCOPE_REFUSAL) + sse_stop()
        return

    # Agent mode: non-streaming fallback
    if use_agent_mode:
        try:
            citations = await _retrieve_and_rerank(query, use_reranker=True)
            answer = await _agent_mode_response(query, session_id, citations)
            await memory.add_message("assistant", answer)
            for idx, line in enumerate(answer.split("\n")):
                yield sse(line if idx == 0 else "\n" + line)
        except Exception:
            yield sse("Sorry, I encountered an error.")
        yield sse_stop()
        return

    # --- RAG mode: real-time streaming from Ollama with thinking ---

    yield sse("<think>Searching documents...")

    try:
        citations = await _retrieve_and_rerank(query, use_reranker=False)
    except Exception:
        logger.exception("Retrieval failed", session_id=session_id)
        citations = []

    context = _format_context_for_llm(citations)
    history_text = _format_history(await memory.get_history(limit=5))

    system_prompt = PromptRegistry.render("system_prompt")
    user_prompt = PromptRegistry.render(
        "user_prompt", query=query, context=context, history=history_text
    )

    full_answer = ""
    in_thinking = True  # <think> already opened above

    try:
        async for chunk in ollama_chat_stream(system_prompt, user_prompt, think=True):
            if chunk.get("thinking"):
                yield sse(chunk["thinking"])

            if chunk.get("content"):
                if in_thinking:
                    yield sse("</think>")
                    in_thinking = False
                full_answer += chunk["content"]
                yield sse(chunk["content"])

            if chunk.get("done"):
                if in_thinking:
                    yield sse("</think>")
                break

        sources = _format_sources_section(citations)
        if sources:
            yield sse(sources)
            full_answer += sources

        await memory.add_message("assistant", full_answer)
    except Exception:
        logger.exception("Streaming failed", session_id=session_id)
        yield sse("Sorry, I encountered an error. Please try again.")

    yield sse(finish="stop")

    if citations:
        data = [c.model_dump(mode="json") for c in citations]
        yield f"data: {json.dumps({'citations': data})}\n\n"

    yield "data: [DONE]\n\n"


@router.post("/v1/chat/completions")
async def chat_completions(
    request: OpenAIChatRequest,
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
    model = request.model or settings.LLM_MODEL

    use_agent_mode = _should_use_agent_mode(query, x_agent_mode)

    logger.info(
        "Chat request received",
        model=model,
        stream=request.stream,
        session_id=session_id,
        agent_mode=use_agent_mode,
    )

    request_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
    created_at = int(time.time())

    if request.stream:
        return StreamingResponse(
            _stream_with_thinking(request_id, model, query, session_id, use_agent_mode, created_at),
            media_type="text/event-stream",
        )

    try:
        answer, citations = await _process_query(query, session_id, use_agent_mode)
    except HTTPException:
        raise
    except Exception:
        logger.exception("Query processing failed", session_id=session_id)
        raise HTTPException(status_code=500, detail="Query processing failed") from None

    return OpenAIChatResponse(
        id=request_id,
        created=created_at,
        model=model,
        choices=[
            OpenAIChatChoice(
                index=0,
                message=OpenAIChatMessage(role="assistant", content=answer),
                finish_reason="stop",
            )
        ],
        usage=TokenUsage(
            prompt_tokens=len(query.split()),
            completion_tokens=len(answer.split()),
            total_tokens=len(query.split()) + len(answer.split()),
        ),
        citations=citations,
    )
