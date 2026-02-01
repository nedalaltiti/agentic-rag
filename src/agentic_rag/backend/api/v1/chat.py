"""OpenAI-compatible chat completions endpoint."""

import asyncio
import hashlib
import time
import uuid
from collections.abc import AsyncGenerator
from typing import Literal

import structlog
from fastapi import APIRouter, Header, HTTPException
from fastapi.responses import StreamingResponse

from agentic_rag.backend.rag.reranker import LLMReranker
from agentic_rag.backend.rag.retriever import HybridRetriever
from agentic_rag.shared.citations import format_citations
from agentic_rag.shared.config import settings
from agentic_rag.shared.llm_factory import get_llm
from agentic_rag.shared.memory import ConversationMemory
from agentic_rag.shared.prompts import PromptRegistry
from agentic_rag.shared.schemas import (
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

logger = structlog.get_logger()

router = APIRouter(tags=["chat"])

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
    """Quick LLM response for greetings/thanks — no RAG needed."""
    system_prompt = """You are a friendly PDPL (Personal Data Protection Law) compliance assistant.
Respond naturally and briefly to the user's message.
Always end by offering help with PDPL/data protection topics.
Keep responses to 1-2 sentences max. Be warm but professional. No emojis."""

    llm = get_llm()
    prompt = f"{system_prompt}\n\nUser: {query}\nAssistant:"

    try:
        response = await llm.acomplete(prompt)
        return str(response.text).strip()
    except Exception:
        return (
            "I'm here to help with PDPL compliance matters. "
            "What would you like to know about data protection?"
        )


def _is_openwebui_internal_request(query: str) -> tuple[bool, str]:
    """Detect and short-circuit OpenWebUI internal requests (title gen, follow-ups, tags)."""
    query_lower = query.strip().lower()

    if "### task:" in query_lower and "title" in query_lower and "emoji" in query_lower:
        return True, '{"title": "PDPL Compliance Chat"}'

    if "### task:" in query_lower and "follow-up questions" in query_lower:
        return True, (
            '{"questions": ["What are the key principles of PDPL?", '
            '"How does PDPL affect data processing?", '
            '"What are PDPL compliance requirements?"]}'
        )

    if "### task:" in query_lower and ("tags" in query_lower or "classify" in query_lower):
        return True, '{"tags": ["pdpl", "compliance", "data-protection"]}'

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
            nodes = await reranker.rerank(query, nodes[:10])
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


async def _fast_rag_response(
    query: str,
    citations: list[Citation],
) -> str:
    """Single LLM call with retrieved context (default path)."""
    context = _format_context_for_llm(citations)

    prompt = PromptRegistry.render(
        "user_prompt",
        query=query,
        context=context,
    )

    llm = get_llm()
    response = await llm.acomplete(prompt)
    answer = str(response.text).strip()
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

    citations = await _retrieve_and_rerank(query, use_reranker=use_agent_mode)

    try:
        if use_agent_mode:
            logger.info("Using Agent Mode (CrewAI)", session_id=session_id)
            answer = await _agent_mode_response(query, session_id, citations)
        else:
            logger.info("Using Fast RAG Mode", session_id=session_id)
            answer = await _fast_rag_response(query, citations)
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
    """Stream the response as SSE chunks."""

    def make_chunk(
        content: str,
        role: Literal["assistant"] | None = None,
        finish: str | None = None,
    ):
        return OpenAIChatStreamChunk(
            id=request_id,
            created=created_at,
            model=model,
            choices=[
                OpenAIChatStreamChoice(
                    index=0,
                    delta=OpenAIChatStreamDelta(
                        role=role,
                        content=content if content else None,
                    ),
                    finish_reason=finish,
                )
            ],
        )

    yield f"data: {make_chunk('', role='assistant').model_dump_json()}\n\n"

    try:
        answer, citations = await _process_query(query, session_id, use_agent_mode)
    except Exception:
        error_msg = "Sorry, I encountered an error. Please try again."
        yield f"data: {make_chunk(error_msg).model_dump_json()}\n\n"
        yield f"data: {make_chunk('', finish='stop').model_dump_json()}\n\n"
        yield "data: [DONE]\n\n"
        return

    words = answer.split()
    for i in range(0, len(words), 4):  # 4 words at a time
        chunk_text = " ".join(words[i : i + 4])
        if i > 0:
            chunk_text = " " + chunk_text

        yield f"data: {make_chunk(chunk_text).model_dump_json()}\n\n"
        await asyncio.sleep(0.02)

    yield f"data: {make_chunk('', finish='stop').model_dump_json()}\n\n"
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
    )
