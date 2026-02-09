"""Shared chat routing and RAG helpers."""

from __future__ import annotations

import asyncio
import hashlib
import re
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Literal

import structlog

from agentic_rag.backend.rag.reranker import LLMReranker
from agentic_rag.backend.rag.retriever import HybridRetriever
from agentic_rag.core.citations import format_citations
from agentic_rag.core.config import settings
from agentic_rag.core.exceptions import DependencyUnavailable, IndexMismatchError
from agentic_rag.core.llm_factory import ollama_chat_with_thinking
from agentic_rag.core.memory import ConversationMemory
from agentic_rag.core.prompts import PromptRegistry
from agentic_rag.core.schemas import Citation
from agentic_rag.core.scope_gate import ScopeGate

logger = structlog.get_logger()

_PROMPT_CACHE: OrderedDict[str, tuple[str, float]] = OrderedDict()
_CONTEXT_CACHE: OrderedDict[str, tuple[str, float]] = OrderedDict()
_HISTORY_CACHE: OrderedDict[str, tuple[str, float]] = OrderedDict()


def _cache_get(cache: OrderedDict[str, tuple[str, float]], key: str, ttl: int) -> str | None:
    if ttl <= 0:
        return None
    cached = cache.get(key)
    if cached is None:
        return None
    value, ts = cached
    if (time.monotonic() - ts) > ttl:
        cache.pop(key, None)
        return None
    cache.move_to_end(key)
    return value


def _cache_set(
    cache: OrderedDict[str, tuple[str, float]],
    key: str,
    value: str,
    max_size: int,
) -> None:
    cache[key] = (value, time.monotonic())
    if len(cache) > max_size:
        cache.popitem(last=False)


def _hash_parts(*parts: str) -> str:
    digest = hashlib.sha256()
    for part in parts:
        digest.update(part.encode("utf-8"))
    return digest.hexdigest()


CLOSING_LINE = f"\n\n{settings.DOMAIN_CLOSING}"

# Patterns that attempt to inject fake prompt sections or override instructions.
_INJECTION_RE = re.compile(
    r"(===\s*(INSTRUCTIONS|CONTEXT|SYSTEM|QUESTION|RULES|HISTORY)\s*===)"
    r"|(\bignore\s+(all\s+)?(previous|above|prior)\s+(instructions|rules|prompts)\b)"
    r"|(\byou\s+are\s+now\b)"
    r"|(\bsystem\s*:\s)",
    re.IGNORECASE,
)


def _sanitize_query(query: str) -> str:
    """Neutralise common prompt-injection patterns in user input.

    Strips delimiter fences and meta-instructions that try to escape the
    user-input boundary.  The original question intent is preserved.
    """
    return _INJECTION_RE.sub("", query).strip()

SCOPE_REFUSAL = (
    f"I specialize in {settings.DOMAIN_TOPICS} only. "
    "I'm not able to help with that question. "
    f"Feel free to ask me anything about {settings.DOMAIN_REGION}'s "
    f"{settings.DOMAIN_FULL_NAME}!"
)


@dataclass
class RouteDecision:
    """Result of routing a user query through the decision chain."""

    kind: Literal["internal", "conversational", "scope_refusal", "agent", "rag"]
    session_id: str
    memory: ConversationMemory | None = None
    internal_response: str = ""


@dataclass
class RagPayload:
    """Prepared RAG inputs for consistent streaming/non-streaming use."""

    citations: list[Citation] = field(default_factory=list)
    history: list = field(default_factory=list)
    context: str = ""
    system_prompt: str = ""
    user_prompt: str = ""


def _is_conversational(query: str) -> bool:
    """Check if query is conversational (greeting, thanks, farewell) - no RAG needed."""
    conversational = {
        "hello",
        "hi",
        "hey",
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
        "cool",
    }
    cleaned = query.strip().lower().rstrip("!.,?")
    collapsed = re.sub(r"(.)\1{2,}$", r"\1", cleaned)
    collapsed2 = re.sub(r"(.)\1{2,}", r"\1\1", cleaned)
    collapsed3 = re.sub(r"(.)\1{2,}", r"\1", cleaned)
    return (
        cleaned in conversational
        or collapsed in conversational
        or collapsed2 in conversational
        or collapsed3 in conversational
    )


async def _conversational_response(query: str, model: str | None = None) -> str:
    """Quick LLM response for greetings/thanks."""
    system_prompt = PromptRegistry.render("conversational_prompt")
    fallback = (
        f"Welcome! I'm here to help with {settings.DOMAIN_REGION}'s "
        f"{settings.DOMAIN_FULL_NAME} ({settings.DOMAIN_NAME}). "
        "What would you like to know?"
    )

    try:
        _, content = await ollama_chat_with_thinking(
            system_prompt=system_prompt,
            user_message=query,
            think=False,
            model=model,
        )
        return content.strip() or fallback
    except Exception:
        return fallback


def _is_openwebui_internal_request(query: str) -> tuple[bool, str]:
    """Detect and short-circuit OpenWebUI internal requests."""
    query_lower = query.strip().lower()
    app_name = settings.APP_NAME

    if "### task:" in query_lower and "title" in query_lower and "emoji" in query_lower:
        return True, f'{{"title": "{app_name} Chat"}}'

    if "### task:" in query_lower and ("tags" in query_lower or "classify" in query_lower):
        return True, '{"tags": ["rag", "knowledge-base", "assistant"]}'

    if query.strip().startswith("### Task:") or query.strip().startswith("###Task:"):
        return True, "OK"

    return False, ""


def _format_context_for_llm(citations: list[Citation]) -> str:
    """Format citations as context for the LLM prompt."""
    if not citations:
        return "No relevant information found in the knowledge base."

    ttl = settings.PROMPT_CACHE_TTL_SECONDS
    max_size = settings.PROMPT_CACHE_MAX
    cache_key = _hash_parts(
        "context",
        *[
            f"{c.chunk_id}|{c.file_name}|{c.page_number}|{c.section_path}|{c.chunk_text}"
            for c in citations
        ],
    )
    cached = _cache_get(_CONTEXT_CACHE, cache_key, ttl)
    if cached is not None:
        return cached

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

    context = "\n\n---\n\n".join(context_parts)
    _cache_set(_CONTEXT_CACHE, cache_key, context, max_size)
    return context


def _fallback_rag_answer(citations: list[Citation]) -> str:
    """Fallback answer when the LLM is unavailable or times out."""
    if not citations:
        return (
            "I'm having trouble generating a full answer right now. Please try again in a moment."
        )

    lines = [
        "I'm having trouble generating a full answer right now.",
        "Here are the most relevant sources I found:",
    ]
    for i, c in enumerate(citations[:3], 1):
        loc_parts = []
        if c.section_path:
            loc_parts.append(c.section_path)
        if c.page_number is not None:
            loc_parts.append(f"p.{c.page_number}")
        loc = f" ({' | '.join(loc_parts)})" if loc_parts else ""
        lines.append(f"[{i}] {c.file_name}{loc}")

    return "\n".join(lines) + CLOSING_LINE


def _format_sources_footer(citations: list[Citation]) -> str:
    """Format a small sources footer for UI visibility."""
    if not citations:
        return ""

    lines = ["", "Sources:"]
    for i, c in enumerate(citations, 1):
        loc_parts = []
        if c.section_path:
            loc_parts.append(c.section_path)
        if c.page_number is not None:
            loc_parts.append(f"p.{c.page_number}")
        loc = f" ({' | '.join(loc_parts)})" if loc_parts else ""
        lines.append(f"[{i}] {c.file_name}{loc}")

    return "\n".join(lines)


def _get_system_prompt() -> str:
    ttl = settings.PROMPT_CACHE_TTL_SECONDS
    max_size = settings.PROMPT_CACHE_MAX
    cache_key = "system_prompt"
    cached = _cache_get(_PROMPT_CACHE, cache_key, ttl)
    if cached is not None:
        return cached
    prompt = PromptRegistry.render("system_prompt")
    _cache_set(_PROMPT_CACHE, cache_key, prompt, max_size)
    return prompt


def _get_user_prompt(query: str, context: str, history: str) -> str:
    safe_query = _sanitize_query(query)
    ttl = settings.PROMPT_CACHE_TTL_SECONDS
    max_size = settings.PROMPT_CACHE_MAX
    cache_key = _hash_parts("user_prompt", safe_query, context, history)
    cached = _cache_get(_PROMPT_CACHE, cache_key, ttl)
    if cached is not None:
        return cached
    prompt = PromptRegistry.render(
        "user_prompt", query=safe_query, context=context, history=history,
    )
    _cache_set(_PROMPT_CACHE, cache_key, prompt, max_size)
    return prompt


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
    except DependencyUnavailable:
        logger.exception("Dependency unavailable during retrieval")
        raise
    except IndexMismatchError:
        logger.exception("Index mismatch during retrieval")
        raise
    except Exception:
        logger.exception("Retrieval failed")
        return []


def _format_history(messages: list) -> str:
    """Format conversation history for prompt injection."""
    if not messages:
        return ""
    ttl = settings.PROMPT_CACHE_TTL_SECONDS
    max_size = settings.PROMPT_CACHE_MAX
    cache_key = _hash_parts(
        "history",
        *[f"{getattr(m.role, 'value', str(m.role))}:{(m.content or '')[:500]}" for m in messages],
    )
    cached = _cache_get(_HISTORY_CACHE, cache_key, ttl)
    if cached is not None:
        return cached
    lines = []
    for msg in messages:
        role = msg.role.value.capitalize() if hasattr(msg.role, "value") else str(msg.role)
        content = msg.content[:500] if msg.content else ""
        lines.append(f"{role}: {content}")
    history = "\n".join(lines)
    _cache_set(_HISTORY_CACHE, cache_key, history, max_size)
    return history


async def _fast_rag_response(
    query: str,
    citations: list[Citation],
    history: list | None = None,
    model: str | None = None,
    system_prompt: str | None = None,
    user_prompt: str | None = None,
) -> str:
    """Single LLM call with retrieved context (default path)."""
    if not citations:
        return "No relevant information found in the knowledge base."

    if system_prompt is None or user_prompt is None:
        context = _format_context_for_llm(citations)
        history_text = _format_history(history or [])

        system_prompt = _get_system_prompt()
        user_prompt = _get_user_prompt(query, context, history_text)

    thinking, content = await ollama_chat_with_thinking(
        system_prompt=system_prompt,
        user_message=user_prompt,
        think=True,
        model=model,
    )

    answer = ""
    if thinking:
        answer = f"<think>\n{thinking}\n</think>\n\n"
    answer += content.strip()
    answer += _format_sources_footer(citations)
    answer += CLOSING_LINE

    return answer


async def _agent_mode_response(
    query: str,
    session_id: str,
    model: str | None = None,
) -> tuple[str, list[Citation]]:
    """Multi-step response via CrewAI agent pipeline."""
    from agentic_rag.backend.crew.runner import CrewRunner

    runner = CrewRunner(session_id, model=model)
    try:
        answer, tool_citations = await asyncio.to_thread(runner.kickoff, query)
    except Exception:
        logger.exception("Agent execution failed; falling back to retrieval", session_id=session_id)
        answer, tool_citations = "", []

    if not tool_citations:
        logger.warning(
            "Agent returned no citations; falling back to retrieval",
            session_id=session_id,
        )
        fallback_citations = await _retrieve_and_rerank(query, use_reranker=True)
        if fallback_citations:
            try:
                fallback_answer = await _fast_rag_response(
                    query,
                    fallback_citations,
                    history=None,
                    model=model,
                )
            except Exception:
                logger.exception("LLM generation failed; using fallback response")
                fallback_answer = _fallback_rag_answer(fallback_citations)
            return fallback_answer, fallback_citations
        return "No relevant information found in the knowledge base.", []

    return answer, tool_citations


async def _route_decision(
    query: str,
    session_id: str,
    use_agent_mode: bool,
) -> RouteDecision:
    """Single source of truth for query routing decisions."""
    is_internal, internal_response = _is_openwebui_internal_request(query)
    if is_internal:
        logger.info(
            "Bypassing RAG for OpenWebUI internal request",
            session_id=session_id,
        )
        return RouteDecision(
            kind="internal",
            session_id=session_id,
            internal_response=internal_response,
        )

    memory = ConversationMemory(session_id)
    await memory.add_message("user", query)

    if _is_conversational(query):
        logger.info("Handling conversational message", session_id=session_id)
        return RouteDecision(
            kind="conversational",
            session_id=session_id,
            memory=memory,
        )

    try:
        in_scope, _ = await ScopeGate.is_in_scope(query)
    except Exception:
        logger.exception("Scope gate error, refusing query", session_id=session_id)
        in_scope = False

    if not in_scope:
        logger.info("Out-of-scope query refused", session_id=session_id)
        return RouteDecision(
            kind="scope_refusal",
            session_id=session_id,
            memory=memory,
        )

    if use_agent_mode:
        logger.info("Using Agent Mode (CrewAI)", session_id=session_id)
        return RouteDecision(
            kind="agent",
            session_id=session_id,
            memory=memory,
        )

    logger.info("Using Fast RAG Mode", session_id=session_id)
    return RouteDecision(
        kind="rag",
        session_id=session_id,
        memory=memory,
    )


async def _prepare_rag(memory: ConversationMemory, query: str) -> RagPayload:
    """Prepare RAG inputs (retrieval + prompt rendering)."""
    citations = await _retrieve_and_rerank(query, use_reranker=False)
    history = await memory.get_history(limit=5)
    context = _format_context_for_llm(citations)
    history_text = _format_history(history)

    system_prompt = _get_system_prompt()
    user_prompt = _get_user_prompt(query, context, history_text)

    return RagPayload(
        citations=citations,
        history=history,
        context=context,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
    )
