"""OpenAI-compatible chat completions endpoint.

Implements /v1/chat/completions with:
- Non-streaming and streaming responses
- CrewAI orchestration
- Conversation memory persistence
- Citation extraction and formatting
"""

import asyncio
import hashlib
import time
import uuid
from typing import AsyncGenerator

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
import structlog

from agentic_rag.backend.crew.runner import CrewRunner
from agentic_rag.shared.config import settings
from agentic_rag.shared.memory import ConversationMemory
from agentic_rag.shared.schemas import (
    OpenAIChatChoice,
    OpenAIChatMessage,
    OpenAIChatRequest,
    OpenAIChatResponse,
    OpenAIChatStreamChunk,
    OpenAIChatStreamChoice,
    OpenAIChatStreamDelta,
    TokenUsage,
)

logger = structlog.get_logger()

router = APIRouter(tags=["chat"])


def _generate_session_id(messages: list) -> str:
    """
    Generate a deterministic session ID from messages.
    
    Uses SHA256 of first user message for stability across processes.
    In production, consider using HTTP headers or request metadata.
    """
    if not messages:
        return str(uuid.uuid4())
    
    # Use SHA256 of first user message for deterministic session identifier
    first_msg = next((m for m in messages if m.role == "user"), None)
    if first_msg:
        return hashlib.sha256(first_msg.content.encode()).hexdigest()[:16]
    return str(uuid.uuid4())


async def _process_query(query: str, session_id: str) -> tuple[str, list]:
    """
    Process query through CrewAI pipeline.
    
    Returns:
        (answer, citations) tuple
    """
    memory = ConversationMemory(session_id)
    
    # Store user message
    await memory.add_message("user", query)
    
    # Run CrewAI synchronously in thread pool
    runner = CrewRunner(session_id)
    try:
        answer = await asyncio.to_thread(runner.kickoff, query)
    except Exception as e:
        logger.error("CrewAI execution failed", error=str(e), session_id=session_id)
        raise HTTPException(status_code=500, detail="Query processing failed")
    
    # Store assistant response
    await memory.add_message("assistant", answer)
    
    # TODO: Extract citations from agent output
    # For now, return empty citations list
    citations = []
    
    return answer, citations


async def _stream_response(
    request_id: str,
    model: str,
    answer: str,
    created_at: int,
) -> AsyncGenerator[str, None]:
    """
    Stream response chunks in SSE format.
    
    Format: data: {json}\n\n
    """
    # Send initial chunk with role
    chunk = OpenAIChatStreamChunk(
        id=request_id,
        created=created_at,
        model=model,
        choices=[
            OpenAIChatStreamChoice(
                index=0,
                delta=OpenAIChatStreamDelta(role="assistant", content=""),
                finish_reason=None,
            )
        ],
    )
    yield f"data: {chunk.model_dump_json()}\n\n"
    
    # Stream content in chunks (simulate streaming for now)
    words = answer.split()
    for i in range(0, len(words), 5):
        chunk_text = " ".join(words[i : i + 5])
        if i > 0:
            chunk_text = " " + chunk_text
        
        chunk = OpenAIChatStreamChunk(
            id=request_id,
            created=created_at,
            model=model,
            choices=[
                OpenAIChatStreamChoice(
                    index=0,
                    delta=OpenAIChatStreamDelta(content=chunk_text),
                    finish_reason=None,
                )
            ],
        )
        yield f"data: {chunk.model_dump_json()}\n\n"
        await asyncio.sleep(0.02)  # Small delay for streaming effect
    
    # Send final chunk with finish reason
    chunk = OpenAIChatStreamChunk(
        id=request_id,
        created=created_at,
        model=model,
        choices=[
            OpenAIChatStreamChoice(
                index=0,
                delta=OpenAIChatStreamDelta(content=""),
                finish_reason="stop",
            )
        ],
    )
    yield f"data: {chunk.model_dump_json()}\n\n"
    
    # Send [DONE] marker
    yield "data: [DONE]\n\n"


@router.post("/v1/chat/completions")
async def chat_completions(request: OpenAIChatRequest):
    """
    OpenAI-compatible chat completions endpoint.
    
    Supports both streaming and non-streaming modes.
    """
    if not request.messages:
        raise HTTPException(status_code=400, detail="Messages list cannot be empty")
    
    # Extract user query (last user message)
    user_messages = [m for m in request.messages if m.role == "user"]
    if not user_messages:
        raise HTTPException(status_code=400, detail="No user message found")
    
    query = user_messages[-1].content
    session_id = _generate_session_id(request.messages)
    
    # Default to configured LLM model if not specified
    model = request.model or settings.LLM_MODEL
    
    logger.info(
        "Chat request received",
        model=model,
        stream=request.stream,
        session_id=session_id,
    )
    
    # Process query
    try:
        answer, citations = await _process_query(query, session_id)
    except Exception as e:
        logger.error("Query processing failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))
    
    # Generate response
    request_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
    created_at = int(time.time())
    
    if request.stream:
        # Streaming response
        return StreamingResponse(
            _stream_response(request_id, model, answer, created_at),
            media_type="text/event-stream",
        )
    else:
        # Non-streaming response
        response = OpenAIChatResponse(
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
        
        return response
