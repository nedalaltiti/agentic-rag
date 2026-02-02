"""LLM and Embedding model factory using Ollama."""

import json as json_mod
from collections.abc import AsyncGenerator

import httpx
import structlog
from llama_index.core import Settings as LlamaIndexSettings
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama

from .config import settings

logger = structlog.get_logger()


def get_llm() -> Ollama:
    """Return the configured Ollama LLM instance."""
    return Ollama(
        model=settings.LLM_MODEL,
        base_url=settings.OLLAMA_BASE_URL,
        request_timeout=300.0,
        temperature=0.1,
        context_window=8192,
    )


def get_eval_llm() -> Ollama:
    """Return an Ollama LLM instance configured for evaluation (separate model)."""
    logger.info("Using evaluator model", model=settings.EVAL_MODEL)
    return Ollama(
        model=settings.EVAL_MODEL,
        base_url=settings.OLLAMA_BASE_URL,
        request_timeout=600.0,
        temperature=0.0,
        context_window=8192,
    )


def get_embedding_model() -> OllamaEmbedding:
    """Return the configured Ollama Embedding instance."""
    return OllamaEmbedding(
        model_name=settings.EMBEDDING_MODEL,
        base_url=settings.OLLAMA_BASE_URL,
    )


async def ollama_chat_with_thinking(
    system_prompt: str,
    user_message: str,
    think: bool = True,
) -> tuple[str, str]:
    """Call Ollama chat API directly with thinking support.

    Returns (thinking_text, content_text).
    """
    payload = {
        "model": settings.LLM_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        "stream": False,
        "think": think,
        "options": {
            "temperature": 0.1,
            "num_ctx": 8192,
        },
    }

    async with httpx.AsyncClient(timeout=300.0) as client:
        resp = await client.post(
            f"{settings.OLLAMA_BASE_URL}/api/chat",
            json=payload,
        )
        resp.raise_for_status()
        data = resp.json()

    msg = data.get("message", {})
    thinking = msg.get("thinking", "") or ""
    content = msg.get("content", "") or ""
    return thinking, content


async def ollama_chat_stream(
    system_prompt: str,
    user_message: str,
    think: bool = True,
) -> AsyncGenerator[dict, None]:
    """Stream from Ollama chat API with thinking support.

    Yields dicts with keys: 'thinking' (str|None), 'content' (str|None), 'done' (bool).
    """
    payload = {
        "model": settings.LLM_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        "stream": True,
        "think": think,
        "options": {
            "temperature": 0.1,
            "num_ctx": 8192,
        },
    }

    async with httpx.AsyncClient(timeout=300.0) as client:
        async with client.stream(
            "POST",
            f"{settings.OLLAMA_BASE_URL}/api/chat",
            json=payload,
        ) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.strip():
                    continue
                chunk = json_mod.loads(line)
                msg = chunk.get("message", {})
                yield {
                    "thinking": msg.get("thinking", None),
                    "content": msg.get("content", None),
                    "done": chunk.get("done", False),
                }


def configure_global_settings() -> None:
    """Set LlamaIndex global defaults. Call once at startup."""
    logger.info(
        "Configuring LlamaIndex Global Settings",
        llm=settings.LLM_MODEL,
        embed=settings.EMBEDDING_MODEL,
    )

    LlamaIndexSettings.llm = get_llm()
    LlamaIndexSettings.embed_model = get_embedding_model()
    LlamaIndexSettings.chunk_size = settings.CHUNK_SIZE
    LlamaIndexSettings.chunk_overlap = settings.CHUNK_OVERLAP
