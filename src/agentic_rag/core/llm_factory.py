"""LLM and Embedding model factory using Ollama."""

from __future__ import annotations

import json as json_mod
from collections.abc import AsyncGenerator, Callable

import httpx
import structlog
from llama_index.core import Settings as LlamaIndexSettings
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama

from .config import settings

logger = structlog.get_logger()


def get_llm(request_timeout: float = 300.0) -> Ollama:
    """Return the configured Ollama LLM instance."""
    return Ollama(
        model=settings.LLM_MODEL,
        base_url=settings.OLLAMA_BASE_URL,
        request_timeout=request_timeout,
        temperature=0.0,
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


def get_tokenizer() -> Callable[[str], list[int]]:
    """Return a tokenizer callable for SentenceSplitter chunk sizing."""
    if hasattr(get_tokenizer, "_cached"):
        cached: Callable[[str], list[int]] = get_tokenizer._cached  # type: ignore[attr-defined]
        return cached

    backend = settings.TOKENIZER.strip()

    if backend.startswith("hf:"):
        repo = backend[3:].strip()
        if not repo:
            logger.warning(
                "TOKENIZER='hf:' with empty repo, falling back to default",
            )
        else:
            try:
                from transformers import AutoTokenizer  # type: ignore[import-untyped]

                hf_tok = AutoTokenizer.from_pretrained(
                    repo, trust_remote_code=True, local_files_only=True,
                )

                def _hf_encode(text: str) -> list[int]:
                    result: list[int] = hf_tok.encode(
                        text, add_special_tokens=False,
                    )
                    return result

                get_tokenizer._cached = _hf_encode  # type: ignore[attr-defined]
                logger.info(
                    "Loaded HuggingFace tokenizer for chunking",
                    repo=repo,
                )
                return _hf_encode
            except Exception:
                logger.warning(
                    "HuggingFace tokenizer not found locally, "
                    "falling back to cl100k_base",
                    repo=repo,
                )

    # Default: LlamaIndex built-in tokenizer (tiktoken cl100k_base)
    from llama_index.core.utils import get_tokenizer as _llama_get_tokenizer

    tok = _llama_get_tokenizer()
    get_tokenizer._cached = tok  # type: ignore[attr-defined]
    logger.info("Using default tokenizer (cl100k_base) for chunking")
    return tok


async def ollama_chat_with_thinking(
    system_prompt: str,
    user_message: str,
    think: bool = True,
    model: str | None = None,
) -> tuple[str, str]:
    """Call Ollama chat API directly with thinking support.

    Returns (thinking_text, content_text).
    """
    payload = {
        "model": model or settings.LLM_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        "stream": False,
        "think": think,
        "options": {
            "temperature": 0.0,
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
    model: str | None = None,
) -> AsyncGenerator[dict, None]:
    """Stream from Ollama chat API with thinking support.

    Yields dicts with keys: 'thinking' (str|None), 'content' (str|None), 'done' (bool).
    """
    payload = {
        "model": model or settings.LLM_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        "stream": True,
        "think": think,
        "options": {
            "temperature": 0.0,
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


def validate_embedding_dimension() -> None:
    """Embed a short probe and check it matches EMBEDDING_DIMENSION.

    Call once at startup (e.g. during ingestion) to catch model/config
    mismatches before writing bad vectors to the DB.
    """
    import asyncio

    embed_model = get_embedding_model()

    async def _probe() -> int:
        vec = await embed_model.aget_text_embedding("dimension probe")
        return len(vec)

    try:
        asyncio.get_running_loop()
        # Already inside an async context — run in a new thread to avoid nesting.
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            actual = pool.submit(asyncio.run, _probe()).result()
    except RuntimeError:
        # No running loop — safe to use asyncio.run directly.
        actual = asyncio.run(_probe())
    expected = settings.EMBEDDING_DIMENSION
    if actual != expected:
        raise ValueError(
            f"EMBEDDING_DIMENSION={expected} but {settings.EMBEDDING_MODEL} "
            f"produces {actual}-d vectors. Update EMBEDDING_DIMENSION or "
            f"the DB schema (migrations/002)."
        )
    logger.info(
        "Embedding dimension validated",
        model=settings.EMBEDDING_MODEL,
        dimension=actual,
    )


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
