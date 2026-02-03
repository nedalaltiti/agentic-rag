"""RAGAS metrics evaluation for the RAG pipeline."""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any

import pandas as pd
import structlog

from agentic_rag.backend.rag.reranker import LLMReranker
from agentic_rag.backend.rag.retriever import HybridRetriever
from agentic_rag.core.citations import format_citations
from agentic_rag.core.config import settings
from agentic_rag.core.llm_factory import get_embedding_model, get_eval_llm, get_llm
from agentic_rag.core.prompts import PromptRegistry
from agentic_rag.core.schemas import Citation

logger = structlog.get_logger()


@dataclass
class EvalResult:
    overall: dict[str, float]
    per_sample: pd.DataFrame


def _get_git_commit() -> str | None:
    try:
        cwd = Path(__file__).resolve().parent
        output = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=cwd, text=True, timeout=2
        )
        return output.strip() or None
    except Exception:
        return None


def _get_pkg_version(name: str) -> str | None:
    try:
        return importlib_metadata.version(name)
    except Exception:
        return None


def _hash_file(path: str) -> str | None:
    try:
        import hashlib

        digest = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except Exception:
        return None


def _format_context(
    citations: list[Citation], max_chunks: int = 5, max_chars_each: int = 1200
) -> str:
    """Format citations into a readable context block for the LLM."""
    if not citations:
        return "No relevant context found."

    parts = []
    for i, c in enumerate(citations[:max_chunks], 1):
        loc = []
        if c.section_path:
            loc.append(c.section_path)
        if c.page_number is not None:
            loc.append(f"p.{c.page_number}")
        loc_str = " | " + ", ".join(loc) if loc else ""
        parts.append(f"[{i}] {c.file_name}{loc_str}\n{c.chunk_text[:max_chars_each]}")
    return "\n\n".join(parts)


async def _answer_with_fast_rag(question: str, citations: list[Citation]) -> str:
    """Single-call answer generation using the user_prompt template."""
    llm = get_llm(request_timeout=float(settings.EVAL_TIMEOUT))
    context = _format_context(citations)

    prompt = ""
    try:
        prompt = PromptRegistry.render("user_prompt", query=question, context=context)
    except Exception:
        prompt = ""

    if not prompt.strip():
        prompt = (
            "Answer the question using ONLY the provided context. "
            "If the context does not contain the answer, say you could not find it.\n\n"
            f"Question: {question}\n\nContext:\n{context}\n"
        )

    resp = await llm.acomplete(prompt)
    return (resp.text or "").strip()


async def evaluate_rag_pipeline(
    testset_path: str,
    output_path: str,
    use_reranker: bool = True,
    rerank_candidates: int = 10,
) -> EvalResult:
    """
    Load testset, run retrieval+answer pipeline per sample, evaluate with RAGAS.

    RAGAS expects a dataset shaped: {question, answer, contexts, ground_truth}.
    """
    with open(testset_path, encoding="utf-8") as f:
        testset = json.load(f)

    questions: list[str] = []
    answers: list[str] = []
    contexts: list[list[str]] = []
    ground_truths: list[str] = []

    retriever = HybridRetriever(include_toc=False)

    reranker = None
    if use_reranker:
        reranker = LLMReranker()
        reranker._semaphore = asyncio.Semaphore(1)

    for item in testset:
        q = item["question"]
        gt = item["ground_truth"]

        nodes = await retriever.aretrieve(q)
        if use_reranker and reranker and nodes:
            nodes = await reranker.rerank(q, nodes[:rerank_candidates])
        else:
            nodes = nodes[:5]

        cits = format_citations(nodes)
        ans = await _answer_with_fast_rag(q, cits)

        ctx_list = [c.chunk_text for c in cits] if cits else []

        questions.append(q)
        answers.append(ans)
        contexts.append(ctx_list)
        ground_truths.append(gt)

    df = pd.DataFrame(
        {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths,
        }
    )

    from datasets import Dataset
    from ragas import evaluate
    from ragas.embeddings import LlamaIndexEmbeddingsWrapper
    from ragas.llms import LlamaIndexLLMWrapper
    from ragas.metrics import (
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
    )
    from ragas.run_config import RunConfig

    eval_llm = get_eval_llm()
    logger.info("Evaluator model: %s", settings.EVAL_MODEL)
    evaluator_llm = LlamaIndexLLMWrapper(eval_llm)
    evaluator_emb = LlamaIndexEmbeddingsWrapper(get_embedding_model())

    ds = Dataset.from_pandas(df)

    # Ollama processes one request at a time — sequential execution prevents timeouts
    run_config = RunConfig(
        max_workers=settings.EVAL_MAX_WORKERS,
        timeout=settings.EVAL_TIMEOUT,
        max_retries=3,
        max_wait=120,
    )

    result = evaluate(
        ds,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=evaluator_llm,
        embeddings=evaluator_emb,
        run_config=run_config,
    )

    try:
        per_sample_df = result.to_pandas()
    except Exception:
        per_sample_df = pd.DataFrame(result)

    overall: dict[str, Any] = {}
    try:
        overall = {k: float(v) for k, v in result.scores.items()}
    except Exception:
        numeric_cols = per_sample_df.select_dtypes(include="number").columns
        overall = {c: float(per_sample_df[c].mean()) for c in numeric_cols}

    packages = {
        "ragas": _get_pkg_version("ragas"),
        "llama_index": _get_pkg_version("llama-index"),
        "llama_index_core": _get_pkg_version("llama-index-core"),
    }
    packages = {k: v for k, v in packages.items() if v}

    metadata = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _get_git_commit(),
        "app_version": settings.APP_VERSION,
        "environment": settings.ENVIRONMENT,
        "phoenix_prompt_tag": settings.PHOENIX_PROMPT_TAG,
        "models": {
            "llm": settings.LLM_MODEL,
            "embedding": settings.EMBEDDING_MODEL,
            "eval": settings.EVAL_MODEL,
        },
        "packages": packages,
        "python_version": sys.version.split()[0],
        "testset_sha256": _hash_file(testset_path),
        "settings": {
            "chunk_size": settings.CHUNK_SIZE,
            "chunk_overlap": settings.CHUNK_OVERLAP,
            "top_k_retrieval": settings.TOP_K_RETRIEVAL,
            "top_k_rerank": settings.TOP_K_RERANK,
            "rrf_weight_vector": settings.RRF_WEIGHT_VECTOR,
            "rrf_weight_keyword": settings.RRF_WEIGHT_KEYWORD,
            "reranker_timeout": settings.RERANKER_TIMEOUT,
            "query_embed_cache_ttl": settings.QUERY_EMBED_CACHE_TTL,
            "eval_max_workers": settings.EVAL_MAX_WORKERS,
            "eval_timeout": settings.EVAL_TIMEOUT,
        },
    }

    payload = {
        "metadata": metadata,
        "overall": overall,
        "per_sample": per_sample_df.to_dict(orient="records"),
        "config": {
            "testset": testset_path,
            "use_reranker": use_reranker,
            "rerank_candidates": rerank_candidates,
        },
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    logger.info("Evaluation complete", output=output_path, overall=overall)
    return EvalResult(overall=overall, per_sample=per_sample_df)


def evaluate_sync(testset_path: str, output_path: str, use_reranker: bool = True) -> None:
    """Synchronous entry point for the CLI."""
    asyncio.run(
        evaluate_rag_pipeline(
            testset_path=testset_path,
            output_path=output_path,
            use_reranker=use_reranker,
        )
    )
