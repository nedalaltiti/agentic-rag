"""RAGAS metrics evaluation for the RAG pipeline."""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
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


def _extract_relevant_ids(item: dict[str, Any]) -> tuple[set[str], set[str]]:
    """Return relevant document_ids and chunk_ids (as strings) from a test item."""
    doc_ids: set[str] = set()
    chunk_ids: set[str] = set()

    doc_id = item.get("document_id")
    if doc_id:
        doc_ids.add(str(doc_id))

    doc_id_list = item.get("document_ids")
    if isinstance(doc_id_list, list):
        doc_ids.update(str(d) for d in doc_id_list if d)

    chunk_id = item.get("chunk_id")
    if chunk_id:
        chunk_ids.add(str(chunk_id))

    chunk_id_list = item.get("chunk_ids")
    if isinstance(chunk_id_list, list):
        chunk_ids.update(str(c) for c in chunk_id_list if c)

    return doc_ids, chunk_ids


def _compute_retrieval_metrics(
    retrieved_doc_ids: list[str],
    retrieved_chunk_ids: list[str],
    relevant_doc_ids: set[str],
    relevant_chunk_ids: set[str],
    k: int,
) -> dict[str, float]:
    """Compute Recall@K, Precision@K, MRR, NDCG@K, HitRate using doc IDs (fallback to chunk IDs)."""

    def _unique_preserve_order(values: list[str]) -> list[str]:
        seen: set[str] = set()
        unique: list[str] = []
        for v in values:
            if v in seen:
                continue
            seen.add(v)
            unique.append(v)
        return unique

    if k <= 0:
        return {
            "recall_at_k": 0.0,
            "precision_at_k": 0.0,
            "mrr": 0.0,
            "ndcg_at_k": 0.0,
            "hit_rate": 0.0,
        }

    # Prefer document-level relevance when available
    use_doc_level = bool(relevant_doc_ids)
    relevant = relevant_doc_ids if use_doc_level else relevant_chunk_ids
    retrieved_raw = retrieved_doc_ids if use_doc_level else retrieved_chunk_ids
    retrieved = _unique_preserve_order(retrieved_raw)

    if not relevant:
        return {
            "recall_at_k": 0.0,
            "precision_at_k": 0.0,
            "mrr": 0.0,
            "ndcg_at_k": 0.0,
            "hit_rate": 0.0,
        }

    top_k = retrieved[:k]
    hits = [1 if rid in relevant else 0 for rid in top_k]
    rel_in_top_k = sum(hits)

    recall = rel_in_top_k / max(len(relevant), 1)
    precision = rel_in_top_k / k
    hit_rate = 1.0 if rel_in_top_k > 0 else 0.0

    first_rank = next((i + 1 for i, r in enumerate(top_k) if r in relevant), None)
    mrr = 1.0 / first_rank if first_rank else 0.0

    # NDCG@K with binary relevance
    import math

    dcg = sum(rel / math.log2(idx + 2) for idx, rel in enumerate(hits))
    ideal_hits = [1] * min(len(relevant), k) + [0] * max(k - len(relevant), 0)
    idcg = sum(rel / math.log2(idx + 2) for idx, rel in enumerate(ideal_hits)) or 1.0
    ndcg = dcg / idcg

    return {
        "recall_at_k": float(recall),
        "precision_at_k": float(precision),
        "mrr": float(mrr),
        "ndcg_at_k": float(ndcg),
        "hit_rate": float(hit_rate),
    }


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
    skip_ragas: bool = False,
) -> EvalResult:
    """
    Load testset, run retrieval+answer pipeline per sample, evaluate with RAGAS.

    RAGAS expects a dataset shaped: {question, answer, contexts, ground_truth}.
    """
    with open(testset_path, encoding="utf-8") as f:
        testset = json.load(f)

    if not isinstance(testset, list):
        raise ValueError("Testset must be a list of samples")

    sample_limit = settings.EVAL_SAMPLE_SIZE
    if sample_limit and sample_limit > 0 and len(testset) > sample_limit:
        logger.info(
            "Truncating testset for evaluation",
            requested=sample_limit,
            total=len(testset),
        )
        testset = testset[:sample_limit]

    retrieval_k = settings.EVAL_RETRIEVAL_K

    questions: list[str] = []
    answers: list[str] = []
    contexts: list[list[str]] = []
    ground_truths: list[str] = []
    recall_at_k: list[float] = []
    precision_at_k: list[float] = []
    mrr: list[float] = []
    ndcg_at_k: list[float] = []
    hit_rate: list[float] = []

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

        if retrieval_k and retrieval_k > 0:
            nodes = nodes[:retrieval_k]

        cits = format_citations(nodes)
        if skip_ragas:
            ans = ""
        else:
            ans = await _answer_with_fast_rag(q, cits)

        ctx_list = [c.chunk_text for c in cits] if cits else []

        relevant_doc_ids, relevant_chunk_ids = _extract_relevant_ids(item)
        retrieved_doc_ids = [
            str((n.node.metadata or {}).get("document_id"))
            for n in nodes
            if (n.node.metadata or {}).get("document_id") is not None
        ]
        retrieved_chunk_ids = [str(n.node.node_id) for n in nodes]
        retrieval_metrics = _compute_retrieval_metrics(
            retrieved_doc_ids,
            retrieved_chunk_ids,
            relevant_doc_ids,
            relevant_chunk_ids,
            k=max(1, retrieval_k),
        )

        questions.append(q)
        answers.append(ans)
        contexts.append(ctx_list)
        ground_truths.append(gt)
        recall_at_k.append(retrieval_metrics["recall_at_k"])
        precision_at_k.append(retrieval_metrics["precision_at_k"])
        mrr.append(retrieval_metrics["mrr"])
        ndcg_at_k.append(retrieval_metrics["ndcg_at_k"])
        hit_rate.append(retrieval_metrics["hit_rate"])

    df = pd.DataFrame(
        {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths,
            "recall_at_k": recall_at_k,
            "precision_at_k": precision_at_k,
            "mrr": mrr,
            "ndcg_at_k": ndcg_at_k,
            "hit_rate": hit_rate,
        }
    )

    retrieval_cols = ["recall_at_k", "precision_at_k", "mrr", "ndcg_at_k", "hit_rate"]
    ragas_error: str | None = None
    if skip_ragas:
        per_sample_df = df.copy()
        overall = {c: float(per_sample_df[c].mean()) for c in retrieval_cols}
    else:
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

        metrics = [faithfulness, answer_relevancy, context_precision, context_recall]
        try:
            from ragas.metrics import answer_correctness

            metrics.append(answer_correctness)
        except Exception:
            logger.info("answer_correctness metric not available; skipping")

        try:
            result = evaluate(
                ds,
                metrics=metrics,
                llm=evaluator_llm,
                embeddings=evaluator_emb,
                run_config=run_config,
            )
        except Exception as e:
            logger.warning("RAGAS evaluation failed; falling back to retrieval-only", error=str(e))
            ragas_error = str(e)
            per_sample_df = df.copy()
            overall = {c: float(per_sample_df[c].mean()) for c in retrieval_cols}
            skip_ragas = True
            result = None

        if result is not None:
            try:
                per_sample_df = result.to_pandas()  # type: ignore[union-attr]
            except Exception:
                per_sample_df = pd.DataFrame(result)

            overall = {}
            try:
                overall = {k: float(v) for k, v in result.scores.items()}  # type: ignore[union-attr]
            except Exception:
                numeric_cols = per_sample_df.select_dtypes(include="number").columns
                overall = {c: float(per_sample_df[c].mean()) for c in numeric_cols}

        # Ensure retrieval metrics are included in overall summary
        for col in retrieval_cols:
            if col in per_sample_df.columns:
                overall[col] = float(per_sample_df[col].mean())

    packages = {
        "ragas": _get_pkg_version("ragas"),
        "llama_index": _get_pkg_version("llama-index"),
        "llama_index_core": _get_pkg_version("llama-index-core"),
    }
    packages = {k: v for k, v in packages.items() if v}

    metadata = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
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
            "eval_sample_size": settings.EVAL_SAMPLE_SIZE,
            "eval_retrieval_k": settings.EVAL_RETRIEVAL_K,
            "eval_max_workers": settings.EVAL_MAX_WORKERS,
            "eval_timeout": settings.EVAL_TIMEOUT,
        },
        "ragas_skipped": bool(skip_ragas),
        "ragas_error": ragas_error,
    }

    payload = {
        "metadata": metadata,
        "overall": overall,
        "per_sample": per_sample_df.to_dict(orient="records"),
        "config": {
            "testset": testset_path,
            "use_reranker": use_reranker,
            "rerank_candidates": rerank_candidates,
            "sample_size": len(testset),
            "retrieval_k": retrieval_k,
            "skip_ragas": bool(skip_ragas),
        },
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    logger.info("Evaluation complete", output=output_path, overall=overall)
    return EvalResult(overall=overall, per_sample=per_sample_df)


def evaluate_sync(
    testset_path: str,
    output_path: str,
    use_reranker: bool = True,
    skip_ragas: bool = False,
) -> None:
    """Synchronous entry point for the CLI."""
    asyncio.run(
        evaluate_rag_pipeline(
            testset_path=testset_path,
            output_path=output_path,
            use_reranker=use_reranker,
            skip_ragas=skip_ragas,
        )
    )
