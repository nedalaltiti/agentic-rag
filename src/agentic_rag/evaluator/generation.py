"""Synthetic test set generation from DB chunks.

Queries random chunks from PostgreSQL and uses the local LLM
to generate question/ground-truth pairs for RAGAS evaluation.
"""

from __future__ import annotations

import asyncio
import json
import random
from dataclasses import dataclass
from typing import Any

import structlog
from sqlalchemy import text

from agentic_rag.shared.database import AsyncSessionLocal
from agentic_rag.shared.llm_factory import get_llm

logger = structlog.get_logger()


@dataclass
class TestSample:
    question: str
    ground_truth: str
    document_id: str
    chunk_id: str
    file_name: str
    section_path: str | None
    context: str


_QA_PROMPT = """You are generating evaluation data for a RAG system.

Given this context snippet from a document, write:
1) A specific question that can be answered ONLY from this snippet.
2) A short ground-truth answer (1-3 sentences) strictly grounded in the snippet.

Return ONLY valid JSON with keys:
- question
- ground_truth

Context snippet:
---
{context}
---
"""


def _safe_json_loads(text_str: str) -> dict[str, Any] | None:
    """Best-effort extraction of the first JSON object from LLM output."""
    start = text_str.find("{")
    end = text_str.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        return dict(json.loads(text_str[start : end + 1]))
    except Exception:
        return None


async def _fetch_random_chunks(limit: int = 50) -> list[dict[str, Any]]:
    """Pull random chunks from DB with metadata needed for evaluation."""
    sql = text(
        """
        SELECT
          c.id as chunk_id,
          c.document_id as document_id,
          d.file_name as file_name,
          c.metadata->>'section_path' as section_path,
          c.content as content
        FROM chunks c
        JOIN documents d ON d.id = c.document_id
        WHERE COALESCE((c.metadata->>'is_toc')::boolean, false) = false
          AND COALESCE((c.metadata->>'is_front_matter')::boolean, false) = false
        ORDER BY random()
        LIMIT :limit
        """
    )
    async with AsyncSessionLocal() as session:
        res = await session.execute(sql, {"limit": limit})
        rows = res.mappings().all()
        return [dict(r) for r in rows]


async def generate_synthetic_testset(
    num_samples: int,
    output_path: str,
    seed: int = 42,
    max_context_chars: int = 1500,
) -> list[TestSample]:
    """Generate synthetic Q/A pairs from random chunks using the local LLM."""
    random.seed(seed)
    llm = get_llm()

    # Pull more candidates than needed to tolerate LLM failures
    candidates = await _fetch_random_chunks(limit=max(50, num_samples * 3))
    random.shuffle(candidates)

    samples: list[TestSample] = []
    for row in candidates:
        if len(samples) >= num_samples:
            break

        context = (row["content"] or "").strip()
        if not context:
            continue

        context = context[:max_context_chars]
        prompt = _QA_PROMPT.format(context=context)

        try:
            resp = await llm.acomplete(prompt)
            data = _safe_json_loads(resp.text or "")
            if not data:
                logger.warning("Failed to parse JSON from generator", file=row["file_name"])
                continue

            q = (data.get("question") or "").strip()
            gt = (data.get("ground_truth") or "").strip()
            if not q or not gt:
                continue

            samples.append(
                TestSample(
                    question=q,
                    ground_truth=gt,
                    document_id=str(row["document_id"]),
                    chunk_id=str(row["chunk_id"]),
                    file_name=str(row["file_name"]),
                    section_path=row.get("section_path"),
                    context=context,
                )
            )
        except Exception as e:
            logger.warning("Generator LLM failed", error=str(e), file=row["file_name"])
            continue

    out = [s.__dict__ for s in samples]
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    logger.info("Synthetic testset created", samples=len(samples), path=output_path)
    return samples


def generate_sync(num_samples: int, output_path: str, seed: int = 42) -> None:
    """Synchronous entry point for the CLI."""
    asyncio.run(
        generate_synthetic_testset(num_samples=num_samples, output_path=output_path, seed=seed)
    )
