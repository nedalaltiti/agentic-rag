"""Semantic cache for fast RAG responses."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

import structlog
from pgvector.sqlalchemy import Vector
from sqlalchemy import Integer, bindparam, text

from agentic_rag.backend.rag.query_embedding import get_query_embedding
from agentic_rag.core.config import settings
from agentic_rag.core.database import AsyncSessionLocal
from agentic_rag.core.schemas import Citation

logger = structlog.get_logger()


@dataclass
class CachedResponse:
    answer: str
    citations: list[Citation]


def _parse_citations(payload: Any) -> list[Citation]:
    if not isinstance(payload, list):
        return []
    citations: list[Citation] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        try:
            citations.append(Citation(**item))
        except Exception:
            continue
    return citations


async def lookup_cache(query: str) -> CachedResponse | None:
    """Lookup a cached response by semantic similarity."""
    if not settings.SEMANTIC_CACHE_ENABLED:
        return None

    cleaned = query.strip()
    if not cleaned:
        return None

    try:
        embedding = await get_query_embedding(cleaned)
    except Exception as e:
        logger.warning("Semantic cache embedding failed", error=str(e))
        return None

    similarity_filter = ""
    params: dict[str, Any] = {
        "embed": embedding,
        "limit": 1,
        "index_version": settings.INDEX_VERSION,
        "embedding_model": settings.EMBEDDING_MODEL,
        "embedding_dimension": settings.EMBEDDING_DIMENSION,
    }
    if settings.SEMANTIC_CACHE_MIN_SIMILARITY is not None:
        similarity_filter = "AND (1 - (query_embedding <=> :embed)) >= :min_similarity"
        params["min_similarity"] = settings.SEMANTIC_CACHE_MIN_SIMILARITY

    stmt = text(f"""
        SELECT answer, citations, (1 - (query_embedding <=> :embed)) AS similarity
        FROM semantic_cache
        WHERE index_version = :index_version
          AND embedding_model = :embedding_model
          AND embedding_dimension = :embedding_dimension
          AND expires_at > NOW()
          {similarity_filter}
        ORDER BY query_embedding <=> :embed
        LIMIT :limit
    """).bindparams(
        bindparam("embed", type_=Vector(settings.EMBEDDING_DIMENSION)),
        bindparam("limit", type_=Integer),
        bindparam("index_version"),
        bindparam("embedding_model"),
        bindparam("embedding_dimension"),
    )
    if "min_similarity" in params:
        stmt = stmt.bindparams(bindparam("min_similarity"))

    async with AsyncSessionLocal() as session:
        result = await session.execute(stmt, params)
        row = result.mappings().first()

    if not row:
        return None

    citations = _parse_citations(row.get("citations"))
    answer = row.get("answer") or ""
    if not answer:
        return None

    logger.info("Semantic cache hit", similarity=row.get("similarity"))
    return CachedResponse(answer=answer, citations=citations)


async def store_cache(query: str, answer: str, citations: list[Citation]) -> None:
    """Store a response in the semantic cache."""
    if not settings.SEMANTIC_CACHE_ENABLED:
        return

    ttl = settings.SEMANTIC_CACHE_TTL_SECONDS
    if ttl <= 0:
        return

    cleaned = query.strip()
    if not cleaned or not answer.strip():
        return

    try:
        embedding = await get_query_embedding(cleaned)
    except Exception as e:
        logger.warning("Semantic cache embedding failed", error=str(e))
        return

    expires_at = datetime.now(UTC) + timedelta(seconds=ttl)
    citations_payload = [c.model_dump(mode="json") for c in citations]

    delete_stmt = text("DELETE FROM semantic_cache WHERE expires_at <= NOW()")
    insert_stmt = text(
        """
        INSERT INTO semantic_cache (
            query_text,
            query_embedding,
            answer,
            citations,
            embedding_model,
            embedding_dimension,
            index_version,
            expires_at
        ) VALUES (
            :query_text,
            :query_embedding,
            :answer,
            :citations,
            :embedding_model,
            :embedding_dimension,
            :index_version,
            :expires_at
        )
        """
    ).bindparams(bindparam("query_embedding", type_=Vector(settings.EMBEDDING_DIMENSION)))

    async with AsyncSessionLocal() as session:
        try:
            await session.execute(delete_stmt)
            await session.execute(
                insert_stmt,
                {
                    "query_text": cleaned,
                    "query_embedding": embedding,
                    "answer": answer,
                    "citations": citations_payload,
                    "embedding_model": settings.EMBEDDING_MODEL,
                    "embedding_dimension": settings.EMBEDDING_DIMENSION,
                    "index_version": settings.INDEX_VERSION,
                    "expires_at": expires_at,
                },
            )
            await session.commit()
        except Exception as e:
            await session.rollback()
            logger.warning("Failed to store semantic cache", error=str(e))
