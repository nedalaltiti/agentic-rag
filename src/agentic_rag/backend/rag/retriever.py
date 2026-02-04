"""Hybrid retrieval combining semantic and keyword search via RRF."""

import asyncio
import time
from collections import OrderedDict

import structlog
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode
from pgvector.sqlalchemy import Vector
from sqlalchemy import Integer, bindparam, text
from sqlalchemy.exc import SQLAlchemyError

from agentic_rag.backend.rag.index_guard import ensure_index_compatible
from agentic_rag.backend.rag.query_embedding import build_query_embedding_text
from agentic_rag.core.config import settings
from agentic_rag.core.database import AsyncSessionLocal
from agentic_rag.core.exceptions import DependencyUnavailable
from agentic_rag.core.llm_factory import get_embedding_model

logger = structlog.get_logger()


class HybridRetriever(BaseRetriever):
    """Custom Hybrid Retriever using parallel execution and type-safe vector binding."""

    RRF_K = 60
    _EMBED_CACHE_MAX = 128
    _embedding_cache: "OrderedDict[str, tuple[list[float], float]]" = OrderedDict()
    # Filter out TOC and front-matter chunks by default
    _FILTER_TOC_FM = """
        AND COALESCE((metadata->>'is_toc')::boolean, false) = false
        AND COALESCE((metadata->>'is_front_matter')::boolean, false) = false
    """

    def __init__(self, include_toc: bool = False):
        super().__init__()
        self.embed_model = get_embedding_model()
        self.top_k = settings.TOP_K_RETRIEVAL
        self.include_toc = include_toc

    def _retrieve(self, query_bundle) -> list[NodeWithScore]:
        """Synchronous wrapper (Not supported in async FastAPI)."""
        raise NotImplementedError("Use 'aretrieve' for async execution in FastAPI.")

    async def _aretrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        query = query_bundle.query_str
        logger.info("Starting Hybrid Search", query=query)
        await ensure_index_compatible()
        return await self._aretrieve_single(query)

    async def _aretrieve_single(self, query: str) -> list[NodeWithScore]:
        """Single-query hybrid retrieval (vector + keyword with RRF)."""
        query_embedding = await self._get_query_embedding(query)

        async with AsyncSessionLocal() as s1, AsyncSessionLocal() as s2:
            results = await asyncio.gather(
                self._vector_search(s1, query_embedding),
                self._keyword_search(s2, query),
            )
            vector_rows, keyword_rows = results

        fused_scores: dict[str, float] = {}
        node_map: dict[str, dict] = {}

        def process_rows(rows, weight=1.0):
            for rank, row in enumerate(rows):
                row_id = str(row.id)
                score = 1.0 / (self.RRF_K + rank)
                fused_scores[row_id] = fused_scores.get(row_id, 0.0) + (score * weight)

                if row_id not in node_map:
                    meta = dict(row.metadata) if row.metadata else {}
                    meta["document_id"] = str(row.document_id)

                    node_map[row_id] = {
                        "content": row.content,
                        "metadata": meta,
                    }

        process_rows(vector_rows, weight=settings.RRF_WEIGHT_VECTOR)
        process_rows(keyword_rows, weight=settings.RRF_WEIGHT_KEYWORD)

        sorted_ids = sorted(fused_scores.keys(), key=lambda x: fused_scores[x], reverse=True)
        # Keep a wider candidate set for reranking, but only if configured.
        candidate_k = max(self.top_k, settings.TOP_K_RERANK)
        final_ids = sorted_ids[:candidate_k]

        nodes = []
        for vid in final_ids:
            data = node_map[vid]
            node = TextNode(
                text=data["content"],
                metadata=data["metadata"],
                id_=vid,
            )
            nodes.append(NodeWithScore(node=node, score=fused_scores[vid]))

        logger.info(
            "Hybrid Retrieval Complete",
            vector_candidates=len(vector_rows),
            keyword_candidates=len(keyword_rows),
            final_results=len(nodes),
        )

        return nodes

    async def _vector_search(self, session, embedding: list[float]):
        """Cosine search via pgvector <=> (matches vector_cosine_ops HNSW index)."""
        if settings.HNSW_EF_SEARCH is not None and settings.HNSW_EF_SEARCH > 0:
            await session.execute(
                text("SET LOCAL hnsw.ef_search = :val"),
                {"val": settings.HNSW_EF_SEARCH},
            )
        filter_clause = "" if self.include_toc else self._FILTER_TOC_FM
        distance_filter = ""
        params = {
            "embed": embedding,
            "limit": self.top_k * 2,
            "index_version": settings.INDEX_VERSION,
            "embedding_model": settings.EMBEDDING_MODEL,
            "embedding_dimension": settings.EMBEDDING_DIMENSION,
        }
        if settings.VECTOR_MIN_SIMILARITY is not None:
            max_distance = 1.0 - settings.VECTOR_MIN_SIMILARITY
            distance_filter = "AND (embedding <=> :embed) <= :max_distance"
            params["max_distance"] = max_distance
        stmt = text(f"""
            SELECT id, document_id, content, metadata
            FROM chunks
            WHERE index_version = :index_version
              AND embedding_model = :embedding_model
              AND embedding_dimension = :embedding_dimension
              {filter_clause}
              {distance_filter}
            ORDER BY embedding <=> :embed
            LIMIT :limit
        """).bindparams(
            bindparam("embed", type_=Vector(settings.EMBEDDING_DIMENSION)),
            bindparam("limit", type_=Integer),
            bindparam("index_version"),
            bindparam("embedding_model"),
            bindparam("embedding_dimension"),
        )
        if "max_distance" in params:
            stmt = stmt.bindparams(bindparam("max_distance"))

        try:
            result = await session.execute(stmt, params)
            return result.fetchall()
        except SQLAlchemyError as e:
            raise DependencyUnavailable(
                "database",
                "vector search failed",
                {"error": str(e)},
            ) from e

    async def _keyword_search(self, session, query: str):
        """Lexical search with type-safe bindings."""
        filter_clause = "" if self.include_toc else self._FILTER_TOC_FM
        score_filter = ""
        params = {
            "query": query,
            "limit": self.top_k * 2,
            "index_version": settings.INDEX_VERSION,
            "embedding_model": settings.EMBEDDING_MODEL,
            "embedding_dimension": settings.EMBEDDING_DIMENSION,
        }
        if settings.KEYWORD_MIN_SCORE is not None:
            score_filter = (
                "AND ts_rank(content_tsv, websearch_to_tsquery('simple', :query)) "
                ">= :min_score"
            )
            params["min_score"] = settings.KEYWORD_MIN_SCORE
        stmt = text(f"""
            SELECT id, document_id, content, metadata
            FROM chunks
            WHERE content_tsv @@ websearch_to_tsquery('simple', :query)
              AND index_version = :index_version
              AND embedding_model = :embedding_model
              AND embedding_dimension = :embedding_dimension
            {filter_clause}
            {score_filter}
            ORDER BY ts_rank(content_tsv, websearch_to_tsquery('simple', :query)) DESC
            LIMIT :limit
        """).bindparams(
            bindparam("query"),
            bindparam("limit", type_=Integer),
            bindparam("index_version"),
            bindparam("embedding_model"),
            bindparam("embedding_dimension"),
        )
        if "min_score" in params:
            stmt = stmt.bindparams(bindparam("min_score"))
        try:
            result = await session.execute(stmt, params)
            return result.fetchall()
        except SQLAlchemyError as e:
            raise DependencyUnavailable(
                "database",
                "keyword search failed",
                {"error": str(e)},
            ) from e

    async def _get_query_embedding(self, query: str) -> list[float]:
        """Best-effort LRU cache for query embeddings."""
        query_text = build_query_embedding_text(query)
        cached = self._embedding_cache.get(query_text)
        ttl = settings.QUERY_EMBED_CACHE_TTL
        now = time.monotonic()
        if cached is not None:
            embedding, ts = cached
            if ttl <= 0 or (now - ts) <= ttl:
                self._embedding_cache.move_to_end(query_text)
                return embedding
            self._embedding_cache.pop(query_text, None)

        try:
            embedding = await self.embed_model.aget_query_embedding(query_text)
        except Exception as e:
            raise DependencyUnavailable(
                "ollama", "embedding generation failed", {"error": str(e)}
            ) from e
        self._embedding_cache[query_text] = (embedding, now)
        if len(self._embedding_cache) > self._EMBED_CACHE_MAX:
            self._embedding_cache.popitem(last=False)
        return embedding
