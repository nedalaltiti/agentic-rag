"""Hybrid retrieval combining semantic and keyword search.

Uses parallel pgvector cosine search and PostgreSQL full-text search,
fused via Reciprocal Rank Fusion (RRF) with type-safe SQL bindings.
"""

import asyncio
from typing import Dict, List

from sqlalchemy import Integer, bindparam, text
from pgvector.sqlalchemy import Vector
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode
import structlog

from agentic_rag.shared.config import settings
from agentic_rag.shared.constants import EMBEDDING_DIMENSION
from agentic_rag.shared.database import AsyncSessionLocal
from agentic_rag.shared.llm_factory import get_embedding_model

logger = structlog.get_logger()


class HybridRetriever(BaseRetriever):
    """Custom Hybrid Retriever using parallel execution and type-safe vector binding."""

    # RRF Tuning Constants
    RRF_K = 60
    WEIGHT_VECTOR = 1.0
    WEIGHT_KEYWORD = 1.0

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

    def _retrieve(self, query_bundle) -> List[NodeWithScore]:
        """Synchronous wrapper (Not supported in async FastAPI)."""
        raise NotImplementedError("Use 'aretrieve' for async execution in FastAPI.")

    async def _aretrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        query = query_bundle.query_str
        logger.info("Starting Hybrid Search", query=query)

        # 1. Generate Query Embedding (instruction-prefixed for Qwen3)
        instruct_query = (
            "Instruct: Given a search query, retrieve relevant passages that answer the query.\n"
            f"Query: {query}"
        )
        query_embedding = await self.embed_model.aget_query_embedding(instruct_query)

        # 2. Execute Parallel Searches
        async with AsyncSessionLocal() as s1, AsyncSessionLocal() as s2:
            results = await asyncio.gather(
                self._vector_search(s1, query_embedding),
                self._keyword_search(s2, query),
            )
            vector_rows, keyword_rows = results

        # 3. Reciprocal Rank Fusion (RRF)
        fused_scores: Dict[str, float] = {}
        node_map: Dict[str, dict] = {}

        def process_rows(rows, weight=1.0):
            for rank, row in enumerate(rows):
                row_id = str(row.id)
                # RRF Formula: 1 / (k + rank)
                score = 1.0 / (self.RRF_K + rank)
                fused_scores[row_id] = fused_scores.get(row_id, 0.0) + (score * weight)

                if row_id not in node_map:
                    meta = dict(row.metadata) if row.metadata else {}
                    meta["document_id"] = str(row.document_id)

                    node_map[row_id] = {
                        "content": row.content,
                        "metadata": meta,
                    }

        process_rows(vector_rows, weight=self.WEIGHT_VECTOR)
        process_rows(keyword_rows, weight=self.WEIGHT_KEYWORD)

        # Sort by fused score
        sorted_ids = sorted(
            fused_scores.keys(), key=lambda x: fused_scores[x], reverse=True
        )
        final_ids = sorted_ids[: self.top_k]

        # Convert to LlamaIndex Nodes
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

    async def _vector_search(self, session, embedding: List[float]):
        """Semantic search with type-safe bindings."""
        # Safe: filter_clause is always "" or the _FILTER_TOC_FM class constant
        filter_clause = "" if self.include_toc else self._FILTER_TOC_FM
        stmt = text(f"""
            SELECT id, document_id, content, metadata
            FROM chunks
            WHERE 1=1 {filter_clause}
            ORDER BY embedding <=> :embed
            LIMIT :limit
        """).bindparams(
            bindparam("embed", type_=Vector(EMBEDDING_DIMENSION)),
            bindparam("limit", type_=Integer),
        )

        result = await session.execute(
            stmt, {"embed": embedding, "limit": self.top_k * 2}
        )
        return result.fetchall()

    async def _keyword_search(self, session, query: str):
        """Lexical search with type-safe bindings."""
        # Safe: filter_clause is always "" or the _FILTER_TOC_FM class constant
        filter_clause = "" if self.include_toc else self._FILTER_TOC_FM
        stmt = text(f"""
            SELECT id, document_id, content, metadata
            FROM chunks
            WHERE content_tsv @@ websearch_to_tsquery('simple', :query)
            {filter_clause}
            ORDER BY ts_rank(content_tsv, websearch_to_tsquery('simple', :query)) DESC
            LIMIT :limit
        """).bindparams(
            bindparam("query"),
            bindparam("limit", type_=Integer),
        )

        result = await session.execute(
            stmt, {"query": query, "limit": self.top_k * 2}
        )
        return result.fetchall()
