"""LLM-based re-ranking of retrieved candidates.

Scores each candidate passage against the query using the LLM,
with concurrency-limited parallel evaluation and adaptive candidate counts.
"""

import asyncio
import re
from typing import List

from llama_index.core.schema import NodeWithScore
import structlog

from agentic_rag.shared.config import settings
from agentic_rag.shared.llm_factory import get_llm

logger = structlog.get_logger()


class LLMReranker:
    """Re-ranks retrieved nodes by LLM-scored relevance (0-10)."""

    def __init__(self):
        self.llm = get_llm()
        self.top_n = settings.TOP_K_RERANK
        # Semaphore limits concurrency to prevent LLM overload
        self._semaphore = asyncio.Semaphore(5)

    async def rerank(
        self, query: str, nodes: List[NodeWithScore]
    ) -> List[NodeWithScore]:
        """Score and re-rank nodes, returning the top_n most relevant."""
        if not nodes:
            return []

        # Ensure we look at at least 20, or 4x the requested output
        candidate_count = max(self.top_n * 4, 20)
        candidates = nodes[:candidate_count]

        logger.info("Re-ranking candidates", count=len(candidates))

        tasks = [self._score_node(query, node) for node in candidates]
        scored_nodes = await asyncio.gather(*tasks)

        scored_nodes.sort(key=lambda x: x.score or 0.0, reverse=True)

        return scored_nodes[: self.top_n]

    async def _score_node(self, query: str, node: NodeWithScore) -> NodeWithScore:
        content_snippet = node.node.get_content()[:500]

        prompt = (
            f"Query: {query}\n"
            f"Passage: {content_snippet}...\n\n"
            "Rate the relevance of this passage to the query on a scale of 0.0 to 10.0. "
            "0 means completely irrelevant, 10 means perfect answer. "
            "Return ONLY the number."
        )

        async with self._semaphore:
            try:
                response = await self.llm.acomplete(prompt)
                score_text = response.text.strip()

                match = re.search(r"\b([0-9]?\.[0-9]+|[0-9]+)\b", score_text)
                if match:
                    raw_score = float(match.group(1))
                    node.score = min(max(raw_score, 0.0), 10.0) / 10.0
                else:
                    node.score = 0.5  # Neutral fallback

            except Exception as e:
                logger.warning("Reranking failed for node", error=str(e))
                pass  # Keep original score

        return node
