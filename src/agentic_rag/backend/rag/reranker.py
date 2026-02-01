"""LLM-based re-ranking for retrieved chunks.

Uses Ollama LLM to score relevance of candidates and re-rank them.
"""

import asyncio
import re

import structlog
from llama_index.core.schema import NodeWithScore

from agentic_rag.shared.config import settings
from agentic_rag.shared.llm_factory import get_llm
from agentic_rag.shared.prompts import PromptRegistry

logger = structlog.get_logger()


class LLMReranker:
    """LLM-based re-ranker using prompt templates from Phoenix."""

    def __init__(self):
        self.llm = get_llm()
        self.top_n = settings.TOP_K_RERANK
        self._semaphore = asyncio.Semaphore(5)

    async def rerank(self, query: str, nodes: list[NodeWithScore]) -> list[NodeWithScore]:
        """
        Re-rank nodes by LLM-scored relevance.

        Args:
            query: User query
            nodes: List of retrieved nodes with scores

        Returns:
            Top-k re-ranked nodes
        """
        if not nodes:
            return []

        # Take more candidates than we need for better selection
        candidates = nodes[: max(self.top_n * 4, 20)]
        logger.info("Re-ranking candidates", count=len(candidates))

        tasks = [self._score_node(query, n) for n in candidates]
        scored_nodes = await asyncio.gather(*tasks)

        # Sort by re-ranked score descending
        scored_nodes.sort(key=lambda x: x.score or 0.0, reverse=True)
        return scored_nodes[: self.top_n]

    async def _score_node(self, query: str, node: NodeWithScore) -> NodeWithScore:
        """Score a single node using LLM."""
        passage = node.node.get_content()[:500]

        # In prod, could fetch template from Phoenix
        if settings.ENVIRONMENT == "prod":
            template = PromptRegistry.get_template("reranker_template")
            prompt = template.replace("{{ query }}", query).replace("{{ passage }}", passage)
        else:
            # Dev: always render locally
            prompt = PromptRegistry.render("reranker_template", query=query, passage=passage)

        async with self._semaphore:
            try:
                response = await self.llm.acomplete(prompt)
                score_text = (response.text or "").strip()
                
                # Extract number from response
                match = re.search(r"\b(10|[0-9](?:\.[0-9]+)?)\b", score_text)
                if match:
                    raw_score = float(match.group(1))
                    # Normalize to 0-1 range
                    node.score = min(max(raw_score, 0.0), 10.0) / 10.0
                else:
                    node.score = 0.5
            except Exception as e:
                logger.warning("Re-ranking failed for node", error=str(e))
                # Keep original score on failure
                pass

        return node
