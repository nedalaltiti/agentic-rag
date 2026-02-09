"""LLM-based re-ranking for retrieved chunks.

Uses Ollama LLM to score relevance of candidates and re-rank them.
"""

import asyncio
import hashlib
import json
import re
import time
from collections import OrderedDict

import structlog
from llama_index.core.schema import NodeWithScore

from agentic_rag.core.config import settings
from agentic_rag.core.llm_factory import get_llm
from agentic_rag.core.prompts import PromptRegistry

logger = structlog.get_logger()


class LLMReranker:
    """LLM-based re-ranker using prompt templates from Phoenix."""

    def __init__(self):
        self.llm = get_llm(request_timeout=settings.RERANKER_TIMEOUT)
        self.top_n = settings.TOP_K_RERANK
        self._semaphore = asyncio.Semaphore(5)
        self._score_cache: OrderedDict[str, tuple[float, float]] = OrderedDict()

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

        candidates = nodes[: self.top_n * 2]
        logger.info("Re-ranking candidates", count=len(candidates))

        timeout = settings.RERANKER_TIMEOUT
        tasks = [self._score_node(query, n) for n in candidates]
        try:
            scored_nodes = await asyncio.wait_for(asyncio.gather(*tasks), timeout=timeout)
        except TimeoutError:
            logger.warning("Reranker timed out, returning original order", timeout=timeout)
            return nodes[: self.top_n]

        # Sort by re-ranked score descending
        scored_nodes.sort(key=lambda x: x.score or 0.0, reverse=True)
        return scored_nodes[: self.top_n]

    async def _score_node(self, query: str, node: NodeWithScore) -> NodeWithScore:
        """Score a single node using LLM."""
        passage = node.node.get_content()[: settings.RERANKER_MAX_PASSAGE_CHARS]

        cache_key = hashlib.sha256(f"{query}\n{passage}".encode()).hexdigest()
        ttl = settings.RERANK_CACHE_TTL
        now = time.monotonic()
        if ttl > 0:
            cached = self._score_cache.get(cache_key)
            if cached is not None:
                score, ts = cached
                if (now - ts) <= ttl:
                    self._score_cache.move_to_end(cache_key)
                    node.score = score
                    return node
                self._score_cache.pop(cache_key, None)

        prompt = PromptRegistry.render("reranker_template", query=query, passage=passage)

        async with self._semaphore:
            try:
                response = await self.llm.acomplete(prompt)
                score_text = (response.text or "").strip()

                raw_score = None
                try:
                    parsed = json.loads(score_text)
                    if isinstance(parsed, dict) and "score" in parsed:
                        raw_score = float(parsed["score"])
                except Exception:
                    raw_score = None

                if raw_score is None:
                    match = re.search(r"\b(10|[0-9](?:\.[0-9]+)?)\b", score_text)
                    if match:
                        raw_score = float(match.group(1))

                if raw_score is not None:
                    # Normalize to 0-1 range
                    node.score = min(max(raw_score, 0.0), 10.0) / 10.0
                else:
                    node.score = 0.5
            except Exception as e:
                logger.warning("Re-ranking failed for node", error=str(e))
                # Keep original score on failure
                pass

        if ttl > 0 and node.score is not None:
            self._score_cache[cache_key] = (float(node.score), now)
            if len(self._score_cache) > settings.RERANK_CACHE_MAX:
                self._score_cache.popitem(last=False)

        return node
