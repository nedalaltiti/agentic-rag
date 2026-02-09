"""Semantic scope gate — short-circuits off-topic queries before retrieval/LLM."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import structlog

from agentic_rag.core.config import settings
from agentic_rag.core.llm_factory import get_embedding_model

logger = structlog.get_logger()

ANCHORS_FILE = Path(__file__).resolve().parent.parent / "prompts" / "scope_anchors.txt"


class ScopeGate:
    """Embedding-based scope classifier. Caches anchor embeddings on first use."""

    _anchor_embeddings: np.ndarray | None = None
    _anchors: list[str] = []

    @classmethod
    def _load_anchors(cls) -> list[str]:
        """Load scope anchor texts from config file."""
        if cls._anchors:
            return cls._anchors
        cls._anchors = [
            line.strip() for line in ANCHORS_FILE.read_text().splitlines() if line.strip()
        ]
        return cls._anchors

    @classmethod
    async def _get_anchor_embeddings(cls) -> np.ndarray:
        """Compute and cache anchor embeddings."""
        if cls._anchor_embeddings is not None:
            return cls._anchor_embeddings

        anchors = cls._load_anchors()
        embed_model = get_embedding_model()

        embeddings = []
        for anchor in anchors:
            emb = await embed_model.aget_text_embedding(anchor)
            embeddings.append(emb)

        cls._anchor_embeddings = np.array(embeddings)
        logger.info("Scope gate initialized", num_anchors=len(anchors))
        return cls._anchor_embeddings

    @classmethod
    async def is_in_scope(cls, query: str) -> tuple[bool, float]:
        """Check if query is semantically within the configured domain scope.

        Returns (in_scope, max_similarity).
        """
        anchor_embs = await cls._get_anchor_embeddings()
        embed_model = get_embedding_model()

        query_emb = np.array(await embed_model.aget_text_embedding(query))

        # Cosine similarity against all anchors
        query_norm = query_emb / (np.linalg.norm(query_emb) + 1e-10)
        anchor_norms = anchor_embs / (np.linalg.norm(anchor_embs, axis=1, keepdims=True) + 1e-10)
        similarities = anchor_norms @ query_norm
        max_sim = float(np.max(similarities))

        in_scope = max_sim >= settings.SCOPE_GATE_THRESHOLD
        logger.info(
            "Scope gate check",
            query=query[:60],
            max_similarity=round(max_sim, 3),
            in_scope=in_scope,
        )
        return in_scope, max_sim
