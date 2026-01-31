"""Citation extraction and formatting from retrieved nodes.

Converts LlamaIndex NodeWithScore objects into validated Citation schemas
with strict UUID enforcement for document and chunk identifiers.
"""

import uuid
from typing import List

from llama_index.core.schema import NodeWithScore
import structlog

from agentic_rag.shared.schemas import Citation

logger = structlog.get_logger()


def format_citations(nodes: List[NodeWithScore]) -> List[Citation]:
    """Convert retrieved nodes into Citation objects, skipping invalid entries."""
    citations = []

    for node_score in nodes:
        node = node_score.node
        meta = node.metadata or {}

        # 1. Document ID
        raw_doc_id = meta.get("document_id")
        try:
            doc_id = uuid.UUID(str(raw_doc_id))
        except (ValueError, TypeError):
            logger.error(
                "Skipping citation: Invalid/Missing document_id", raw=raw_doc_id
            )
            continue

        # 2. Chunk ID
        try:
            chunk_id = uuid.UUID(str(node.node_id))
        except (ValueError, TypeError):
            logger.error("Skipping citation: Invalid chunk_id", raw=node.node_id)
            continue

        # 3. Metadata
        file_name = meta.get("file_name", "Unknown Source")
        page_num = meta.get("page_number")

        citation = Citation(
            document_id=doc_id,
            chunk_id=chunk_id,
            file_name=file_name,
            page_number=int(page_num) if page_num is not None else None,
            chunk_text=node.get_content().strip(),
            score=node_score.score or 0.0,
        )
        citations.append(citation)

    return citations
