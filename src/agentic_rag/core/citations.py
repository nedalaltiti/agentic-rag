"""Citation extraction from retrieved nodes."""

import uuid

import structlog
from llama_index.core.schema import NodeWithScore

from agentic_rag.core.schemas import Citation

logger = structlog.get_logger()


def format_citations(nodes: list[NodeWithScore]) -> list[Citation]:
    """Convert retrieved nodes into Citation objects, skipping invalid entries."""
    citations = []

    for node_score in nodes:
        node = node_score.node
        meta = node.metadata or {}

        raw_doc_id = meta.get("document_id")
        try:
            doc_id = uuid.UUID(str(raw_doc_id))
        except (ValueError, TypeError):
            logger.error("Skipping citation: Invalid/Missing document_id", raw=raw_doc_id)
            continue

        try:
            chunk_id = uuid.UUID(str(node.node_id))
        except (ValueError, TypeError):
            logger.error("Skipping citation: Invalid chunk_id", raw=node.node_id)
            continue

        file_name = meta.get("file_name", "Unknown Source")
        page_num = meta.get("page_number")
        section_path = meta.get("section_path")

        citation = Citation(
            document_id=doc_id,
            chunk_id=chunk_id,
            file_name=file_name,
            page_number=int(page_num) if page_num is not None else None,
            section_path=section_path or None,
            chunk_text=node.get_content().strip(),
            score=float(node_score.score or 0.0),
        )
        citations.append(citation)

    return citations
