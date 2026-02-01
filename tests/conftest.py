"""Shared test fixtures for the agentic-rag test suite."""

import uuid

import pytest
from llama_index.core.schema import NodeWithScore, TextNode

from agentic_rag.shared.schemas import Citation


def make_node_with_score(
    doc_id: str,
    node_id: str,
    text: str,
    metadata: dict | None = None,
    score: float = 0.9,
) -> NodeWithScore:
    """Factory to build a LlamaIndex NodeWithScore for testing."""
    meta = metadata or {}
    meta.setdefault("document_id", doc_id)
    node = TextNode(id_=node_id, text=text, metadata=meta)
    return NodeWithScore(node=node, score=score)


@pytest.fixture()
def sample_citations() -> list[Citation]:
    """Return a list of Citation objects for tests that need pre-built citations."""
    return [
        Citation(
            document_id=uuid.uuid4(),
            chunk_id=uuid.uuid4(),
            file_name="pdpl+overview.md",
            page_number=3,
            section_path="Chapter 1 > Scope",
            chunk_text="The PDPL applies to all personal data processing.",
            score=0.95,
        ),
        Citation(
            document_id=uuid.uuid4(),
            chunk_id=uuid.uuid4(),
            file_name="compliance_guide.md",
            page_number=None,
            section_path=None,
            chunk_text="Organizations must appoint a data protection officer.",
            score=0.88,
        ),
    ]
