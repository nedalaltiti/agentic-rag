"""Tests for agentic_rag.shared.citations.format_citations."""

import uuid

import pytest

from agentic_rag.shared.citations import format_citations
from tests.conftest import make_node_with_score


class TestFormatCitations:
    def test_format_citations_valid(self):
        doc_id = str(uuid.uuid4())
        node_id = str(uuid.uuid4())
        nodes = [
            make_node_with_score(
                doc_id=doc_id,
                node_id=node_id,
                text="Article 5 defines lawful processing.",
                metadata={
                    "document_id": doc_id,
                    "file_name": "pdpl.md",
                    "page_number": 5,
                    "section_path": "Chapter 2 > Article 5",
                },
                score=0.92,
            )
        ]
        result = format_citations(nodes)

        assert len(result) == 1
        cit = result[0]
        assert str(cit.document_id) == doc_id
        assert str(cit.chunk_id) == node_id
        assert cit.file_name == "pdpl.md"
        assert cit.page_number == 5
        assert cit.section_path == "Chapter 2 > Article 5"
        assert cit.chunk_text == "Article 5 defines lawful processing."
        assert cit.score == pytest.approx(0.92)

    def test_format_citations_invalid_doc_id(self):
        nodes = [
            make_node_with_score(
                doc_id="not-a-uuid",
                node_id=str(uuid.uuid4()),
                text="Some text",
            )
        ]
        result = format_citations(nodes)
        assert result == []

    def test_format_citations_invalid_chunk_id(self):
        nodes = [
            make_node_with_score(
                doc_id=str(uuid.uuid4()),
                node_id="bad-chunk-id",
                text="Some text",
            )
        ]
        result = format_citations(nodes)
        assert result == []

    def test_format_citations_empty(self):
        assert format_citations([]) == []

    def test_format_citations_missing_metadata(self):
        doc_id = str(uuid.uuid4())
        node_id = str(uuid.uuid4())
        nodes = [
            make_node_with_score(
                doc_id=doc_id,
                node_id=node_id,
                text="Minimal node",
                metadata={"document_id": doc_id},
                score=0.5,
            )
        ]
        result = format_citations(nodes)

        assert len(result) == 1
        cit = result[0]
        assert cit.file_name == "Unknown Source"
        assert cit.page_number is None
        assert cit.section_path is None
        assert cit.score == pytest.approx(0.5)
