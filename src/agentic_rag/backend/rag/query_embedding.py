"""Shared query embedding helpers to keep cache and retrieval aligned."""

from agentic_rag.core.llm_factory import get_embedding_model


def build_query_embedding_text(query: str) -> str:
    """Build the exact query text used for embeddings.

    This mirrors the instruct-style prefix used in the retriever.
    """
    return (
        "Instruct: Given a search query, retrieve relevant passages that answer the query.\n"
        f"Query: {query}"
    )


async def get_query_embedding(query: str) -> list[float]:
    """Return the embedding for a query using the standardized query text."""
    embed_model = get_embedding_model()
    query_text = build_query_embedding_text(query)
    return await embed_model.aget_query_embedding(query_text)
