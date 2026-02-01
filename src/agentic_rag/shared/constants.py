"""Application-wide constants and metadata keys.

These constants ensure consistency across indexing and retrieval.
"""

# =============================================================================
# Document Metadata Keys
# =============================================================================
# Used in vector store metadata for consistent access

DOC_ID = "document_id"
DOC_NAME = "file_name"
DOC_HASH = "content_hash"
PAGE_NUMBER = "page_number"
CHUNK_INDEX = "chunk_index"
SECTION_TITLE = "section_title"
CREATED_AT = "created_at"

# Context prefix for contextual chunking
CONTEXT_PREFIX = "context_prefix"

# =============================================================================
# Vector Store Constants
# =============================================================================

EMBEDDING_DIMENSION = 1024  # qwen3-embedding dimension
COLLECTION_NAME = "documents"

# =============================================================================
# Agent Names
# =============================================================================

AGENT_PLANNER = "Query Planner"
AGENT_RETRIEVER = "Knowledge Retriever"
AGENT_SYNTHESIZER = "Response Synthesizer"

# =============================================================================
# API Constants
# =============================================================================

OPENAI_CHAT_COMPLETION_OBJECT = "chat.completion"
OPENAI_CHAT_COMPLETION_CHUNK_OBJECT = "chat.completion.chunk"
OPENAI_MODEL_OBJECT = "model"
OPENAI_MODEL_LIST_OBJECT = "list"
