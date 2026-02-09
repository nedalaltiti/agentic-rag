"""Application-wide constants and metadata keys."""

# Document metadata keys (used in vector store for consistent access)
DOC_ID = "document_id"
DOC_NAME = "file_name"
DOC_HASH = "content_hash"
PAGE_NUMBER = "page_number"
CHUNK_INDEX = "chunk_index"
SECTION_TITLE = "section_title"
CREATED_AT = "created_at"
CONTEXT_PREFIX = "context_prefix"

# Vector store
COLLECTION_NAME = "documents"

# Agent names
AGENT_PLANNER = "Query Planner"
AGENT_RETRIEVER = "Knowledge Retriever"
AGENT_SYNTHESIZER = "Response Synthesizer"

# OpenAI-compat object types
OPENAI_CHAT_COMPLETION_OBJECT = "chat.completion"
OPENAI_CHAT_COMPLETION_CHUNK_OBJECT = "chat.completion.chunk"
OPENAI_MODEL_OBJECT = "model"
OPENAI_MODEL_LIST_OBJECT = "list"
