-- Add index versioning fields to documents and chunks
-- NOTE: Defaults reflect the baseline config; update these if your .env differs.
ALTER TABLE documents
    ADD COLUMN IF NOT EXISTS index_version TEXT NOT NULL DEFAULT 'v1',
    ADD COLUMN IF NOT EXISTS embedding_model TEXT NOT NULL DEFAULT 'qwen3-embedding:0.6b',
    ADD COLUMN IF NOT EXISTS embedding_dimension INT NOT NULL DEFAULT 1024;

ALTER TABLE chunks
    ADD COLUMN IF NOT EXISTS index_version TEXT NOT NULL DEFAULT 'v1',
    ADD COLUMN IF NOT EXISTS embedding_model TEXT NOT NULL DEFAULT 'qwen3-embedding:0.6b',
    ADD COLUMN IF NOT EXISTS embedding_dimension INT NOT NULL DEFAULT 1024,
    ADD COLUMN IF NOT EXISTS chunk_hash VARCHAR(64);

-- Semantic cache for fast RAG responses (answer + citations)
-- query_embedding dimension must match EMBEDDING_DIMENSION in config/.env
CREATE TABLE IF NOT EXISTS semantic_cache (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    query_text TEXT NOT NULL,
    query_embedding vector(1024) NOT NULL,
    answer TEXT NOT NULL,
    citations JSONB NOT NULL DEFAULT '[]'::jsonb,
    embedding_model TEXT NOT NULL,
    embedding_dimension INT NOT NULL,
    index_version TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    expires_at TIMESTAMPTZ NOT NULL
);

-- Indexes for semantic cache lookups
CREATE INDEX IF NOT EXISTS idx_semantic_cache_embedding_hnsw
ON semantic_cache
USING hnsw (query_embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

CREATE INDEX IF NOT EXISTS idx_semantic_cache_version_model_dim_expires
ON semantic_cache(index_version, embedding_model, embedding_dimension, expires_at);

-- Index to support retrieval filtering by index version/model/dimension
CREATE INDEX IF NOT EXISTS idx_chunks_version_model_dim
ON chunks(index_version, embedding_model, embedding_dimension);

CREATE INDEX IF NOT EXISTS idx_chunks_chunk_hash_version_model_dim
ON chunks(chunk_hash, index_version, embedding_model, embedding_dimension);
