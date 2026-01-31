-- HNSW indexes for fast approximate nearest neighbor search
-- Optimized for pgvector with cosine similarity

-- Create HNSW index on document embeddings
-- m: max connections per layer (higher = better recall, more memory)
-- ef_construction: size of dynamic candidate list during build (higher = better quality, slower build)
CREATE INDEX IF NOT EXISTS idx_documents_embedding_hnsw
ON documents
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- Create HNSW index on chunk embeddings (main retrieval table)
-- Using higher ef_construction for better recall on chunks
CREATE INDEX IF NOT EXISTS idx_chunks_embedding_hnsw
ON chunks
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 100);

-- B-tree indexes for metadata filtering
CREATE INDEX IF NOT EXISTS idx_chunks_document_id
ON chunks (document_id);

CREATE INDEX IF NOT EXISTS idx_chunks_created_at
ON chunks (created_at DESC);

CREATE INDEX IF NOT EXISTS idx_documents_file_name
ON documents (file_name);

-- GIN index for full-text search on chunk content (hybrid search)
CREATE INDEX IF NOT EXISTS idx_chunks_content_fts
ON chunks
USING gin (to_tsvector('english', content));

-- Set HNSW search parameters for queries
-- ef: size of dynamic candidate list during search (higher = better recall, slower search)
-- Can be adjusted per-session: SET hnsw.ef_search = 100;
COMMENT ON INDEX idx_chunks_embedding_hnsw IS 'HNSW index for semantic search. Set hnsw.ef_search (default 40) for recall/speed tradeoff.';
