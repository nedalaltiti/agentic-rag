-- 1. Vector Index (HNSW)
-- Tuned for Recall vs Speed balance (m=16 is standard)
-- Note: Creation might be slow on massive datasets, but fine for this assessment.
CREATE INDEX IF NOT EXISTS idx_chunks_embedding_hnsw
ON chunks 
USING hnsw (embedding vector_cosine_ops) 
WITH (m = 16, ef_construction = 64);

-- 2. Metadata Filter Index (GIN)
CREATE INDEX IF NOT EXISTS idx_chunks_metadata_gin
ON chunks 
USING gin (metadata);

-- 3. Keyword Search Index (GIN on TSVector)
CREATE INDEX IF NOT EXISTS idx_chunks_content_tsv_gin
ON chunks 
USING gin (content_tsv);

-- 4. Standard Lookups
CREATE INDEX IF NOT EXISTS idx_documents_file_name ON documents(file_name);
CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON chunks(document_id);
CREATE INDEX IF NOT EXISTS idx_conversations_session_id ON conversations(session_id);
CREATE INDEX IF NOT EXISTS idx_conversations_created_at ON conversations(created_at);
