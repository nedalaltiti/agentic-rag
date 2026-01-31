-- 1. Documents: Stores file-level metadata
CREATE TABLE IF NOT EXISTS documents (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    file_name VARCHAR(255) NOT NULL,
    file_path VARCHAR(1024),
    file_hash VARCHAR(64), -- SHA256 for deduplication
    page_count INT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Optimization: Prevent uploading the same file twice
CREATE UNIQUE INDEX IF NOT EXISTS ux_documents_file_hash
ON documents(file_hash)
WHERE file_hash IS NOT NULL;

-- 2. Chunks: The core RAG storage
-- embedding is vector(768) to match 'nomic-embed-text'
CREATE TABLE IF NOT EXISTS chunks (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    
    -- The raw text content
    content TEXT NOT NULL,
    
    -- The "Contextual" text (chunk + document context prepended)
    contextual_content TEXT,
    
    -- Citation/Filter metadata (page_number, section_title)
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    
    -- Full-Text Search Vector (Auto-generated from content)
    -- Using 'simple' config to support non-English (including Arabic) text better
    content_tsv tsvector GENERATED ALWAYS AS (
      to_tsvector('simple', coalesce(content, ''))
    ) STORED,
    
    -- Vector Embedding (768 dimensions)
    -- Strict Mode: Must be present for indexing to succeed
    embedding vector(768) NOT NULL,
    
    chunk_index INT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- 3. Conversations: Agent Memory persistence
CREATE TABLE IF NOT EXISTS conversations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id VARCHAR(255) NOT NULL,
    
    -- Enforce valid roles (CrewAI / System / User)
    role VARCHAR(50) NOT NULL CHECK (role IN ('user', 'assistant', 'system')),
    
    content TEXT NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- 4. Idempotent Trigger Setup (DB Purist Version)
-- First, define the function
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Second, apply the trigger safely (Idempotent check matching Table + Trigger Name)
DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1
    FROM pg_trigger t
    JOIN pg_class c ON c.oid = t.tgrelid
    WHERE t.tgname = 'update_documents_updated_at'
      AND c.relname = 'documents'
  ) THEN
    CREATE TRIGGER update_documents_updated_at
      BEFORE UPDATE ON documents
      FOR EACH ROW
      EXECUTE FUNCTION update_updated_at_column();
  END IF;
END
$$;
