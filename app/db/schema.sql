CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
    id UUID PRIMARY KEY,
    user_id TEXT NOT NULL,
    project_id TEXT NOT NULL,
    name TEXT,
    file_name TEXT NOT NULL,
    file_path TEXT NOT NULL,
    status TEXT NOT NULL,
    error_message TEXT,
    ai_provider TEXT,
    embedding_model TEXT,
    embedding_tokens BIGINT NOT NULL DEFAULT 0,
    ocr_model TEXT,
    ocr_input_tokens BIGINT NOT NULL DEFAULT 0,
    ocr_cached_input_tokens BIGINT NOT NULL DEFAULT 0,
    ocr_output_tokens BIGINT NOT NULL DEFAULT 0,
    estimated_cost_usd NUMERIC(18, 10) NOT NULL DEFAULT 0,
    created_at TIMESTAMP DEFAULT NOW()
);

ALTER TABLE documents
ADD COLUMN IF NOT EXISTS name TEXT;

UPDATE documents
SET name = file_name
WHERE name IS NULL OR BTRIM(name) = '';

ALTER TABLE documents ADD COLUMN IF NOT EXISTS ai_provider TEXT;
ALTER TABLE documents ADD COLUMN IF NOT EXISTS embedding_model TEXT;
ALTER TABLE documents ADD COLUMN IF NOT EXISTS embedding_tokens BIGINT NOT NULL DEFAULT 0;
ALTER TABLE documents ADD COLUMN IF NOT EXISTS ocr_model TEXT;
ALTER TABLE documents ADD COLUMN IF NOT EXISTS ocr_input_tokens BIGINT NOT NULL DEFAULT 0;
ALTER TABLE documents ADD COLUMN IF NOT EXISTS ocr_cached_input_tokens BIGINT NOT NULL DEFAULT 0;
ALTER TABLE documents ADD COLUMN IF NOT EXISTS ocr_output_tokens BIGINT NOT NULL DEFAULT 0;
ALTER TABLE documents ADD COLUMN IF NOT EXISTS estimated_cost_usd NUMERIC(18, 10) NOT NULL DEFAULT 0;

CREATE TABLE IF NOT EXISTS document_chunks (
    id UUID PRIMARY KEY,
    document_id UUID REFERENCES documents(id) ON DELETE CASCADE,
    user_id TEXT NOT NULL,
    project_id TEXT NOT NULL,
    content TEXT NOT NULL,
    page_number INT,
    chunk_index INT NOT NULL,
    embedding vector(768),
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_documents_project_user
ON documents(project_id, user_id);

CREATE INDEX IF NOT EXISTS idx_chunks_project_user
ON document_chunks(project_id, user_id);

CREATE INDEX IF NOT EXISTS idx_chunks_document
ON document_chunks(document_id);

CREATE INDEX IF NOT EXISTS idx_chunks_embedding_hnsw
ON document_chunks USING hnsw (embedding vector_cosine_ops);

CREATE TABLE IF NOT EXISTS chat_usage (
    id UUID PRIMARY KEY,
    user_id TEXT NOT NULL,
    project_id TEXT NOT NULL,
    question TEXT NOT NULL,
    document_ids TEXT[],
    ai_provider TEXT NOT NULL,
    embedding_model TEXT,
    chat_model TEXT,
    query_embedding_tokens BIGINT NOT NULL DEFAULT 0,
    input_tokens BIGINT NOT NULL DEFAULT 0,
    cached_input_tokens BIGINT NOT NULL DEFAULT 0,
    output_tokens BIGINT NOT NULL DEFAULT 0,
    total_tokens BIGINT NOT NULL DEFAULT 0,
    estimated_cost_usd NUMERIC(18, 10) NOT NULL DEFAULT 0,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_chat_usage_project_user_created
ON chat_usage(project_id, user_id, created_at DESC);
