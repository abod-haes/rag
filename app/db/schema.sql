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
    created_at TIMESTAMP DEFAULT NOW()
);

ALTER TABLE documents
ADD COLUMN IF NOT EXISTS name TEXT;

UPDATE documents
SET name = file_name
WHERE name IS NULL OR BTRIM(name) = '';

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
