# RAG Service

Independent Python FastAPI service for PDF-based Retrieval-Augmented Generation.

## What this service does

- Upload PDF files per `userId` and `projectId`
- Extract text from PDFs
- Split text into chunks
- Generate Gemini embeddings for chunks
- Store vectors in PostgreSQL using pgvector
- Answer questions using only retrieved chunks from the uploaded files
- Return sources with file name, page number, and chunk index

## Tech stack

- Python FastAPI
- PostgreSQL + pgvector
- Gemini API
- pypdf
- Docker Compose

## Local run

1. Copy environment file:

```bash
cp .env.example .env
```

2. Add your Gemini key and RAG API key inside `.env`.

3. Start services:

```bash
docker compose up --build
```

4. Open Swagger:

```txt
http://localhost:8000/docs
```

## API security

All endpoints except `/health` require:

```http
X-API-Key: your-secret-key
```

## Main endpoints

```http
GET /health
POST /api/documents/upload
GET /api/documents?userId=...&projectId=...
DELETE /api/documents/{documentId}?userId=...&projectId=...
POST /api/chat/ask
```

## First MVP scope

This MVP supports text-based PDF files. Scanned/image PDFs need OCR and will be added later.
