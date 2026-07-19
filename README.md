# RAG Service

Independent Python FastAPI service for PDF-based Retrieval-Augmented Generation.

## What this service does

- Upload PDF files per `userId` and `projectId`
- Extract text from PDFs
- Split text into chunks
- Generate OpenAI embeddings for chunks
- Store vectors in PostgreSQL using pgvector
- Answer questions using only retrieved chunks from the uploaded files
- Return sources with file name, page number, and chunk index

## Tech stack

- Python FastAPI
- PostgreSQL + pgvector
- OpenAI API
- pypdf
- Docker Compose

## Local run

1. Copy environment file:

```bash
cp .env.example .env
```

2. Add your OpenAI key and RAG API key inside `.env`.

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
GET /api/documents
GET /api/documents/{documentId}/usage
DELETE /api/documents/{documentId}
POST /api/chat/ask
POST /api/chat/stream
GET /api/chat/usage?limit=50
```

`POST /api/documents/upload` accepts multipart fields:

- `file`: required PDF file
- `name`: optional display name; defaults to the original PDF file name

`GET /api/documents` returns both the stored display `name` and original
`fileName` with each document `id`. It also returns a `usage` object containing
the embedding/OCR token counts and estimated indexing cost in USD. The same
usage object is available from `GET /api/documents/{documentId}/usage`.

`POST /api/chat/ask` returns a `usage` object next to `answer` and `sources`.
It includes the query-embedding tokens, chat input/cached/output tokens, total
tokens, and estimated cost in USD. `GET /api/chat/usage` returns the latest
stored question usage records.

Answers use the retrieved chunks as textbook material: the model can apply a
rule or worked-example pattern to a new question even when the exact exercise
and final answer do not appear verbatim in the PDF.

`POST /api/chat/stream` accepts the same JSON body as `/api/chat/ask` and
returns Server-Sent Events in this order: `started`, `sources`, one or more
`delta` events, `usage`, then `done`. Failures are emitted as an `error` event.

Token counts are read from the AI provider response. USD amounts are estimates
calculated from the configurable `*_PRICE_PER_MILLION_TOKENS` environment
variables. Existing records created before usage tracking have zero usage.

## OCR fallback

Text extraction is attempted first. Optional OpenAI OCR fallback for scanned
PDFs can be enabled with `ENABLE_OCR_FALLBACK=true` and limited with
`MAX_OCR_PAGES`.
