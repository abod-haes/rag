# RAG Service

Independent Python FastAPI service for PDF-based Retrieval-Augmented Generation.

## What this service does

- Extracts selectable Arabic and English text from PDF files
- Uses token-aware, paragraph-aware chunking
- Generates batched embeddings and stores them in PostgreSQL with pgvector
- Combines vector similarity, PostgreSQL lexical search, and exact token overlap
- Locally reranks results and includes neighboring chunks when useful
- Answers directly from the document when supported
- Derives a solution from a nearby rule or worked example when possible
- Falls back to reliable general knowledge when the files do not contain a useful answer
- Returns only sources cited in the generated answer
- Supports persistent follow-up conversations
- Tracks token usage and estimated costs
- Supports synchronous and background document indexing

## Tech stack

- Python 3.12 + FastAPI
- PostgreSQL 17 + pgvector
- OpenAI or Gemini provider abstraction
- PyMuPDF + pypdf
- Docker Compose

## Local run

1. Copy the environment file:

```bash
cp .env.example .env
```

2. Add the provider key and change `RAG_API_KEY` inside `.env`.

3. Start the services:

```bash
docker compose up --build
```

4. Open Swagger:

```txt
http://localhost:8000/docs
```

## API security and request scope

All endpoints except `/health` require:

```http
X-API-Key: your-secret-key
```

The .NET backend can isolate data by also sending:

```http
X-User-Id: application-user-id
X-Project-Id: application-project-id
```

When these headers are omitted, the service uses `default-user` and
`default-project` for backward compatibility. The public client should not call
the RAG service directly; the trusted backend should attach these headers.

## Document endpoints

```http
POST /api/documents/upload
POST /api/documents/upload-async
GET /api/documents
GET /api/documents/{documentId}/status
GET /api/documents/{documentId}/usage
POST /api/documents/{documentId}/retry
DELETE /api/documents/{documentId}
```

`POST /api/documents/upload` keeps the original synchronous behavior.

`POST /api/documents/upload-async` returns `202 Accepted` with a `statusUrl`.
The status response includes the current stage, indexed chunk count, total chunk
count, and progress percentage.

Uploads are validated using both the `.pdf` extension and the PDF file header.
The service enforces `MAX_UPLOAD_SIZE_MB` and detects duplicates by SHA-256 hash
inside the same user/project scope. Set `ALLOW_DUPLICATE_DOCUMENTS=true` to allow
identical files.

Embedding requests are grouped using `EMBEDDING_BATCH_SIZE`. Failed or ready
documents can be rebuilt through the retry endpoint.

## Chat endpoints

```http
POST /api/chat/ask
POST /api/chat/stream
GET /api/chat/usage?limit=50
GET /api/chat/conversations
GET /api/chat/conversations/{conversationId}/messages
```

First question:

```json
{
  "question": "اشرحلي محتوى الملف باختصار",
  "documentIds": ["optional-document-id"]
}
```

The response includes a generated `conversationId`. Send it with a follow-up:

```json
{
  "question": "طيب ليش استخدمنا هالقانون؟",
  "conversationId": "conversation-id-from-the-first-response",
  "documentIds": ["optional-document-id"]
}
```

The service rewrites follow-up questions into standalone retrieval queries while
preserving the original question for the final answer.

The answer process is automatic; there is no strict/tutor mode switch:

1. Use an explicit answer from the retrieved document when available.
2. Otherwise derive from a relevant definition, rule, formula, or example.
3. Otherwise answer from stable general knowledge without inventing a document citation.
4. If the user asks what a document says and the material is not found, state that clearly.

The non-streaming response returns only sources referenced in the answer as
`[S1]`, `[S2]`, and so on. It also returns `retrievedSourceCount` for debugging.

Streaming events are emitted in this order:

```txt
started
resolved_question
sources
delta ...
usage
done
```

The initial `sources` event contains retrieval candidates. The final `done`
event contains only the sources actually cited by the completed answer.

## Retrieval configuration

Useful environment variables:

```env
TOP_K=5
RETRIEVAL_CANDIDATE_K=20
MIN_RELEVANCE_SCORE=0.20
VECTOR_WEIGHT=0.60
LEXICAL_WEIGHT=0.25
EXACT_MATCH_WEIGHT=0.15
NEIGHBOR_WINDOW=1
MAX_CONTEXT_CHUNKS=12
```

New PDFs are chunked using:

```env
MAX_CHUNK_TOKENS=900
CHUNK_OVERLAP_TOKENS=120
```

Existing indexed documents keep their old chunks until they are retried or
uploaded again.

## Usage, observability, and limits

Responses include provider token usage and configurable estimated USD costs.
Every HTTP response includes:

```http
X-Request-Id
X-Response-Time-Ms
```

The service logs request method, path, status code, request ID, and duration. A
basic per-process limiter is controlled with:

```env
RATE_LIMIT_REQUESTS_PER_MINUTE=120
```

Use a shared gateway or Redis-backed limiter when running multiple API replicas.
OpenAI Responses requests use `store=false`.

## Automated evaluation

Copy and edit the example dataset:

```txt
evals/dataset.example.json
```

Enable cases, replace document IDs, and run:

```bash
python scripts/evaluate_rag.py evals/dataset.example.json
```

Optional environment variables:

```env
RAG_BASE_URL=http://localhost:8000
RAG_API_KEY=change-this-secret
RAG_USER_ID=default-user
RAG_PROJECT_ID=default-project
```

The evaluator checks expected keywords, source pages, forbidden phrases,
answer/fallback behavior, pass rate, and estimated cost.

## OCR fallback

Selectable text extraction is always attempted first. Optional OpenAI OCR for
scanned PDFs can be enabled with:

```env
ENABLE_OCR_FALLBACK=true
MAX_OCR_PAGES=10
```
