# RAG Service

Independent Python FastAPI service for PDF-based Retrieval-Augmented Generation.

## What this service does

- Extracts selectable Arabic and English text from PDF files
- Uses token-aware, paragraph-aware chunking
- Generates batched embeddings and stores them in PostgreSQL with pgvector
- Automatically selects the relevant document from the user question
- Asks for the subject when the document cannot be determined confidently
- Keeps the selected subject/documents active during follow-up conversation messages
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
X-Project-Id: course-or-tenant-id
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

## Chat and conversation endpoints

```http
POST /api/chat/conversations
GET /api/chat/conversations
GET /api/chat/conversations/{conversationId}/messages
DELETE /api/chat/conversations/{conversationId}
POST /api/chat/ask
POST /api/chat/stream
GET /api/chat/usage?limit=50
```

The recommended backend flow is:

1. Call `POST /api/chat/conversations` when the application creates a new chat.
2. Store the returned RAG conversation `id` in the .NET conversation record.
3. Send it as `conversationId` with every question.
4. Do not send `documentIds` during normal chat; the RAG service routes the question automatically.
5. Keep the same `conversationId` when replying to a clarification question.

Create a conversation:

```json
{
  "title": "اختياري"
}
```

Ask without choosing a document:

```json
{
  "question": "اشرحلي درس المتتاليات",
  "conversationId": "conversation-id"
}
```

`documentIds` remains optional for admin tools, tests, or a UI opened inside one
specific book:

```json
{
  "question": "اشرح هذه الصفحة",
  "conversationId": "conversation-id",
  "documentIds": ["forced-document-id"]
}
```

## Automatic document routing

The service first rewrites follow-up messages into a standalone search query.
It then scores documents using:

- semantic similarity from document chunks
- full-text lexical matches
- file/display-name matches
- recognized subject names
- the active documents from the current conversation

When one document or one subject is clear, the response includes:

```json
{
  "conversationId": "...",
  "needsClarification": false,
  "routingStatus": "selected",
  "selectedDocuments": [
    {
      "documentId": "...",
      "name": "دليل المعلم رياضيات صف 12",
      "subject": "الرياضيات",
      "score": 0.91,
      "selectionReason": "automatic"
    }
  ],
  "answer": "...",
  "sources": []
}
```

When the question is too vague, such as `اشرحلي الدرس الأول`, the API does not
generate a guessed answer. It returns:

```json
{
  "conversationId": "...",
  "needsClarification": true,
  "routingStatus": "clarification",
  "clarificationQuestion": "شو المادة يلي بدك تسأل عنها؟",
  "candidateSubjects": ["الرياضيات", "الفيزياء", "أمن المعلومات"],
  "candidateDocuments": [],
  "answer": "شو المادة يلي بدك تسأل عنها؟",
  "sources": []
}
```

Send the user's reply with the same conversation ID:

```json
{
  "question": "رياضيات",
  "conversationId": "same-conversation-id"
}
```

The service combines this reply with the unresolved earlier question, chooses the
math document, and stores those documents as active for later messages such as
`طيب حل السؤال الخامس`.

The answer process is automatic; there is no strict/tutor mode switch:

1. Use an explicit answer from the retrieved document when available.
2. Otherwise derive from a relevant definition, rule, formula, or example.
3. Otherwise answer from stable general knowledge without inventing a document citation.
4. If the user asks what a document says and the material is not found, state that clearly.

The non-streaming response returns only sources referenced in the answer as
`[S1]`, `[S2]`, and so on. It also returns `retrievedSourceCount` for debugging.

Streaming normally emits:

```txt
started
resolved_question
routing
sources
delta ...
usage
done
```

When a subject clarification is required, it emits:

```txt
started
resolved_question
clarification
delta
usage
done
```

The initial `sources` event contains retrieval candidates. The final `done`
event contains only the sources actually cited by the completed answer.

## Routing and retrieval configuration

```env
DOCUMENT_ROUTING_CANDIDATE_CHUNKS=80
DOCUMENT_ROUTING_MIN_SCORE=0.28
DOCUMENT_ROUTING_AMBIGUITY_MARGIN=0.08
DOCUMENT_ROUTING_MAX_DOCUMENTS=3
DOCUMENT_ROUTING_ACTIVE_BOOST=0.12

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

## .NET integration

A complete .NET HttpClient and DTO example is available in:

```txt
docs/dotnet-conversations.md
```

Use the .NET database as the application source of truth for users,
conversations, permissions, and UI messages. Store the RAG `conversationId`
beside the application conversation ID. The RAG database keeps its own message
copy only for retrieval and follow-up context.

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
