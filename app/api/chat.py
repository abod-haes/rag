import json
import traceback
from collections.abc import Iterator

from pydantic import BaseModel, Field
from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from app.api.documents import DEFAULT_PROJECT_ID, DEFAULT_USER_ID
from app.core.security import verify_api_key
from app.services.chat_service import GeminiChatService
from app.services.embedding_service import EmbeddingService
from app.services.prompt_service import build_rag_prompt
from app.services.retrieval_service import RetrievalService

router = APIRouter(
    prefix="/api/chat",
    tags=["Chat"],
    dependencies=[Depends(verify_api_key)],
)


class AskRequest(BaseModel):
    question: str = Field(min_length=1)
    document_ids: list[str] | None = Field(default=None, alias="documentIds")


def _build_sources(chunks: list[dict]) -> list[dict]:
    return [
        {
            "documentId": chunk["document_id"],
            "name": chunk.get("name") or chunk["file_name"],
            "fileName": chunk["file_name"],
            "pageNumber": chunk["page_number"],
            "chunkIndex": chunk["chunk_index"],
            "score": round(chunk["score"], 4),
        }
        for chunk in chunks
    ]


def _sse(event: str, data: dict | list) -> str:
    payload = json.dumps(data, ensure_ascii=False)
    return f"event: {event}\ndata: {payload}\n\n"


@router.post("/ask")
def ask_question(request: AskRequest):
    embedding_service = EmbeddingService()
    retrieval_service = RetrievalService()
    chat_service = GeminiChatService()

    query_embedding = embedding_service.embed_query(request.question)
    chunks = retrieval_service.retrieve(
        query_embedding=query_embedding,
        user_id=DEFAULT_USER_ID,
        project_id=DEFAULT_PROJECT_ID,
        document_ids=request.document_ids,
    )

    if not chunks:
        return {
            "answer": "لا يوجد جواب واضح في الملفات المرفوعة.",
            "sources": [],
        }

    prompt = build_rag_prompt(request.question, chunks)
    answer = chat_service.generate_answer(prompt)

    sources = _build_sources(chunks)

    return {"answer": answer, "sources": sources}


@router.post("/stream")
def stream_question(request: AskRequest):
    def event_stream() -> Iterator[str]:
        try:
            yield _sse("started", {"message": "Processing question"})

            embedding_service = EmbeddingService()
            retrieval_service = RetrievalService()
            chat_service = GeminiChatService()

            query_embedding = embedding_service.embed_query(request.question)
            chunks = retrieval_service.retrieve(
                query_embedding=query_embedding,
                user_id=DEFAULT_USER_ID,
                project_id=DEFAULT_PROJECT_ID,
                document_ids=request.document_ids,
            )
            sources = _build_sources(chunks)
            yield _sse("sources", sources)

            if not chunks:
                fallback = "لا يوجد جواب واضح في الملفات المرفوعة."
                yield _sse("delta", {"text": fallback})
                yield _sse("done", {"sources": []})
                return

            prompt = build_rag_prompt(request.question, chunks)
            for delta in chat_service.stream_answer(prompt):
                yield _sse("delta", {"text": delta})

            yield _sse("done", {"sources": sources})
        except Exception:
            traceback.print_exc()
            yield _sse(
                "error",
                {"message": "Unable to generate the answer. Check server logs."},
            )

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
