import json
import traceback
import uuid
from collections.abc import Iterator
from decimal import Decimal

from pydantic import BaseModel, Field
from fastapi import APIRouter, Depends, Query
from fastapi.responses import StreamingResponse

from app.api.documents import DEFAULT_PROJECT_ID, DEFAULT_USER_ID
from app.core.config import get_settings
from app.core.security import verify_api_key
from app.db.database import dict_cursor, get_connection
from app.services.chat_service import GeminiChatService
from app.services.embedding_service import EmbeddingService
from app.services.prompt_service import build_rag_prompt
from app.services.retrieval_service import RetrievalService
from app.services.usage_service import (
    TokenUsage,
    decimal_to_json,
    estimate_chat_cost_usd,
    estimate_embedding_cost_usd,
)

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

    query_result = embedding_service.embed_query_with_usage(request.question)
    chunks = retrieval_service.retrieve(
        query_embedding=query_result.values,
        user_id=DEFAULT_USER_ID,
        project_id=DEFAULT_PROJECT_ID,
        document_ids=request.document_ids,
    )

    if not chunks:
        usage = _record_question_usage(
            question=request.question,
            document_ids=request.document_ids,
            query_embedding_tokens=query_result.usage.input_tokens,
            chat_usage=TokenUsage(),
        )
        return {
            "answer": "لا يوجد جواب واضح في الملفات المرفوعة.",
            "sources": [],
            "usage": usage,
        }

    prompt = build_rag_prompt(request.question, chunks)
    answer, chat_usage = chat_service.generate_answer_with_usage(prompt)

    sources = _build_sources(chunks)
    usage = _record_question_usage(
        question=request.question,
        document_ids=request.document_ids,
        query_embedding_tokens=query_result.usage.input_tokens,
        chat_usage=chat_usage,
    )

    return {"answer": answer, "sources": sources, "usage": usage}


@router.get("/usage")
def list_question_usage(limit: int = Query(default=50, ge=1, le=200)):
    with get_connection(cursor_factory=dict_cursor()) as (_, cursor):
        cursor.execute(
            """
            SELECT
                id::text,
                question,
                document_ids,
                ai_provider,
                embedding_model,
                chat_model,
                query_embedding_tokens,
                input_tokens,
                cached_input_tokens,
                output_tokens,
                total_tokens,
                estimated_cost_usd,
                created_at
            FROM chat_usage
            WHERE user_id = %s AND project_id = %s
            ORDER BY created_at DESC
            LIMIT %s
            """,
            (DEFAULT_USER_ID, DEFAULT_PROJECT_ID, limit),
        )
        rows = cursor.fetchall()

    return [_build_question_usage_from_row(row) for row in rows]


@router.post("/stream")
def stream_question(request: AskRequest):
    def event_stream() -> Iterator[str]:
        try:
            yield _sse("started", {"message": "Processing question"})

            embedding_service = EmbeddingService()
            retrieval_service = RetrievalService()
            chat_service = GeminiChatService()

            query_result = embedding_service.embed_query_with_usage(
                request.question
            )
            chunks = retrieval_service.retrieve(
                query_embedding=query_result.values,
                user_id=DEFAULT_USER_ID,
                project_id=DEFAULT_PROJECT_ID,
                document_ids=request.document_ids,
            )
            sources = _build_sources(chunks)
            yield _sse("sources", sources)

            if not chunks:
                fallback = "لا يوجد جواب واضح في الملفات المرفوعة."
                yield _sse("delta", {"text": fallback})
                usage = _record_question_usage(
                    question=request.question,
                    document_ids=request.document_ids,
                    query_embedding_tokens=query_result.usage.input_tokens,
                    chat_usage=TokenUsage(),
                )
                yield _sse("usage", usage)
                yield _sse("done", {"sources": []})
                return

            prompt = build_rag_prompt(request.question, chunks)
            for delta in chat_service.stream_answer(prompt):
                yield _sse("delta", {"text": delta})

            usage = _record_question_usage(
                question=request.question,
                document_ids=request.document_ids,
                query_embedding_tokens=query_result.usage.input_tokens,
                chat_usage=chat_service.last_usage,
            )
            yield _sse("usage", usage)
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


def _record_question_usage(
    *,
    question: str,
    document_ids: list[str] | None,
    query_embedding_tokens: int,
    chat_usage: TokenUsage,
) -> dict:
    settings = get_settings()
    provider = settings.ai_provider.lower().strip()
    embedding_model = (
        settings.openai_embedding_model
        if provider == "openai"
        else settings.gemini_embedding_model
    )
    chat_model = (
        settings.openai_chat_model
        if provider == "openai"
        else settings.gemini_chat_model
    )
    estimated_cost_usd = estimate_embedding_cost_usd(
        query_embedding_tokens,
        provider=provider,
        settings=settings,
    ) + estimate_chat_cost_usd(
        chat_usage,
        provider=provider,
        settings=settings,
    )
    usage_id = str(uuid.uuid4())
    total_tokens = query_embedding_tokens + chat_usage.total_tokens

    with get_connection() as (_, cursor):
        cursor.execute(
            """
            INSERT INTO chat_usage (
                id,
                user_id,
                project_id,
                question,
                document_ids,
                ai_provider,
                embedding_model,
                chat_model,
                query_embedding_tokens,
                input_tokens,
                cached_input_tokens,
                output_tokens,
                total_tokens,
                estimated_cost_usd
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                usage_id,
                DEFAULT_USER_ID,
                DEFAULT_PROJECT_ID,
                question,
                document_ids,
                provider,
                embedding_model,
                chat_model,
                query_embedding_tokens,
                chat_usage.input_tokens,
                chat_usage.cached_input_tokens,
                chat_usage.output_tokens,
                total_tokens,
                estimated_cost_usd,
            ),
        )

    return _build_question_usage(
        usage_id=usage_id,
        provider=provider,
        embedding_model=embedding_model,
        chat_model=chat_model,
        query_embedding_tokens=query_embedding_tokens,
        input_tokens=chat_usage.input_tokens,
        cached_input_tokens=chat_usage.cached_input_tokens,
        output_tokens=chat_usage.output_tokens,
        total_tokens=total_tokens,
        estimated_cost_usd=estimated_cost_usd,
    )


def _build_question_usage_from_row(row: dict) -> dict:
    return {
        "id": row["id"],
        "question": row["question"],
        "documentIds": row["document_ids"] or [],
        "usage": _build_question_usage(
            usage_id=row["id"],
            provider=row["ai_provider"],
            embedding_model=row["embedding_model"],
            chat_model=row["chat_model"],
            query_embedding_tokens=row["query_embedding_tokens"],
            input_tokens=row["input_tokens"],
            cached_input_tokens=row["cached_input_tokens"],
            output_tokens=row["output_tokens"],
            total_tokens=row["total_tokens"],
            estimated_cost_usd=row["estimated_cost_usd"],
        ),
        "createdAt": row["created_at"].isoformat() if row["created_at"] else None,
    }


def _build_question_usage(
    *,
    usage_id,
    provider,
    embedding_model,
    chat_model,
    query_embedding_tokens,
    input_tokens,
    cached_input_tokens,
    output_tokens,
    total_tokens,
    estimated_cost_usd,
) -> dict:
    return {
        "usageId": usage_id,
        "provider": provider,
        "embeddingModel": embedding_model,
        "chatModel": chat_model,
        "queryEmbeddingTokens": int(query_embedding_tokens or 0),
        "inputTokens": int(input_tokens or 0),
        "cachedInputTokens": int(cached_input_tokens or 0),
        "outputTokens": int(output_tokens or 0),
        "totalTokens": int(total_tokens or 0),
        "estimatedCostUsd": decimal_to_json(
            Decimal(str(estimated_cost_usd or 0))
        ),
        "costTrackingAvailable": provider == "openai",
    }
