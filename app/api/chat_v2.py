import json
import re
import traceback
import uuid
from collections.abc import Iterator
from decimal import Decimal

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.core.config import get_settings
from app.core.request_scope import RequestScope, get_request_scope
from app.core.security import verify_api_key
from app.db.database import dict_cursor, get_connection
from app.services.chat_service import ChatService
from app.services.conversation_service import (
    ConversationNotFoundError,
    ConversationService,
)
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

SOURCE_REFERENCE_RE = re.compile(r"\[(S\d+)(?=[,\]\s])", re.IGNORECASE)


class AskRequest(BaseModel):
    question: str = Field(min_length=1)
    document_ids: list[str] | None = Field(default=None, alias="documentIds")
    conversation_id: str | None = Field(default=None, alias="conversationId")


def _build_sources(chunks: list[dict]) -> list[dict]:
    sources: list[dict] = []
    for index, chunk in enumerate(chunks, start=1):
        sources.append(
            {
                "sourceId": f"S{index}",
                "documentId": chunk["document_id"],
                "name": chunk.get("name") or chunk["file_name"],
                "fileName": chunk["file_name"],
                "pageNumber": chunk["page_number"],
                "chunkIndex": chunk["chunk_index"],
                "sectionTitle": chunk.get("section_title"),
                "contentType": chunk.get("content_type") or "text",
                "isNeighbor": bool(chunk.get("is_neighbor")),
                "score": round(float(chunk.get("score") or 0.0), 4),
            }
        )
    return sources


def _filter_used_sources(answer: str, sources: list[dict]) -> list[dict]:
    referenced_ids = {
        match.upper() for match in SOURCE_REFERENCE_RE.findall(answer or "")
    }
    if not referenced_ids:
        return []
    return [source for source in sources if source["sourceId"] in referenced_ids]


def _sse(event: str, data: dict | list) -> str:
    payload = json.dumps(data, ensure_ascii=False)
    return f"event: {event}\ndata: {payload}\n\n"


@router.post("/ask")
def ask_question(
    request: AskRequest,
    scope: RequestScope = Depends(get_request_scope),
):
    settings = get_settings()
    conversation_service = ConversationService()
    try:
        conversation_id = conversation_service.ensure_conversation(
            conversation_id=request.conversation_id,
            user_id=scope.user_id,
            project_id=scope.project_id,
            first_question=request.question,
        )
    except ConversationNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    history = conversation_service.get_history(
        conversation_id=conversation_id,
        user_id=scope.user_id,
        project_id=scope.project_id,
        limit=settings.conversation_history_messages,
    )

    embedding_service = EmbeddingService()
    retrieval_service = RetrievalService()
    chat_service = ChatService()

    resolved_question, rewrite_usage = chat_service.rewrite_follow_up(
        question=request.question,
        history=history,
    )
    query_result = embedding_service.embed_query_with_usage(resolved_question)
    chunks = retrieval_service.retrieve(
        query_embedding=query_result.values,
        query_text=resolved_question,
        user_id=scope.user_id,
        project_id=scope.project_id,
        document_ids=request.document_ids,
    )

    prompt = build_rag_prompt(request.question, chunks, history=history)
    answer, answer_usage = chat_service.generate_answer_with_usage(prompt)

    candidate_sources = _build_sources(chunks)
    used_sources = _filter_used_sources(answer, candidate_sources)
    combined_chat_usage = rewrite_usage + answer_usage
    usage = _record_question_usage(
        scope=scope,
        question=request.question,
        document_ids=request.document_ids,
        query_embedding_tokens=query_result.usage.input_tokens,
        chat_usage=combined_chat_usage,
    )

    conversation_service.add_message(
        conversation_id=conversation_id,
        role="user",
        content=request.question,
    )
    conversation_service.add_message(
        conversation_id=conversation_id,
        role="assistant",
        content=answer,
        sources=used_sources,
    )

    return {
        "conversationId": conversation_id,
        "resolvedQuestion": resolved_question,
        "answer": answer,
        "sources": used_sources,
        "retrievedSourceCount": len(candidate_sources),
        "usage": usage,
    }


@router.post("/stream")
def stream_question(
    request: AskRequest,
    scope: RequestScope = Depends(get_request_scope),
):
    def event_stream() -> Iterator[str]:
        try:
            settings = get_settings()
            conversation_service = ConversationService()
            conversation_id = conversation_service.ensure_conversation(
                conversation_id=request.conversation_id,
                user_id=scope.user_id,
                project_id=scope.project_id,
                first_question=request.question,
            )
            history = conversation_service.get_history(
                conversation_id=conversation_id,
                user_id=scope.user_id,
                project_id=scope.project_id,
                limit=settings.conversation_history_messages,
            )
            yield _sse(
                "started",
                {
                    "message": "Processing question",
                    "conversationId": conversation_id,
                },
            )

            embedding_service = EmbeddingService()
            retrieval_service = RetrievalService()
            chat_service = ChatService()

            resolved_question, rewrite_usage = chat_service.rewrite_follow_up(
                question=request.question,
                history=history,
            )
            yield _sse("resolved_question", {"text": resolved_question})

            query_result = embedding_service.embed_query_with_usage(resolved_question)
            chunks = retrieval_service.retrieve(
                query_embedding=query_result.values,
                query_text=resolved_question,
                user_id=scope.user_id,
                project_id=scope.project_id,
                document_ids=request.document_ids,
            )
            candidate_sources = _build_sources(chunks)
            yield _sse("sources", candidate_sources)

            prompt = build_rag_prompt(request.question, chunks, history=history)
            answer_parts: list[str] = []
            for delta in chat_service.stream_answer(prompt):
                answer_parts.append(delta)
                yield _sse("delta", {"text": delta})

            answer = "".join(answer_parts)
            used_sources = _filter_used_sources(answer, candidate_sources)
            combined_chat_usage = rewrite_usage + chat_service.last_usage
            usage = _record_question_usage(
                scope=scope,
                question=request.question,
                document_ids=request.document_ids,
                query_embedding_tokens=query_result.usage.input_tokens,
                chat_usage=combined_chat_usage,
            )
            conversation_service.add_message(
                conversation_id=conversation_id,
                role="user",
                content=request.question,
            )
            conversation_service.add_message(
                conversation_id=conversation_id,
                role="assistant",
                content=answer,
                sources=used_sources,
            )

            yield _sse("usage", usage)
            yield _sse(
                "done",
                {
                    "conversationId": conversation_id,
                    "sources": used_sources,
                    "retrievedSourceCount": len(candidate_sources),
                },
            )
        except ConversationNotFoundError as exc:
            yield _sse("error", {"message": str(exc), "statusCode": 404})
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


@router.get("/usage")
def list_question_usage(
    limit: int = Query(default=50, ge=1, le=200),
    scope: RequestScope = Depends(get_request_scope),
):
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
            (scope.user_id, scope.project_id, limit),
        )
        rows = cursor.fetchall()

    return [_build_question_usage_from_row(row) for row in rows]


@router.get("/conversations")
def list_conversations(
    limit: int = Query(default=50, ge=1, le=200),
    scope: RequestScope = Depends(get_request_scope),
):
    return ConversationService().list_conversations(
        user_id=scope.user_id,
        project_id=scope.project_id,
        limit=limit,
    )


@router.get("/conversations/{conversation_id}/messages")
def list_conversation_messages(
    conversation_id: str,
    limit: int = Query(default=100, ge=1, le=500),
    scope: RequestScope = Depends(get_request_scope),
):
    service = ConversationService()
    try:
        service.ensure_conversation(
            conversation_id=conversation_id,
            user_id=scope.user_id,
            project_id=scope.project_id,
            first_question="",
        )
    except ConversationNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    return service.get_history(
        conversation_id=conversation_id,
        user_id=scope.user_id,
        project_id=scope.project_id,
        limit=limit,
    )


def _record_question_usage(
    *,
    scope: RequestScope,
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
                scope.user_id,
                scope.project_id,
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
