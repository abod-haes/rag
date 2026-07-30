import json
import re
import traceback
from collections.abc import Iterator

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.api.chat_v2 import (
    _record_question_usage,
    list_question_usage as legacy_list_question_usage,
)
from app.core.config import get_settings
from app.core.request_scope import RequestScope, get_request_scope
from app.core.security import verify_api_key
from app.services.chat_service import ChatService
from app.services.conversation_service import (
    ConversationNotFoundError,
    ConversationService,
)
from app.services.document_routing_service import (
    DocumentRoutingError,
    DocumentRoutingResult,
    DocumentRoutingService,
)
from app.services.embedding_service import EmbeddingService
from app.services.prompt_service import build_rag_prompt
from app.services.retrieval_service import RetrievalService
from app.services.usage_service import TokenUsage


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
        history = conversation_service.get_history(
            conversation_id=conversation_id,
            user_id=scope.user_id,
            project_id=scope.project_id,
            limit=settings.conversation_history_messages,
        )
        active_document_ids = conversation_service.get_active_document_ids(
            conversation_id=conversation_id,
            user_id=scope.user_id,
            project_id=scope.project_id,
        )

        chat_service = ChatService()
        resolved_question, rewrite_usage = chat_service.rewrite_follow_up(
            question=request.question,
            history=history,
        )
        query_result = EmbeddingService().embed_query_with_usage(resolved_question)
        routing = DocumentRoutingService().route(
            query_embedding=query_result.values,
            query_text=resolved_question,
            user_id=scope.user_id,
            project_id=scope.project_id,
            explicit_document_ids=request.document_ids,
            active_document_ids=active_document_ids,
        )
    except ConversationNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except DocumentRoutingError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if routing.status == "clarification":
        return _handle_clarification(
            request=request,
            scope=scope,
            conversation_id=conversation_id,
            resolved_question=resolved_question,
            routing=routing,
            query_embedding_tokens=query_result.usage.input_tokens,
            rewrite_usage=rewrite_usage,
            conversation_service=conversation_service,
        )

    selected_document_ids = routing.selected_document_ids
    conversation_service.set_active_document_ids(
        conversation_id=conversation_id,
        user_id=scope.user_id,
        project_id=scope.project_id,
        document_ids=selected_document_ids,
    )

    chunks = RetrievalService().retrieve(
        query_embedding=query_result.values,
        query_text=resolved_question,
        user_id=scope.user_id,
        project_id=scope.project_id,
        document_ids=selected_document_ids,
    )
    prompt = build_rag_prompt(request.question, chunks, history=history)
    answer, answer_usage = chat_service.generate_answer_with_usage(prompt)

    candidate_sources = _build_sources(chunks)
    used_sources = _filter_used_sources(answer, candidate_sources)
    combined_chat_usage = rewrite_usage + answer_usage
    usage = _record_question_usage(
        scope=scope,
        question=request.question,
        document_ids=selected_document_ids,
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
        "needsClarification": False,
        "routingStatus": "selected",
        "resolvedQuestion": resolved_question,
        "selectedDocuments": routing.selected_documents,
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
            active_document_ids = conversation_service.get_active_document_ids(
                conversation_id=conversation_id,
                user_id=scope.user_id,
                project_id=scope.project_id,
            )
            yield _sse(
                "started",
                {
                    "message": "Processing question",
                    "conversationId": conversation_id,
                },
            )

            chat_service = ChatService()
            resolved_question, rewrite_usage = chat_service.rewrite_follow_up(
                question=request.question,
                history=history,
            )
            yield _sse("resolved_question", {"text": resolved_question})

            query_result = EmbeddingService().embed_query_with_usage(resolved_question)
            routing = DocumentRoutingService().route(
                query_embedding=query_result.values,
                query_text=resolved_question,
                user_id=scope.user_id,
                project_id=scope.project_id,
                explicit_document_ids=request.document_ids,
                active_document_ids=active_document_ids,
            )

            if routing.status == "clarification":
                clarification = routing.clarification_question or "Which subject are you asking about?"
                usage = _record_question_usage(
                    scope=scope,
                    question=request.question,
                    document_ids=[],
                    query_embedding_tokens=query_result.usage.input_tokens,
                    chat_usage=rewrite_usage,
                )
                conversation_service.add_message(
                    conversation_id=conversation_id,
                    role="user",
                    content=request.question,
                )
                conversation_service.add_message(
                    conversation_id=conversation_id,
                    role="assistant",
                    content=clarification,
                )
                clarification_payload = {
                    "conversationId": conversation_id,
                    "needsClarification": True,
                    "question": clarification,
                    "candidateSubjects": routing.candidate_subjects,
                    "candidateDocuments": routing.candidates,
                }
                yield _sse("clarification", clarification_payload)
                yield _sse("delta", {"text": clarification})
                yield _sse("usage", usage)
                yield _sse(
                    "done",
                    {
                        **clarification_payload,
                        "routingStatus": "clarification",
                        "sources": [],
                    },
                )
                return

            selected_document_ids = routing.selected_document_ids
            conversation_service.set_active_document_ids(
                conversation_id=conversation_id,
                user_id=scope.user_id,
                project_id=scope.project_id,
                document_ids=selected_document_ids,
            )
            yield _sse(
                "routing",
                {
                    "status": "selected",
                    "selectedDocuments": routing.selected_documents,
                },
            )

            chunks = RetrievalService().retrieve(
                query_embedding=query_result.values,
                query_text=resolved_question,
                user_id=scope.user_id,
                project_id=scope.project_id,
                document_ids=selected_document_ids,
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
                document_ids=selected_document_ids,
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
                    "needsClarification": False,
                    "routingStatus": "selected",
                    "selectedDocuments": routing.selected_documents,
                    "sources": used_sources,
                    "retrievedSourceCount": len(candidate_sources),
                },
            )
        except ConversationNotFoundError as exc:
            yield _sse("error", {"message": str(exc), "statusCode": 404})
        except DocumentRoutingError as exc:
            yield _sse("error", {"message": str(exc), "statusCode": 400})
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
    return legacy_list_question_usage(limit=limit, scope=scope)


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


def _handle_clarification(
    *,
    request: AskRequest,
    scope: RequestScope,
    conversation_id: str,
    resolved_question: str,
    routing: DocumentRoutingResult,
    query_embedding_tokens: int,
    rewrite_usage: TokenUsage,
    conversation_service: ConversationService,
) -> dict:
    clarification = routing.clarification_question or "Which subject are you asking about?"
    usage = _record_question_usage(
        scope=scope,
        question=request.question,
        document_ids=[],
        query_embedding_tokens=query_embedding_tokens,
        chat_usage=rewrite_usage,
    )
    conversation_service.add_message(
        conversation_id=conversation_id,
        role="user",
        content=request.question,
    )
    conversation_service.add_message(
        conversation_id=conversation_id,
        role="assistant",
        content=clarification,
    )
    return {
        "conversationId": conversation_id,
        "needsClarification": True,
        "routingStatus": "clarification",
        "resolvedQuestion": resolved_question,
        "clarificationQuestion": clarification,
        "candidateSubjects": routing.candidate_subjects,
        "candidateDocuments": routing.candidates,
        "selectedDocuments": [],
        "answer": clarification,
        "sources": [],
        "retrievedSourceCount": 0,
        "usage": usage,
    }


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
