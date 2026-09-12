import traceback
from collections.abc import Iterator

from fastapi import Depends
from fastapi.responses import StreamingResponse

from app.api.chat_v5 import (
    AskRequest,
    CURRICULUM,
    GENERAL_CHAT,
    NO_CONTEXT,
    _build_general_chat_prompt,
    _classify_intent,
    _curriculum_clarification_response,
    _record_question_usage,
    _routing_has_no_documents,
    _sse,
    _store_exchange,
    router,
)
from app.core.config import get_settings
from app.core.request_scope import RequestScope, get_request_scope
from app.services.answer_service import AnswerService
from app.services.chat_service import ChatService
from app.services.conversation_service import ConversationNotFoundError, ConversationService
from app.services.document_routing_service import DocumentRoutingError, DocumentRoutingService
from app.services.embedding_service import EmbeddingService
from app.services.retrieval_pipeline_service import RetrievalPipelineResult, RetrievalPipelineService
from app.services.usage_service import TokenUsage


# chat_v6 keeps all conversation/usage endpoints from v5 and replaces only the
# two answer-generation routes with the retrieval-quality pipeline.
router.routes[:] = [
    route
    for route in router.routes
    if not (
        getattr(route, "path", None) in {"/api/chat/ask", "/api/chat/stream"}
        and "POST" in (getattr(route, "methods", set()) or set())
    )
]


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
    except ConversationNotFoundError as exc:
        from fastapi import HTTPException

        raise HTTPException(status_code=404, detail=str(exc)) from exc

    chat_service = ChatService()
    intent, intent_usage = _classify_intent(
        chat_service=chat_service,
        question=request.question,
        history=history,
    )

    if intent == GENERAL_CHAT:
        answer, answer_usage = chat_service.generate_answer_with_usage(
            _build_general_chat_prompt(request.question, history)
        )
        usage = _record_question_usage(
            scope=scope,
            question=request.question,
            document_ids=[],
            query_embedding_tokens=0,
            chat_usage=intent_usage + answer_usage,
        )
        _store_exchange(
            conversation_service=conversation_service,
            conversation_id=conversation_id,
            question=request.question,
            answer=answer,
        )
        return {
            "conversationId": conversation_id,
            "needsClarification": False,
            "routingStatus": "general",
            "answerMode": GENERAL_CHAT,
            "hasContext": False,
            "resolvedQuestion": request.question,
            "selectedDocuments": [],
            "answer": answer,
            "sources": [],
            "retrievedSourceCount": 0,
            "usage": usage,
        }

    try:
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
    except DocumentRoutingError as exc:
        from fastapi import HTTPException

        raise HTTPException(status_code=400, detail=str(exc)) from exc

    hidden_usage = intent_usage + rewrite_usage
    if routing.status == "clarification":
        if _routing_has_no_documents(routing):
            return _no_context_response(
                request=request,
                scope=scope,
                conversation_id=conversation_id,
                resolved_question=resolved_question,
                document_ids=[],
                query_embedding_tokens=query_result.usage.input_tokens,
                chat_usage=hidden_usage,
                conversation_service=conversation_service,
            )
        return _curriculum_clarification_response(
            request=request,
            scope=scope,
            conversation_id=conversation_id,
            resolved_question=resolved_question,
            routing=routing,
            query_embedding_tokens=query_result.usage.input_tokens,
            chat_usage=hidden_usage,
            conversation_service=conversation_service,
        )

    selected_document_ids = routing.selected_document_ids
    conversation_service.set_active_document_ids(
        conversation_id=conversation_id,
        user_id=scope.user_id,
        project_id=scope.project_id,
        document_ids=selected_document_ids,
    )

    pipeline = RetrievalPipelineService(chat_service=chat_service).retrieve(
        query=resolved_question,
        query_embedding=query_result.values,
        user_id=scope.user_id,
        project_id=scope.project_id,
        document_ids=selected_document_ids,
        history=history,
    )
    hidden_usage = hidden_usage + pipeline.chat_usage
    total_embedding_tokens = (
        query_result.usage.input_tokens + pipeline.additional_embedding_tokens
    )

    if not pipeline.accepted:
        return _no_context_response(
            request=request,
            scope=scope,
            conversation_id=conversation_id,
            resolved_question=resolved_question,
            document_ids=selected_document_ids,
            query_embedding_tokens=total_embedding_tokens,
            chat_usage=hidden_usage,
            conversation_service=conversation_service,
            selected_documents=routing.selected_documents,
            pipeline=pipeline,
        )

    structured_answer, answer_usage = AnswerService(chat_service).generate(
        question=request.question,
        chunks=pipeline.chunks,
        history=history,
    )
    candidate_sources = _build_sources(pipeline.chunks)
    used_sources = _select_sources(
        candidate_sources,
        structured_answer.used_source_ids,
    )
    confidence = min(
        structured_answer.confidence,
        max(0.0, min(1.0, float(pipeline.gate.score))),
    )
    usage = _record_question_usage(
        scope=scope,
        question=request.question,
        document_ids=selected_document_ids,
        query_embedding_tokens=total_embedding_tokens,
        chat_usage=hidden_usage + answer_usage,
    )
    _store_exchange(
        conversation_service=conversation_service,
        conversation_id=conversation_id,
        question=request.question,
        answer=structured_answer.answer,
        sources=used_sources,
    )

    return {
        "conversationId": conversation_id,
        "needsClarification": False,
        "routingStatus": "selected",
        "answerMode": CURRICULUM,
        "hasContext": True,
        "resolvedQuestion": resolved_question,
        "selectedDocuments": routing.selected_documents,
        "answer": structured_answer.answer,
        "sources": used_sources,
        "retrievedSourceCount": len(candidate_sources),
        "confidence": round(confidence, 4),
        "groundingMode": structured_answer.grounding_mode,
        "retrievalDiagnostics": pipeline.diagnostics(),
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
            intent, intent_usage = _classify_intent(
                chat_service=chat_service,
                question=request.question,
                history=history,
            )
            yield _sse("mode", {"answerMode": intent, "hasContext": False})

            if intent == GENERAL_CHAT:
                answer_parts: list[str] = []
                prompt = _build_general_chat_prompt(request.question, history)
                for delta in chat_service.stream_answer(prompt):
                    answer_parts.append(delta)
                    yield _sse("delta", {"text": delta})

                answer = "".join(answer_parts)
                usage = _record_question_usage(
                    scope=scope,
                    question=request.question,
                    document_ids=[],
                    query_embedding_tokens=0,
                    chat_usage=intent_usage + chat_service.last_usage,
                )
                _store_exchange(
                    conversation_service=conversation_service,
                    conversation_id=conversation_id,
                    question=request.question,
                    answer=answer,
                )
                yield _sse("usage", usage)
                yield _sse(
                    "done",
                    {
                        "conversationId": conversation_id,
                        "needsClarification": False,
                        "routingStatus": "general",
                        "answerMode": GENERAL_CHAT,
                        "hasContext": False,
                        "selectedDocuments": [],
                        "sources": [],
                        "retrievedSourceCount": 0,
                    },
                )
                return

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
            hidden_usage = intent_usage + rewrite_usage

            if routing.status == "clarification":
                if _routing_has_no_documents(routing):
                    answer = _curriculum_fallback(request.question)
                    usage = _record_question_usage(
                        scope=scope,
                        question=request.question,
                        document_ids=[],
                        query_embedding_tokens=query_result.usage.input_tokens,
                        chat_usage=hidden_usage,
                    )
                    _store_exchange(
                        conversation_service=conversation_service,
                        conversation_id=conversation_id,
                        question=request.question,
                        answer=answer,
                    )
                    yield _sse("routing", {"status": NO_CONTEXT})
                    yield _sse("delta", {"text": answer})
                    yield _sse("usage", usage)
                    yield _sse(
                        "done",
                        {
                            "conversationId": conversation_id,
                            "needsClarification": False,
                            "routingStatus": NO_CONTEXT,
                            "answerMode": NO_CONTEXT,
                            "hasContext": False,
                            "selectedDocuments": [],
                            "sources": [],
                            "retrievedSourceCount": 0,
                        },
                    )
                    return

                clarification_response = _curriculum_clarification_response(
                    request=request,
                    scope=scope,
                    conversation_id=conversation_id,
                    resolved_question=resolved_question,
                    routing=routing,
                    query_embedding_tokens=query_result.usage.input_tokens,
                    chat_usage=hidden_usage,
                    conversation_service=conversation_service,
                )
                yield _sse("clarification", clarification_response)
                yield _sse("delta", {"text": clarification_response["answer"]})
                yield _sse("usage", clarification_response["usage"])
                yield _sse("done", clarification_response)
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
                    "answerMode": CURRICULUM,
                    "selectedDocuments": routing.selected_documents,
                },
            )

            pipeline = RetrievalPipelineService(chat_service=chat_service).retrieve(
                query=resolved_question,
                query_embedding=query_result.values,
                user_id=scope.user_id,
                project_id=scope.project_id,
                document_ids=selected_document_ids,
                history=history,
            )
            hidden_usage = hidden_usage + pipeline.chat_usage
            total_embedding_tokens = (
                query_result.usage.input_tokens + pipeline.additional_embedding_tokens
            )
            yield _sse("retrieval", pipeline.diagnostics())

            if not pipeline.accepted:
                answer = _curriculum_fallback(request.question)
                usage = _record_question_usage(
                    scope=scope,
                    question=request.question,
                    document_ids=selected_document_ids,
                    query_embedding_tokens=total_embedding_tokens,
                    chat_usage=hidden_usage,
                )
                _store_exchange(
                    conversation_service=conversation_service,
                    conversation_id=conversation_id,
                    question=request.question,
                    answer=answer,
                )
                yield _sse("sources", [])
                yield _sse("delta", {"text": answer})
                yield _sse("usage", usage)
                yield _sse(
                    "done",
                    {
                        "conversationId": conversation_id,
                        "needsClarification": False,
                        "routingStatus": NO_CONTEXT,
                        "answerMode": NO_CONTEXT,
                        "hasContext": False,
                        "selectedDocuments": routing.selected_documents,
                        "sources": [],
                        "retrievedSourceCount": 0,
                        "retrievalDiagnostics": pipeline.diagnostics(),
                    },
                )
                return

            structured_answer, answer_usage = AnswerService(chat_service).generate(
                question=request.question,
                chunks=pipeline.chunks,
                history=history,
            )
            candidate_sources = _build_sources(pipeline.chunks)
            used_sources = _select_sources(
                candidate_sources,
                structured_answer.used_source_ids,
            )
            confidence = min(
                structured_answer.confidence,
                max(0.0, min(1.0, float(pipeline.gate.score))),
            )
            usage = _record_question_usage(
                scope=scope,
                question=request.question,
                document_ids=selected_document_ids,
                query_embedding_tokens=total_embedding_tokens,
                chat_usage=hidden_usage + answer_usage,
            )
            _store_exchange(
                conversation_service=conversation_service,
                conversation_id=conversation_id,
                question=request.question,
                answer=structured_answer.answer,
                sources=used_sources,
            )

            yield _sse("sources", used_sources)
            yield _sse("mode", {"answerMode": CURRICULUM, "hasContext": True})
            for delta in _chunk_answer(structured_answer.answer):
                yield _sse("delta", {"text": delta})
            yield _sse("usage", usage)
            yield _sse(
                "done",
                {
                    "conversationId": conversation_id,
                    "needsClarification": False,
                    "routingStatus": "selected",
                    "answerMode": CURRICULUM,
                    "hasContext": True,
                    "selectedDocuments": routing.selected_documents,
                    "sources": used_sources,
                    "retrievedSourceCount": len(candidate_sources),
                    "confidence": round(confidence, 4),
                    "groundingMode": structured_answer.grounding_mode,
                    "retrievalDiagnostics": pipeline.diagnostics(),
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


def _curriculum_fallback(question: str) -> str:
    if any("\u0600" <= char <= "\u06ff" for char in question or ""):
        return (
            "هالمعلومة مو موجودة ضمن منهاجك الحالي. "
            "إذا بدك، اسألني عن درس أو فكرة موجودة بمنهاجك وبساعدك فيها."
        )
    return (
        "This information is not in your current curriculum. "
        "Ask me about a lesson or idea that is in your curriculum and I'll help you with it."
    )


def _no_context_response(
    *,
    request: AskRequest,
    scope: RequestScope,
    conversation_id: str,
    resolved_question: str,
    document_ids: list[str],
    query_embedding_tokens: int,
    chat_usage: TokenUsage,
    conversation_service: ConversationService,
    selected_documents: list[dict] | None = None,
    pipeline: RetrievalPipelineResult | None = None,
) -> dict:
    answer = _curriculum_fallback(request.question)
    usage = _record_question_usage(
        scope=scope,
        question=request.question,
        document_ids=document_ids,
        query_embedding_tokens=query_embedding_tokens,
        chat_usage=chat_usage,
    )
    _store_exchange(
        conversation_service=conversation_service,
        conversation_id=conversation_id,
        question=request.question,
        answer=answer,
    )
    return {
        "conversationId": conversation_id,
        "needsClarification": False,
        "routingStatus": NO_CONTEXT,
        "answerMode": NO_CONTEXT,
        "hasContext": False,
        "resolvedQuestion": resolved_question,
        "selectedDocuments": selected_documents or [],
        "answer": answer,
        "sources": [],
        "retrievedSourceCount": 0,
        "retrievalDiagnostics": pipeline.diagnostics() if pipeline else None,
        "usage": usage,
    }


def _build_sources(chunks: list[dict]) -> list[dict]:
    sources: list[dict] = []
    for index, chunk in enumerate(chunks, start=1):
        sources.append(
            {
                "sourceId": f"S{index}",
                "documentId": chunk["document_id"],
                "name": chunk.get("name") or chunk.get("file_name"),
                "fileName": chunk.get("file_name"),
                "pageNumber": chunk.get("page_number"),
                "chunkIndex": chunk.get("chunk_index"),
                "sectionTitle": chunk.get("section_title"),
                "contentType": chunk.get("content_type") or "text",
                "isNeighbor": bool(chunk.get("is_neighbor")),
                "isParent": bool(chunk.get("is_parent")),
                "retrievalRelation": chunk.get("retrieval_relation") or "direct",
                "score": round(float(chunk.get("score") or 0.0), 4),
                "rerankScore": round(float(chunk.get("rerank_score") or 0.0), 4),
            }
        )
    return sources


def _select_sources(sources: list[dict], source_ids: list[str]) -> list[dict]:
    wanted = {source_id.upper() for source_id in source_ids}
    return [source for source in sources if source["sourceId"].upper() in wanted]


def _chunk_answer(answer: str, size: int = 96) -> Iterator[str]:
    if not answer:
        return
    for start in range(0, len(answer), max(24, size)):
        yield answer[start : start + max(24, size)]
