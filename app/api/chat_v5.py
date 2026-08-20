import re
import traceback
from collections.abc import Iterator

from fastapi import Depends
from fastapi.responses import StreamingResponse

from app.api.chat_v2 import _record_question_usage
from app.api.chat_v3 import (
    AskRequest,
    _build_sources,
    _filter_used_sources,
    _sse,
)
from app.api.chat_v4 import router
from app.core.config import get_settings
from app.core.request_scope import RequestScope, get_request_scope
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


GENERAL_CHAT = "general"
CURRICULUM = "curriculum"
NO_CONTEXT = "no_context"
ARABIC_RE = re.compile(r"[\u0600-\u06ff]")
ACADEMIC_HINT_RE = re.compile(
    r"(?:"
    r"رياضيات|فيزياء|كيمياء|أحياء|احياء|بيولوجيا|جبر|هندسة|تفاضل|تكامل|"
    r"مشتق|متتال|احتمال|معادلة|دالة|نيوتن|ميكانيك|كهرباء|مغناطيس|"
    r"ذرة|جزي|تفاعل|وراثة|خلية|نحو|صرف|بلاغة|برمجة|خوارزم|شبكات|"
    r"قواعد\s*بيانات|تاريخ|جغرافيا|فلسفة|"
    r"math|physics|chemistry|biology|algebra|geometry|calculus|derivative|"
    r"integral|equation|function|newton|mechanics|electric|magnet|atom|"
    r"molecule|genetics|cell|grammar|programming|algorithm|network|database"
    r")",
    re.IGNORECASE,
)
MATH_EXPRESSION_RE = re.compile(
    r"(?:\d\s*[+\-*/=^]\s*\d|[xyz]\s*[=+\-*/^]|\\frac|√|∫|∑)",
    re.IGNORECASE,
)


# chat_v4 extends the v3 router. Replace only the two answer-generation routes
# while keeping conversations, usage, and the rest of the API untouched.
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
            return _curriculum_no_context_response(
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

    chunks = RetrievalService().retrieve(
        query_embedding=query_result.values,
        query_text=resolved_question,
        user_id=scope.user_id,
        project_id=scope.project_id,
        document_ids=selected_document_ids,
    )
    if not chunks:
        return _curriculum_no_context_response(
            request=request,
            scope=scope,
            conversation_id=conversation_id,
            resolved_question=resolved_question,
            document_ids=selected_document_ids,
            query_embedding_tokens=query_result.usage.input_tokens,
            chat_usage=hidden_usage,
            conversation_service=conversation_service,
            selected_documents=routing.selected_documents,
        )

    prompt = build_rag_prompt(request.question, chunks, history=history)
    answer, answer_usage = chat_service.generate_answer_with_usage(prompt)

    candidate_sources = _build_sources(chunks)
    used_sources = _filter_used_sources(answer, candidate_sources)
    usage = _record_question_usage(
        scope=scope,
        question=request.question,
        document_ids=selected_document_ids,
        query_embedding_tokens=query_result.usage.input_tokens,
        chat_usage=hidden_usage + answer_usage,
    )
    _store_exchange(
        conversation_service=conversation_service,
        conversation_id=conversation_id,
        question=request.question,
        answer=answer,
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
            intent, intent_usage = _classify_intent(
                chat_service=chat_service,
                question=request.question,
                history=history,
            )
            yield _sse(
                "mode",
                {"answerMode": intent, "hasContext": False},
            )

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

                clarification = routing.clarification_question or _clarification_fallback(
                    request.question
                )
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
                    answer=clarification,
                )
                clarification_payload = {
                    "conversationId": conversation_id,
                    "needsClarification": True,
                    "question": clarification,
                    "candidateSubjects": routing.candidate_subjects,
                    "candidateDocuments": routing.candidates,
                    "answerMode": CURRICULUM,
                    "hasContext": False,
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
                    "answerMode": CURRICULUM,
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
            if not chunks:
                answer = _curriculum_fallback(request.question)
                usage = _record_question_usage(
                    scope=scope,
                    question=request.question,
                    document_ids=selected_document_ids,
                    query_embedding_tokens=query_result.usage.input_tokens,
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
                    },
                )
                return

            candidate_sources = _build_sources(chunks)
            yield _sse("sources", candidate_sources)
            yield _sse(
                "mode",
                {"answerMode": CURRICULUM, "hasContext": True},
            )

            prompt = build_rag_prompt(request.question, chunks, history=history)
            answer_parts: list[str] = []
            for delta in chat_service.stream_answer(prompt):
                answer_parts.append(delta)
                yield _sse("delta", {"text": delta})

            answer = "".join(answer_parts)
            used_sources = _filter_used_sources(answer, candidate_sources)
            usage = _record_question_usage(
                scope=scope,
                question=request.question,
                document_ids=selected_document_ids,
                query_embedding_tokens=query_result.usage.input_tokens,
                chat_usage=hidden_usage + chat_service.last_usage,
            )
            _store_exchange(
                conversation_service=conversation_service,
                conversation_id=conversation_id,
                question=request.question,
                answer=answer,
                sources=used_sources,
            )

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


def _classify_intent(
    *,
    chat_service: ChatService,
    question: str,
    history: list[dict],
) -> tuple[str, TokenUsage]:
    prompt = _build_intent_prompt(question, history)
    raw_label, usage = chat_service.generate_answer_with_usage(prompt)
    first_token = re.split(r"[^A-Z_]", (raw_label or "").strip().upper(), maxsplit=1)[0]

    if first_token == "CURRICULUM":
        return CURRICULUM, usage
    if first_token == "GENERAL_CHAT":
        return GENERAL_CHAT, usage
    return _fallback_intent(question, history), usage


def _build_intent_prompt(question: str, history: list[dict]) -> str:
    history_text = _format_history(history)
    return f"""
You are a routing classifier for Quizy, an educational assistant.
Classify the LATEST USER MESSAGE into exactly one label:

GENERAL_CHAT
- greetings, thanks, casual conversation, everyday advice, product/help questions
- ordinary non-academic general knowledge or trivia not framed as a lesson or exercise
- examples: "مرحبا", "كيفك؟", "شو عاصمة فرنسا؟", "thank you"

CURRICULUM
- school/university learning questions
- scientific, mathematical, technical, language/grammar, or other academic explanations
- exercises, formulas, derivations, definitions, homework, lesson/course/book questions
- follow-ups whose meaning is academic because of the conversation history
- examples: "اشرح قانون نيوتن الثاني", "حل x^2-4=0", "شو وظيفة الميتوكوندريا؟"

Important:
- If the message is plausibly an academic/scientific learning question, choose CURRICULUM.
- Do not answer the user.
- Return exactly GENERAL_CHAT or CURRICULUM and nothing else.

RECENT CONVERSATION:
{history_text}

LATEST USER MESSAGE:
{question}
""".strip()


def _build_general_chat_prompt(question: str, history: list[dict]) -> str:
    return f"""
You are Quizy, a friendly and concise study companion.
This message has been routed as general conversation, so answer naturally without
using or claiming to use curriculum documents.

Rules:
- Answer in the same language as the user's latest message.
- Be warm, direct, and useful without being overly formal.
- You may handle greetings, casual conversation, everyday advice, and stable
  general-knowledge questions normally.
- Never mention RAG, embeddings, indexing, chunks, vector search, or internal files.
- Do not claim access to live/current information unless it is actually provided.
- Conversation history is only for continuity; the latest user message has priority.

RECENT CONVERSATION:
{_format_history(history)}

LATEST USER MESSAGE:
{question}
""".strip()


def _fallback_intent(question: str, history: list[dict]) -> str:
    combined = "\n".join(
        [question]
        + [str(item.get("content") or "") for item in history[-4:]]
    )
    if ACADEMIC_HINT_RE.search(combined) or MATH_EXPRESSION_RE.search(question or ""):
        return CURRICULUM
    return GENERAL_CHAT


def _format_history(history: list[dict]) -> str:
    if not history:
        return "No previous conversation messages."

    lines: list[str] = []
    for item in history[-8:]:
        role = "User" if item.get("role") == "user" else "Assistant"
        content = " ".join(str(item.get("content") or "").split())
        if content:
            lines.append(f"{role}: {content}")
    return "\n".join(lines) or "No previous conversation messages."


def _routing_has_no_documents(routing: DocumentRoutingResult) -> bool:
    return not (
        routing.selected_documents
        or routing.candidates
        or routing.candidate_subjects
    )


def _curriculum_fallback(question: str) -> str:
    if ARABIC_RE.search(question or ""):
        return (
            "ما لقيت هالمعلومة ضمن المحتوى الدراسي المتاح عندي حالياً. "
            "جرّب تسألني عن درس موجود ضمن موادك."
        )
    return (
        "I couldn't find this information in the study content currently available to me. "
        "Try asking about a lesson that's available in your materials."
    )


def _clarification_fallback(question: str) -> str:
    if ARABIC_RE.search(question or ""):
        return "أي مادة أو ملف دراسي تقصد؟"
    return "Which subject or study document do you mean?"


def _curriculum_no_context_response(
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
        "usage": usage,
    }


def _curriculum_clarification_response(
    *,
    request: AskRequest,
    scope: RequestScope,
    conversation_id: str,
    resolved_question: str,
    routing: DocumentRoutingResult,
    query_embedding_tokens: int,
    chat_usage: TokenUsage,
    conversation_service: ConversationService,
) -> dict:
    clarification = routing.clarification_question or _clarification_fallback(
        request.question
    )
    usage = _record_question_usage(
        scope=scope,
        question=request.question,
        document_ids=[],
        query_embedding_tokens=query_embedding_tokens,
        chat_usage=chat_usage,
    )
    _store_exchange(
        conversation_service=conversation_service,
        conversation_id=conversation_id,
        question=request.question,
        answer=clarification,
    )
    return {
        "conversationId": conversation_id,
        "needsClarification": True,
        "routingStatus": "clarification",
        "answerMode": CURRICULUM,
        "hasContext": False,
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


def _store_exchange(
    *,
    conversation_service: ConversationService,
    conversation_id: str,
    question: str,
    answer: str,
    sources: list[dict] | None = None,
) -> None:
    conversation_service.add_message(
        conversation_id=conversation_id,
        role="user",
        content=question,
    )
    conversation_service.add_message(
        conversation_id=conversation_id,
        role="assistant",
        content=answer,
        sources=sources,
    )
