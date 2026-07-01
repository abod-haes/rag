from pydantic import BaseModel, Field
from fastapi import APIRouter, Depends

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
    question: str
    document_ids: list[str] | None = Field(default=None, alias="documentIds")


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

    sources = [
        {
            "documentId": chunk["document_id"],
            "fileName": chunk["file_name"],
            "pageNumber": chunk["page_number"],
            "chunkIndex": chunk["chunk_index"],
            "score": round(chunk["score"], 4),
        }
        for chunk in chunks
    ]

    return {"answer": answer, "sources": sources}
