from fastapi import Depends, HTTPException, status
from pydantic import BaseModel, Field

from app.api.chat_v3 import router
from app.core.request_scope import RequestScope, get_request_scope
from app.services.conversation_service import (
    ConversationNotFoundError,
    ConversationService,
)


class CreateConversationRequest(BaseModel):
    title: str | None = Field(default=None, max_length=100)


@router.post("/conversations", status_code=status.HTTP_201_CREATED)
def create_conversation(
    request: CreateConversationRequest | None = None,
    scope: RequestScope = Depends(get_request_scope),
):
    return ConversationService().create_conversation(
        user_id=scope.user_id,
        project_id=scope.project_id,
        title=request.title if request else None,
    )


@router.delete("/conversations/{conversation_id}")
def delete_conversation(
    conversation_id: str,
    scope: RequestScope = Depends(get_request_scope),
):
    try:
        ConversationService().delete_conversation(
            conversation_id=conversation_id,
            user_id=scope.user_id,
            project_id=scope.project_id,
        )
    except ConversationNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"success": True}
