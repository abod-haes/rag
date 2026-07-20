import json
import uuid

from app.db.database import dict_cursor, get_connection


class ConversationNotFoundError(Exception):
    pass


class ConversationService:
    def ensure_conversation(
        self,
        *,
        conversation_id: str | None,
        user_id: str,
        project_id: str,
        first_question: str,
    ) -> str:
        if conversation_id:
            normalized_id = _normalize_uuid(conversation_id)
            with get_connection(cursor_factory=dict_cursor()) as (_, cursor):
                cursor.execute(
                    """
                    SELECT id::text
                    FROM chat_conversations
                    WHERE id = %s AND user_id = %s AND project_id = %s
                    """,
                    (normalized_id, user_id, project_id),
                )
                row = cursor.fetchone()
            if not row:
                raise ConversationNotFoundError("Conversation not found")
            return row["id"]

        new_id = str(uuid.uuid4())
        title = _build_title(first_question)
        with get_connection() as (_, cursor):
            cursor.execute(
                """
                INSERT INTO chat_conversations (id, user_id, project_id, title)
                VALUES (%s, %s, %s, %s)
                """,
                (new_id, user_id, project_id, title),
            )
        return new_id

    def get_history(
        self,
        *,
        conversation_id: str,
        user_id: str,
        project_id: str,
        limit: int = 6,
    ) -> list[dict]:
        normalized_id = _normalize_uuid(conversation_id)
        with get_connection(cursor_factory=dict_cursor()) as (_, cursor):
            cursor.execute(
                """
                SELECT role, content, sources, created_at
                FROM chat_messages
                WHERE conversation_id = %s
                  AND EXISTS (
                      SELECT 1
                      FROM chat_conversations c
                      WHERE c.id = chat_messages.conversation_id
                        AND c.user_id = %s
                        AND c.project_id = %s
                  )
                ORDER BY created_at DESC
                LIMIT %s
                """,
                (normalized_id, user_id, project_id, limit),
            )
            rows = list(cursor.fetchall())

        rows.reverse()
        return [
            {
                "role": row["role"],
                "content": row["content"],
                "sources": row["sources"] or [],
                "createdAt": row["created_at"].isoformat() if row["created_at"] else None,
            }
            for row in rows
        ]

    def add_message(
        self,
        *,
        conversation_id: str,
        role: str,
        content: str,
        sources: list[dict] | None = None,
    ) -> None:
        if role not in {"user", "assistant"}:
            raise ValueError("Conversation role must be user or assistant")

        normalized_id = _normalize_uuid(conversation_id)
        message_id = str(uuid.uuid4())
        with get_connection() as (_, cursor):
            cursor.execute(
                """
                INSERT INTO chat_messages (id, conversation_id, role, content, sources)
                VALUES (%s, %s, %s, %s, %s::jsonb)
                """,
                (
                    message_id,
                    normalized_id,
                    role,
                    content,
                    json.dumps(sources or [], ensure_ascii=False),
                ),
            )
            cursor.execute(
                """
                UPDATE chat_conversations
                SET updated_at = NOW()
                WHERE id = %s
                """,
                (normalized_id,),
            )

    def list_conversations(
        self,
        *,
        user_id: str,
        project_id: str,
        limit: int = 50,
    ) -> list[dict]:
        with get_connection(cursor_factory=dict_cursor()) as (_, cursor):
            cursor.execute(
                """
                SELECT id::text, title, created_at, updated_at
                FROM chat_conversations
                WHERE user_id = %s AND project_id = %s
                ORDER BY updated_at DESC
                LIMIT %s
                """,
                (user_id, project_id, limit),
            )
            rows = cursor.fetchall()

        return [
            {
                "id": row["id"],
                "title": row["title"],
                "createdAt": row["created_at"].isoformat() if row["created_at"] else None,
                "updatedAt": row["updated_at"].isoformat() if row["updated_at"] else None,
            }
            for row in rows
        ]


def _normalize_uuid(value: str) -> str:
    try:
        return str(uuid.UUID(str(value).strip()))
    except (ValueError, AttributeError, TypeError) as exc:
        raise ConversationNotFoundError("Invalid conversationId") from exc


def _build_title(question: str) -> str:
    compact = " ".join((question or "").split())
    return compact[:100] or "New conversation"
