from app.core.config import get_settings
from app.db.database import dict_cursor, get_connection
from app.services.embedding_service import to_pgvector


class RetrievalService:
    def __init__(self) -> None:
        self.settings = get_settings()

    def retrieve(
        self,
        *,
        query_embedding: list[float],
        user_id: str,
        project_id: str,
        document_ids: list[str] | None = None,
        top_k: int | None = None,
    ) -> list[dict]:
        limit = top_k or self.settings.top_k
        query_vector = to_pgvector(query_embedding)

        if document_ids:
            sql = """
                SELECT
                    dc.document_id::text,
                    COALESCE(NULLIF(BTRIM(d.name), ''), d.file_name) AS name,
                    d.file_name,
                    dc.content,
                    dc.page_number,
                    dc.chunk_index,
                    (dc.embedding <=> %s::vector) AS distance
                FROM document_chunks dc
                JOIN documents d ON d.id = dc.document_id
                WHERE dc.user_id = %s
                  AND dc.project_id = %s
                  AND dc.document_id = ANY(%s::uuid[])
                ORDER BY dc.embedding <=> %s::vector
                LIMIT %s;
            """
            params = [query_vector, user_id, project_id, document_ids, query_vector, limit]
        else:
            sql = """
                SELECT
                    dc.document_id::text,
                    COALESCE(NULLIF(BTRIM(d.name), ''), d.file_name) AS name,
                    d.file_name,
                    dc.content,
                    dc.page_number,
                    dc.chunk_index,
                    (dc.embedding <=> %s::vector) AS distance
                FROM document_chunks dc
                JOIN documents d ON d.id = dc.document_id
                WHERE dc.user_id = %s
                  AND dc.project_id = %s
                ORDER BY dc.embedding <=> %s::vector
                LIMIT %s;
            """
            params = [query_vector, user_id, project_id, query_vector, limit]

        with get_connection(cursor_factory=dict_cursor()) as (_, cursor):
            cursor.execute(sql, params)
            rows = cursor.fetchall()

        results: list[dict] = []
        for row in rows:
            distance = float(row["distance"])
            results.append(
                {
                    "document_id": row["document_id"],
                    "name": row["name"],
                    "file_name": row["file_name"],
                    "content": row["content"],
                    "page_number": row["page_number"],
                    "chunk_index": row["chunk_index"],
                    "distance": distance,
                    "score": max(0.0, 1.0 - distance),
                }
            )

        return results
