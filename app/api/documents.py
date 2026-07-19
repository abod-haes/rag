import shutil
import time
import traceback
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status

from app.core.config import get_settings
from app.core.security import verify_api_key
from app.db.database import dict_cursor, get_connection
from app.services.chunk_service import build_chunks_from_pages
from app.services.embedding_service import EmbeddingService, to_pgvector
from app.services.ocr_service import OcrExtractionError, OpenAIOcrService
from app.services.pdf_service import PdfExtractionError, extract_pdf_pages

DEFAULT_USER_ID = "default-user"
DEFAULT_PROJECT_ID = "default-project"

router = APIRouter(
    prefix="/api/documents",
    tags=["Documents"],
    dependencies=[Depends(verify_api_key)],
)


@router.post("/upload", status_code=status.HTTP_201_CREATED)
def upload_document(
    file: UploadFile = File(...),
    name: str | None = Form(default=None),
):
    user_id = DEFAULT_USER_ID
    project_id = DEFAULT_PROJECT_ID
    extraction_method = "text"

    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    settings = get_settings()
    upload_dir = Path(settings.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)

    document_id = str(uuid.uuid4())
    safe_name = Path(file.filename).name
    document_name = (name or "").strip() or safe_name
    if len(document_name) > 255:
        raise HTTPException(
            status_code=400,
            detail="Document name must be 255 characters or fewer",
        )
    stored_name = f"{document_id}_{safe_name}"
    file_path = upload_dir / stored_name

    with file_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    with get_connection() as (_, cursor):
        cursor.execute(
            """
            INSERT INTO documents
            (id, user_id, project_id, name, file_name, file_path, status)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            """,
            (
                document_id,
                user_id,
                project_id,
                document_name,
                safe_name,
                str(file_path),
                "processing",
            ),
        )

    try:
        try:
            pages = extract_pdf_pages(str(file_path))
        except PdfExtractionError as exc:
            if not settings.enable_ocr_fallback:
                raise exc

            extraction_method = "ocr"
            pages = OpenAIOcrService().extract_pdf_pages(str(file_path))

        chunks = build_chunks_from_pages(pages)
        original_chunk_count = len(chunks)

        if settings.max_chunks_per_document > 0:
            chunks = chunks[: settings.max_chunks_per_document]

        embedding_service = EmbeddingService()

        with get_connection() as (_, cursor):
            for index, chunk in enumerate(chunks, start=1):
                if index > 1 and settings.embedding_request_delay_seconds > 0:
                    time.sleep(settings.embedding_request_delay_seconds)

                chunk_id = str(uuid.uuid4())
                embedding = embedding_service.embed_document(chunk["content"])
                cursor.execute(
                    """
                    INSERT INTO document_chunks
                    (id, document_id, user_id, project_id, content, page_number, chunk_index, embedding)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s::vector)
                    """,
                    (
                        chunk_id,
                        document_id,
                        user_id,
                        project_id,
                        chunk["content"],
                        chunk["page_number"],
                        chunk["chunk_index"],
                        to_pgvector(embedding),
                    ),
                )

            cursor.execute(
                "UPDATE documents SET status = %s, error_message = NULL WHERE id = %s",
                ("ready", document_id),
            )

    except PdfExtractionError as exc:
        _mark_failed(document_id, str(exc))
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except OcrExtractionError as exc:
        _mark_failed(document_id, str(exc))
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        traceback.print_exc()
        error_message = f"{exc.__class__.__name__}: {str(exc)}"
        _mark_failed(document_id, error_message)
        raise HTTPException(status_code=500, detail="Document indexing failed. Check docker logs.") from exc

    return {
        "documentId": document_id,
        "name": document_name,
        "fileName": safe_name,
        "status": "ready",
        "extractionMethod": extraction_method,
        "indexedChunks": len(chunks),
        "totalExtractedChunks": original_chunk_count,
        "isPartialIndex": len(chunks) < original_chunk_count,
    }


@router.get("")
def list_documents():
    with get_connection(cursor_factory=dict_cursor()) as (_, cursor):
        cursor.execute(
            """
            SELECT
                id::text,
                COALESCE(NULLIF(BTRIM(name), ''), file_name) AS name,
                file_name,
                status,
                error_message,
                created_at
            FROM documents
            WHERE user_id = %s AND project_id = %s
            ORDER BY created_at DESC
            """,
            (DEFAULT_USER_ID, DEFAULT_PROJECT_ID),
        )
        rows = cursor.fetchall()

    return [
        {
            "id": row["id"],
            "name": row["name"],
            "fileName": row["file_name"],
            "status": row["status"],
            "errorMessage": row["error_message"],
            "createdAt": row["created_at"].isoformat() if row["created_at"] else None,
        }
        for row in rows
    ]


@router.delete("/{document_id}")
def delete_document(document_id: str):
    with get_connection(cursor_factory=dict_cursor()) as (_, cursor):
        cursor.execute(
            """
            SELECT file_path
            FROM documents
            WHERE id = %s AND user_id = %s AND project_id = %s
            """,
            (document_id, DEFAULT_USER_ID, DEFAULT_PROJECT_ID),
        )
        row = cursor.fetchone()

        if not row:
            raise HTTPException(status_code=404, detail="Document not found")

        cursor.execute(
            "DELETE FROM documents WHERE id = %s AND user_id = %s AND project_id = %s",
            (document_id, DEFAULT_USER_ID, DEFAULT_PROJECT_ID),
        )

    file_path = Path(row["file_path"])
    if file_path.exists():
        file_path.unlink()

    return {"success": True}


def _mark_failed(document_id: str, error_message: str) -> None:
    with get_connection() as (_, cursor):
        cursor.execute(
            "UPDATE documents SET status = %s, error_message = %s WHERE id = %s",
            ("failed", error_message[:1000], document_id),
        )
