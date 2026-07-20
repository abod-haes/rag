import hashlib
import shutil
import uuid
from pathlib import Path

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    Form,
    HTTPException,
    UploadFile,
    status,
)

from app.api.documents import (
    DEFAULT_PROJECT_ID,
    DEFAULT_USER_ID,
    _build_document_usage,
)
from app.core.config import get_settings
from app.core.security import verify_api_key
from app.db.database import dict_cursor, get_connection
from app.services.document_indexing_service import (
    DocumentIndexingResult,
    DocumentIndexingService,
)
from app.services.ocr_service import OcrExtractionError
from app.services.pdf_service import PdfExtractionError

router = APIRouter(
    prefix="/api/documents",
    tags=["Documents"],
    dependencies=[Depends(verify_api_key)],
)

READ_SIZE = 1024 * 1024


@router.post("/upload", status_code=status.HTTP_201_CREATED)
def upload_document(
    file: UploadFile = File(...),
    name: str | None = Form(default=None),
):
    document = _create_document_record(file=file, name=name, initial_status="processing")
    try:
        result = DocumentIndexingService().index_document(
            document_id=document["documentId"],
            user_id=DEFAULT_USER_ID,
            project_id=DEFAULT_PROJECT_ID,
            file_path=document["filePath"],
        )
    except (PdfExtractionError, OcrExtractionError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Document indexing failed. Check docker logs.",
        ) from exc

    return _build_upload_response(document=document, result=result)


@router.post("/upload-async", status_code=status.HTTP_202_ACCEPTED)
def upload_document_async(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    name: str | None = Form(default=None),
):
    document = _create_document_record(file=file, name=name, initial_status="queued")
    background_tasks.add_task(
        _run_background_indexing,
        document["documentId"],
        document["filePath"],
    )
    return {
        "documentId": document["documentId"],
        "name": document["name"],
        "fileName": document["fileName"],
        "status": "queued",
        "statusUrl": f"/api/documents/{document['documentId']}/status",
    }


@router.post("/{document_id}/retry", status_code=status.HTTP_202_ACCEPTED)
def retry_document(document_id: str, background_tasks: BackgroundTasks):
    with get_connection(cursor_factory=dict_cursor()) as (_, cursor):
        cursor.execute(
            """
            SELECT id::text, file_path, status
            FROM documents
            WHERE id = %s AND user_id = %s AND project_id = %s
            """,
            (document_id, DEFAULT_USER_ID, DEFAULT_PROJECT_ID),
        )
        row = cursor.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Document not found")
        if row["status"] in {"queued", "processing"}:
            raise HTTPException(status_code=409, detail="Document is already being indexed")
        if not Path(row["file_path"]).exists():
            raise HTTPException(status_code=409, detail="Stored PDF file no longer exists")

        cursor.execute(
            """
            UPDATE documents
            SET status = %s,
                indexing_stage = %s,
                indexed_chunks = 0,
                total_chunks = 0,
                error_message = NULL,
                updated_at = NOW()
            WHERE id = %s
            """,
            ("queued", "queued", document_id),
        )

    background_tasks.add_task(
        _run_background_indexing,
        document_id,
        row["file_path"],
    )
    return {
        "documentId": document_id,
        "status": "queued",
        "statusUrl": f"/api/documents/{document_id}/status",
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
                indexing_stage,
                indexed_chunks,
                total_chunks,
                error_message,
                ai_provider,
                embedding_model,
                embedding_tokens,
                ocr_model,
                ocr_input_tokens,
                ocr_cached_input_tokens,
                ocr_output_tokens,
                estimated_cost_usd,
                created_at,
                updated_at
            FROM documents
            WHERE user_id = %s AND project_id = %s
            ORDER BY created_at DESC
            """,
            (DEFAULT_USER_ID, DEFAULT_PROJECT_ID),
        )
        rows = cursor.fetchall()

    return [_build_document_row(row) for row in rows]


@router.get("/{document_id}/status")
def get_document_status(document_id: str):
    row = _get_document(document_id)
    return {
        "documentId": row["id"],
        "name": row["name"],
        "fileName": row["file_name"],
        "status": row["status"],
        "stage": row["indexing_stage"],
        "indexedChunks": int(row["indexed_chunks"] or 0),
        "totalChunks": int(row["total_chunks"] or 0),
        "progressPercent": _progress_percent(
            row["indexed_chunks"],
            row["total_chunks"],
            row["status"],
        ),
        "errorMessage": row["error_message"],
        "updatedAt": row["updated_at"].isoformat() if row["updated_at"] else None,
    }


@router.get("/{document_id}/usage")
def get_document_usage(document_id: str):
    row = _get_document(document_id)
    return {
        "id": row["id"],
        "name": row["name"],
        "fileName": row["file_name"],
        "usage": _usage_from_row(row),
    }


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


def _create_document_record(
    *,
    file: UploadFile,
    name: str | None,
    initial_status: str,
) -> dict:
    settings = get_settings()
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    safe_name = Path(file.filename).name
    document_name = (name or "").strip() or safe_name
    if len(document_name) > 255:
        raise HTTPException(
            status_code=400,
            detail="Document name must be 255 characters or fewer",
        )

    document_id = str(uuid.uuid4())
    upload_dir = Path(settings.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)
    file_path = upload_dir / f"{document_id}_{safe_name}"
    file_hash, file_size = _save_and_hash_upload(file=file, file_path=file_path)

    max_bytes = settings.max_upload_size_mb * 1024 * 1024
    if file_size > max_bytes:
        file_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=413,
            detail=f"PDF exceeds the {settings.max_upload_size_mb} MB upload limit",
        )

    if not settings.allow_duplicate_documents:
        duplicate = _find_duplicate_document(file_hash)
        if duplicate:
            file_path.unlink(missing_ok=True)
            raise HTTPException(
                status_code=409,
                detail={
                    "message": "This PDF is already uploaded",
                    "existingDocumentId": duplicate["id"],
                    "existingDocumentName": duplicate["name"],
                },
            )

    with get_connection() as (_, cursor):
        cursor.execute(
            """
            INSERT INTO documents (
                id,
                user_id,
                project_id,
                name,
                file_name,
                file_path,
                file_hash,
                status,
                indexing_stage,
                updated_at
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
            """,
            (
                document_id,
                DEFAULT_USER_ID,
                DEFAULT_PROJECT_ID,
                document_name,
                safe_name,
                str(file_path),
                file_hash,
                initial_status,
                initial_status,
            ),
        )

    return {
        "documentId": document_id,
        "name": document_name,
        "fileName": safe_name,
        "filePath": str(file_path),
    }


def _save_and_hash_upload(*, file: UploadFile, file_path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    total_size = 0
    header = b""

    try:
        with file_path.open("wb") as output:
            while True:
                chunk = file.file.read(READ_SIZE)
                if not chunk:
                    break
                if len(header) < 5:
                    header += chunk[: 5 - len(header)]
                digest.update(chunk)
                output.write(chunk)
                total_size += len(chunk)
    except Exception:
        file_path.unlink(missing_ok=True)
        raise

    if not header.startswith(b"%PDF-"):
        file_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid PDF")

    return digest.hexdigest(), total_size


def _find_duplicate_document(file_hash: str) -> dict | None:
    with get_connection(cursor_factory=dict_cursor()) as (_, cursor):
        cursor.execute(
            """
            SELECT
                id::text,
                COALESCE(NULLIF(BTRIM(name), ''), file_name) AS name
            FROM documents
            WHERE user_id = %s
              AND project_id = %s
              AND file_hash = %s
              AND status IN ('queued', 'processing', 'ready')
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (DEFAULT_USER_ID, DEFAULT_PROJECT_ID, file_hash),
        )
        return cursor.fetchone()


def _run_background_indexing(document_id: str, file_path: str) -> None:
    try:
        DocumentIndexingService().index_document(
            document_id=document_id,
            user_id=DEFAULT_USER_ID,
            project_id=DEFAULT_PROJECT_ID,
            file_path=file_path,
        )
    except Exception:
        # The indexing service already records the failure and traceback.
        return


def _get_document(document_id: str) -> dict:
    with get_connection(cursor_factory=dict_cursor()) as (_, cursor):
        cursor.execute(
            """
            SELECT
                id::text,
                COALESCE(NULLIF(BTRIM(name), ''), file_name) AS name,
                file_name,
                status,
                indexing_stage,
                indexed_chunks,
                total_chunks,
                error_message,
                ai_provider,
                embedding_model,
                embedding_tokens,
                ocr_model,
                ocr_input_tokens,
                ocr_cached_input_tokens,
                ocr_output_tokens,
                estimated_cost_usd,
                created_at,
                updated_at
            FROM documents
            WHERE id = %s AND user_id = %s AND project_id = %s
            """,
            (document_id, DEFAULT_USER_ID, DEFAULT_PROJECT_ID),
        )
        row = cursor.fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Document not found")
    return row


def _build_upload_response(
    *,
    document: dict,
    result: DocumentIndexingResult,
) -> dict:
    return {
        "documentId": document["documentId"],
        "name": document["name"],
        "fileName": document["fileName"],
        "status": "ready",
        "extractionMethod": result.extraction_method,
        "indexedChunks": result.indexed_chunks,
        "totalExtractedChunks": result.total_extracted_chunks,
        "isPartialIndex": result.indexed_chunks < result.total_extracted_chunks,
        "usage": _build_document_usage(
            provider=result.provider,
            embedding_model=result.embedding_model,
            embedding_tokens=result.embedding_usage.input_tokens,
            ocr_model=result.ocr_model,
            ocr_input_tokens=result.ocr_usage.input_tokens,
            ocr_cached_input_tokens=result.ocr_usage.cached_input_tokens,
            ocr_output_tokens=result.ocr_usage.output_tokens,
            estimated_cost_usd=result.estimated_cost_usd,
        ),
    }


def _build_document_row(row: dict) -> dict:
    return {
        "id": row["id"],
        "name": row["name"],
        "fileName": row["file_name"],
        "status": row["status"],
        "stage": row["indexing_stage"],
        "indexedChunks": int(row["indexed_chunks"] or 0),
        "totalChunks": int(row["total_chunks"] or 0),
        "progressPercent": _progress_percent(
            row["indexed_chunks"],
            row["total_chunks"],
            row["status"],
        ),
        "errorMessage": row["error_message"],
        "usage": _usage_from_row(row),
        "createdAt": row["created_at"].isoformat() if row["created_at"] else None,
        "updatedAt": row["updated_at"].isoformat() if row["updated_at"] else None,
    }


def _usage_from_row(row: dict) -> dict:
    return _build_document_usage(
        provider=row["ai_provider"],
        embedding_model=row["embedding_model"],
        embedding_tokens=row["embedding_tokens"],
        ocr_model=row["ocr_model"],
        ocr_input_tokens=row["ocr_input_tokens"],
        ocr_cached_input_tokens=row["ocr_cached_input_tokens"],
        ocr_output_tokens=row["ocr_output_tokens"],
        estimated_cost_usd=row["estimated_cost_usd"],
    )


def _progress_percent(indexed_chunks, total_chunks, status_value: str) -> float:
    if status_value == "ready":
        return 100.0
    total = int(total_chunks or 0)
    if total <= 0:
        return 0.0
    return round(min(100.0, int(indexed_chunks or 0) * 100 / total), 2)
