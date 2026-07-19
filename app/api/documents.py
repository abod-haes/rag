import shutil
import time
import traceback
import uuid
from decimal import Decimal
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status

from app.core.config import get_settings
from app.core.security import verify_api_key
from app.db.database import dict_cursor, get_connection
from app.services.chunk_service import build_chunks_from_pages
from app.services.embedding_service import EmbeddingService, to_pgvector
from app.services.ocr_service import OcrExtractionError, OpenAIOcrService
from app.services.pdf_service import PdfExtractionError, extract_pdf_pages
from app.services.usage_service import (
    TokenUsage,
    decimal_to_json,
    estimate_chat_cost_usd,
    estimate_embedding_cost_usd,
)

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
    provider = settings.ai_provider.lower().strip()
    embedding_model = (
        settings.openai_embedding_model
        if provider == "openai"
        else settings.gemini_embedding_model
    )
    embedding_usage = TokenUsage()
    ocr_usage = TokenUsage()
    ocr_model: str | None = None
    ocr_service: OpenAIOcrService | None = None
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
            ocr_model = settings.openai_ocr_model
            ocr_service = OpenAIOcrService()
            pages = ocr_service.extract_pdf_pages(str(file_path))
            ocr_usage = ocr_service.last_usage

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
                embedding_result = embedding_service.embed_document_with_usage(
                    chunk["content"]
                )
                embedding_usage = embedding_usage + embedding_result.usage
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
                        to_pgvector(embedding_result.values),
                    ),
                )

            estimated_cost_usd = _calculate_indexing_cost(
                provider=provider,
                embedding_usage=embedding_usage,
                ocr_usage=ocr_usage,
            )
            cursor.execute(
                """
                UPDATE documents
                SET status = %s,
                    error_message = NULL,
                    ai_provider = %s,
                    embedding_model = %s,
                    embedding_tokens = %s,
                    ocr_model = %s,
                    ocr_input_tokens = %s,
                    ocr_cached_input_tokens = %s,
                    ocr_output_tokens = %s,
                    estimated_cost_usd = %s
                WHERE id = %s
                """,
                (
                    "ready",
                    provider,
                    embedding_model,
                    embedding_usage.input_tokens,
                    ocr_model,
                    ocr_usage.input_tokens,
                    ocr_usage.cached_input_tokens,
                    ocr_usage.output_tokens,
                    estimated_cost_usd,
                    document_id,
                ),
            )

    except PdfExtractionError as exc:
        _mark_failed(
            document_id,
            str(exc),
            provider=provider,
            embedding_model=embedding_model,
            embedding_usage=embedding_usage,
            ocr_model=ocr_model,
            ocr_usage=ocr_usage,
        )
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except OcrExtractionError as exc:
        if ocr_service is not None:
            ocr_usage = ocr_service.last_usage
        _mark_failed(
            document_id,
            str(exc),
            provider=provider,
            embedding_model=embedding_model,
            embedding_usage=embedding_usage,
            ocr_model=ocr_model,
            ocr_usage=ocr_usage,
        )
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        traceback.print_exc()
        if ocr_service is not None:
            ocr_usage = ocr_service.last_usage
        error_message = f"{exc.__class__.__name__}: {str(exc)}"
        _mark_failed(
            document_id,
            error_message,
            provider=provider,
            embedding_model=embedding_model,
            embedding_usage=embedding_usage,
            ocr_model=ocr_model,
            ocr_usage=ocr_usage,
        )
        raise HTTPException(
            status_code=500,
            detail="Document indexing failed. Check docker logs.",
        ) from exc

    usage = _build_document_usage(
        provider=provider,
        embedding_model=embedding_model,
        embedding_tokens=embedding_usage.input_tokens,
        ocr_model=ocr_model,
        ocr_input_tokens=ocr_usage.input_tokens,
        ocr_cached_input_tokens=ocr_usage.cached_input_tokens,
        ocr_output_tokens=ocr_usage.output_tokens,
        estimated_cost_usd=estimated_cost_usd,
    )
    return {
        "documentId": document_id,
        "name": document_name,
        "fileName": safe_name,
        "status": "ready",
        "extractionMethod": extraction_method,
        "indexedChunks": len(chunks),
        "totalExtractedChunks": original_chunk_count,
        "isPartialIndex": len(chunks) < original_chunk_count,
        "usage": usage,
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
                ai_provider,
                embedding_model,
                embedding_tokens,
                ocr_model,
                ocr_input_tokens,
                ocr_cached_input_tokens,
                ocr_output_tokens,
                estimated_cost_usd,
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
            "usage": _build_document_usage(
                provider=row["ai_provider"],
                embedding_model=row["embedding_model"],
                embedding_tokens=row["embedding_tokens"],
                ocr_model=row["ocr_model"],
                ocr_input_tokens=row["ocr_input_tokens"],
                ocr_cached_input_tokens=row["ocr_cached_input_tokens"],
                ocr_output_tokens=row["ocr_output_tokens"],
                estimated_cost_usd=row["estimated_cost_usd"],
            ),
            "createdAt": row["created_at"].isoformat() if row["created_at"] else None,
        }
        for row in rows
    ]


@router.get("/{document_id}/usage")
def get_document_usage(document_id: str):
    with get_connection(cursor_factory=dict_cursor()) as (_, cursor):
        cursor.execute(
            """
            SELECT
                id::text,
                COALESCE(NULLIF(BTRIM(name), ''), file_name) AS name,
                file_name,
                ai_provider,
                embedding_model,
                embedding_tokens,
                ocr_model,
                ocr_input_tokens,
                ocr_cached_input_tokens,
                ocr_output_tokens,
                estimated_cost_usd
            FROM documents
            WHERE id = %s AND user_id = %s AND project_id = %s
            """,
            (document_id, DEFAULT_USER_ID, DEFAULT_PROJECT_ID),
        )
        row = cursor.fetchone()

    if not row:
        raise HTTPException(status_code=404, detail="Document not found")

    return {
        "id": row["id"],
        "name": row["name"],
        "fileName": row["file_name"],
        "usage": _build_document_usage(
            provider=row["ai_provider"],
            embedding_model=row["embedding_model"],
            embedding_tokens=row["embedding_tokens"],
            ocr_model=row["ocr_model"],
            ocr_input_tokens=row["ocr_input_tokens"],
            ocr_cached_input_tokens=row["ocr_cached_input_tokens"],
            ocr_output_tokens=row["ocr_output_tokens"],
            estimated_cost_usd=row["estimated_cost_usd"],
        ),
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


def _mark_failed(
    document_id: str,
    error_message: str,
    *,
    provider: str,
    embedding_model: str,
    embedding_usage: TokenUsage,
    ocr_model: str | None,
    ocr_usage: TokenUsage,
) -> None:
    estimated_cost_usd = _calculate_indexing_cost(
        provider=provider,
        embedding_usage=embedding_usage,
        ocr_usage=ocr_usage,
    )
    with get_connection() as (_, cursor):
        cursor.execute(
            """
            UPDATE documents
            SET status = %s,
                error_message = %s,
                ai_provider = %s,
                embedding_model = %s,
                embedding_tokens = %s,
                ocr_model = %s,
                ocr_input_tokens = %s,
                ocr_cached_input_tokens = %s,
                ocr_output_tokens = %s,
                estimated_cost_usd = %s
            WHERE id = %s
            """,
            (
                "failed",
                error_message[:1000],
                provider,
                embedding_model,
                embedding_usage.input_tokens,
                ocr_model,
                ocr_usage.input_tokens,
                ocr_usage.cached_input_tokens,
                ocr_usage.output_tokens,
                estimated_cost_usd,
                document_id,
            ),
        )


def _calculate_indexing_cost(
    *,
    provider: str,
    embedding_usage: TokenUsage,
    ocr_usage: TokenUsage,
) -> Decimal:
    settings = get_settings()
    embedding_cost = estimate_embedding_cost_usd(
        embedding_usage.input_tokens,
        provider=provider,
        settings=settings,
    )
    ocr_cost = estimate_chat_cost_usd(
        ocr_usage,
        provider=provider,
        settings=settings,
        purpose="ocr",
    )
    return embedding_cost + ocr_cost


def _build_document_usage(
    *,
    provider,
    embedding_model,
    embedding_tokens,
    ocr_model,
    ocr_input_tokens,
    ocr_cached_input_tokens,
    ocr_output_tokens,
    estimated_cost_usd,
) -> dict:
    embedding_tokens = int(embedding_tokens or 0)
    ocr_input_tokens = int(ocr_input_tokens or 0)
    ocr_cached_input_tokens = int(ocr_cached_input_tokens or 0)
    ocr_output_tokens = int(ocr_output_tokens or 0)
    cost = Decimal(str(estimated_cost_usd or 0))

    return {
        "provider": provider,
        "embeddingModel": embedding_model,
        "embeddingTokens": embedding_tokens,
        "ocrModel": ocr_model,
        "ocrInputTokens": ocr_input_tokens,
        "ocrCachedInputTokens": ocr_cached_input_tokens,
        "ocrOutputTokens": ocr_output_tokens,
        "totalTokens": embedding_tokens + ocr_input_tokens + ocr_output_tokens,
        "estimatedCostUsd": decimal_to_json(cost),
        "costTrackingAvailable": provider == "openai",
    }
