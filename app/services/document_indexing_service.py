import time
import traceback
import uuid
from dataclasses import dataclass
from decimal import Decimal

from app.core.config import get_settings
from app.db.database import get_connection
from app.services.chunk_service import build_chunks_from_pages
from app.services.embedding_service import EmbeddingService, to_pgvector
from app.services.ocr_service import OcrExtractionError, OpenAIOcrService
from app.services.pdf_service import PdfExtractionError, extract_pdf_pages
from app.services.usage_service import (
    TokenUsage,
    estimate_chat_cost_usd,
    estimate_embedding_cost_usd,
)


@dataclass(frozen=True)
class DocumentIndexingResult:
    extraction_method: str
    indexed_chunks: int
    total_extracted_chunks: int
    provider: str
    embedding_model: str
    embedding_usage: TokenUsage
    ocr_model: str | None
    ocr_usage: TokenUsage
    estimated_cost_usd: Decimal


class DocumentIndexingService:
    def __init__(self) -> None:
        self.settings = get_settings()

    def index_document(
        self,
        *,
        document_id: str,
        user_id: str,
        project_id: str,
        file_path: str,
    ) -> DocumentIndexingResult:
        provider = self.settings.ai_provider.lower().strip()
        embedding_model = (
            self.settings.openai_embedding_model
            if provider == "openai"
            else self.settings.gemini_embedding_model
        )
        embedding_usage = TokenUsage()
        ocr_usage = TokenUsage()
        ocr_model: str | None = None
        ocr_service: OpenAIOcrService | None = None
        extraction_method = "text"

        self._set_progress(
            document_id=document_id,
            status="processing",
            stage="extracting",
            indexed_chunks=0,
            total_chunks=0,
            error_message=None,
        )

        try:
            try:
                pages = extract_pdf_pages(file_path)
            except PdfExtractionError:
                if not self.settings.enable_ocr_fallback:
                    raise
                extraction_method = "ocr"
                ocr_model = self.settings.openai_ocr_model
                ocr_service = OpenAIOcrService()
                pages = ocr_service.extract_pdf_pages(file_path)
                ocr_usage = ocr_service.last_usage

            chunks = build_chunks_from_pages(pages)
            total_extracted_chunks = len(chunks)
            if self.settings.max_chunks_per_document > 0:
                chunks = chunks[: self.settings.max_chunks_per_document]

            if not chunks:
                raise PdfExtractionError("No indexable text chunks were produced")

            self._prepare_chunk_storage(
                document_id=document_id,
                total_chunks=len(chunks),
            )

            embedding_service = EmbeddingService()
            batch_size = max(1, self.settings.embedding_batch_size)

            for batch_start in range(0, len(chunks), batch_size):
                batch = chunks[batch_start : batch_start + batch_size]
                batch_result = embedding_service.embed_documents_batch_with_usage(
                    [chunk["content"] for chunk in batch]
                )
                embedding_usage = embedding_usage + batch_result.usage

                with get_connection() as (_, cursor):
                    for chunk, embedding in zip(
                        batch,
                        batch_result.values,
                        strict=True,
                    ):
                        cursor.execute(
                            """
                            INSERT INTO document_chunks (
                                id,
                                document_id,
                                user_id,
                                project_id,
                                content,
                                page_number,
                                chunk_index,
                                section_title,
                                content_type,
                                embedding
                            )
                            VALUES (
                                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::vector
                            )
                            """,
                            (
                                str(uuid.uuid4()),
                                document_id,
                                user_id,
                                project_id,
                                chunk["content"],
                                chunk["page_number"],
                                chunk["chunk_index"],
                                chunk.get("section_title"),
                                chunk.get("content_type") or "text",
                                to_pgvector(embedding),
                            ),
                        )

                    indexed_chunks = min(batch_start + len(batch), len(chunks))
                    cursor.execute(
                        """
                        UPDATE documents
                        SET indexed_chunks = %s,
                            indexing_stage = %s,
                            updated_at = NOW()
                        WHERE id = %s
                        """,
                        (indexed_chunks, "embedding", document_id),
                    )

                if (
                    batch_start + batch_size < len(chunks)
                    and self.settings.embedding_request_delay_seconds > 0
                ):
                    time.sleep(self.settings.embedding_request_delay_seconds)

            estimated_cost_usd = self._calculate_cost(
                provider=provider,
                embedding_usage=embedding_usage,
                ocr_usage=ocr_usage,
            )
            with get_connection() as (_, cursor):
                cursor.execute(
                    """
                    UPDATE documents
                    SET status = %s,
                        indexing_stage = %s,
                        indexed_chunks = %s,
                        total_chunks = %s,
                        error_message = NULL,
                        ai_provider = %s,
                        embedding_model = %s,
                        embedding_tokens = %s,
                        ocr_model = %s,
                        ocr_input_tokens = %s,
                        ocr_cached_input_tokens = %s,
                        ocr_output_tokens = %s,
                        estimated_cost_usd = %s,
                        updated_at = NOW()
                    WHERE id = %s
                    """,
                    (
                        "ready",
                        "ready",
                        len(chunks),
                        len(chunks),
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

            return DocumentIndexingResult(
                extraction_method=extraction_method,
                indexed_chunks=len(chunks),
                total_extracted_chunks=total_extracted_chunks,
                provider=provider,
                embedding_model=embedding_model,
                embedding_usage=embedding_usage,
                ocr_model=ocr_model,
                ocr_usage=ocr_usage,
                estimated_cost_usd=estimated_cost_usd,
            )
        except (PdfExtractionError, OcrExtractionError, Exception) as exc:
            traceback.print_exc()
            if ocr_service is not None:
                ocr_usage = ocr_service.last_usage
            estimated_cost_usd = self._calculate_cost(
                provider=provider,
                embedding_usage=embedding_usage,
                ocr_usage=ocr_usage,
            )
            self._mark_failed(
                document_id=document_id,
                error_message=f"{exc.__class__.__name__}: {exc}",
                provider=provider,
                embedding_model=embedding_model,
                embedding_usage=embedding_usage,
                ocr_model=ocr_model,
                ocr_usage=ocr_usage,
                estimated_cost_usd=estimated_cost_usd,
            )
            raise

    def _prepare_chunk_storage(self, *, document_id: str, total_chunks: int) -> None:
        with get_connection() as (_, cursor):
            cursor.execute(
                "DELETE FROM document_chunks WHERE document_id = %s",
                (document_id,),
            )
            cursor.execute(
                """
                UPDATE documents
                SET indexing_stage = %s,
                    indexed_chunks = 0,
                    total_chunks = %s,
                    updated_at = NOW()
                WHERE id = %s
                """,
                ("embedding", total_chunks, document_id),
            )

    def _set_progress(
        self,
        *,
        document_id: str,
        status: str,
        stage: str,
        indexed_chunks: int,
        total_chunks: int,
        error_message: str | None,
    ) -> None:
        with get_connection() as (_, cursor):
            cursor.execute(
                """
                UPDATE documents
                SET status = %s,
                    indexing_stage = %s,
                    indexed_chunks = %s,
                    total_chunks = %s,
                    error_message = %s,
                    updated_at = NOW()
                WHERE id = %s
                """,
                (
                    status,
                    stage,
                    indexed_chunks,
                    total_chunks,
                    error_message,
                    document_id,
                ),
            )

    def _mark_failed(
        self,
        *,
        document_id: str,
        error_message: str,
        provider: str,
        embedding_model: str,
        embedding_usage: TokenUsage,
        ocr_model: str | None,
        ocr_usage: TokenUsage,
        estimated_cost_usd: Decimal,
    ) -> None:
        with get_connection() as (_, cursor):
            cursor.execute(
                """
                UPDATE documents
                SET status = %s,
                    indexing_stage = %s,
                    error_message = %s,
                    ai_provider = %s,
                    embedding_model = %s,
                    embedding_tokens = %s,
                    ocr_model = %s,
                    ocr_input_tokens = %s,
                    ocr_cached_input_tokens = %s,
                    ocr_output_tokens = %s,
                    estimated_cost_usd = %s,
                    updated_at = NOW()
                WHERE id = %s
                """,
                (
                    "failed",
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

    def _calculate_cost(
        self,
        *,
        provider: str,
        embedding_usage: TokenUsage,
        ocr_usage: TokenUsage,
    ) -> Decimal:
        return estimate_embedding_cost_usd(
            embedding_usage.input_tokens,
            provider=provider,
            settings=self.settings,
        ) + estimate_chat_cost_usd(
            ocr_usage,
            provider=provider,
            settings=self.settings,
            purpose="ocr",
        )
