from pathlib import Path

import fitz
from pypdf import PdfReader

from app.services.chunk_service import clean_text_for_storage


class PdfExtractionError(Exception):
    pass


def extract_pdf_pages(file_path: str) -> list[dict]:
    path = Path(file_path)
    if not path.exists():
        raise PdfExtractionError(f"File not found: {file_path}")

    pages = _extract_with_pymupdf(path)
    if pages:
        return pages

    pages = _extract_with_pypdf(path)
    if pages:
        return pages

    raise PdfExtractionError(
        "No selectable text was found by the text extractors. OCR fallback is disabled."
    )


def _normalize_extracted_text(text: str) -> str:
    return clean_text_for_storage(text or "")


def _extract_with_pymupdf(path: Path) -> list[dict]:
    pages: list[dict] = []

    try:
        document = fitz.open(str(path))
    except Exception:
        return pages

    try:
        for index, page in enumerate(document, start=1):
            text = _get_page_text_with_blocks(page)
            text = _normalize_extracted_text(text)
            if text:
                pages.append({"page_number": index, "text": text})
    finally:
        document.close()

    return pages


def _get_page_text_with_blocks(page: fitz.Page) -> str:
    try:
        blocks = page.get_text("blocks", sort=True) or []
    except TypeError:
        blocks = page.get_text("blocks") or []

    text_blocks: list[str] = []
    for block in blocks:
        if len(block) < 5:
            continue
        block_text = str(block[4] or "").strip()
        if block_text:
            text_blocks.append(block_text)

    if text_blocks:
        # Keep visual blocks as semantic paragraph boundaries for chunking.
        return "\n\n".join(text_blocks)

    return page.get_text("text") or ""


def _extract_with_pypdf(path: Path) -> list[dict]:
    pages: list[dict] = []

    try:
        reader = PdfReader(str(path))
    except Exception as exc:
        raise PdfExtractionError("Could not read PDF file") from exc

    for index, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text = _normalize_extracted_text(text)
        if text:
            pages.append({"page_number": index, "text": text})

    return pages
