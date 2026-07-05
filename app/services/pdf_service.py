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
        "No selectable text was found by the text extractors. If you can select text manually, send the PDF sample so we can tune extraction."
    )


def _normalize_extracted_text(text: str) -> str:
    text = clean_text_for_storage(text or "")
    return " ".join(text.split())


def _extract_with_pymupdf(path: Path) -> list[dict]:
    pages: list[dict] = []

    try:
        document = fitz.open(str(path))
    except Exception:
        return pages

    try:
        for index, page in enumerate(document, start=1):
            text = page.get_text("text") or ""
            text = _normalize_extracted_text(text)
            if text:
                pages.append({"page_number": index, "text": text})
    finally:
        document.close()

    return pages


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
