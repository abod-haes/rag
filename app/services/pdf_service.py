from pathlib import Path
from pypdf import PdfReader


class PdfExtractionError(Exception):
    pass


def extract_pdf_pages(file_path: str) -> list[dict]:
    path = Path(file_path)
    if not path.exists():
        raise PdfExtractionError(f"File not found: {file_path}")

    try:
        reader = PdfReader(str(path))
    except Exception as exc:
        raise PdfExtractionError("Could not read PDF file") from exc

    pages: list[dict] = []
    for index, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text = " ".join(text.split())
        if text:
            pages.append({"page_number": index, "text": text})

    if not pages:
        raise PdfExtractionError(
            "No selectable text was found in this PDF. OCR support will be added later."
        )

    return pages
