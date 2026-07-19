import base64
from io import BytesIO

import fitz
from openai import OpenAI

from app.core.config import get_settings
from app.services.chunk_service import clean_text_for_storage
from app.services.usage_service import TokenUsage, extract_openai_usage


class OcrExtractionError(Exception):
    pass


class OpenAIOcrService:
    def __init__(self) -> None:
        self.settings = get_settings()
        if not self.settings.openai_api_key:
            raise OcrExtractionError("OPENAI_API_KEY is required for OCR fallback")
        self.client = OpenAI(api_key=self.settings.openai_api_key)
        self.last_usage = TokenUsage()

    def extract_pdf_pages(self, file_path: str) -> list[dict]:
        pages: list[dict] = []
        self.last_usage = TokenUsage()

        try:
            document = fitz.open(file_path)
        except Exception as exc:
            raise OcrExtractionError(f"Unable to open PDF for OCR: {exc}") from exc

        try:
            pages_to_process = min(len(document), self.settings.max_ocr_pages)
            for page_index in range(pages_to_process):
                page = document.load_page(page_index)
                image_data_url = self._render_page_to_data_url(page)
                text = self._extract_text_from_image(image_data_url, page_index + 1)
                text = clean_text_for_storage(text)

                if text:
                    pages.append(
                        {
                            "page_number": page_index + 1,
                            "text": text,
                        }
                    )
        finally:
            document.close()

        if not pages:
            raise OcrExtractionError("OCR did not extract readable text from this PDF")

        return pages

    def _render_page_to_data_url(self, page: fitz.Page) -> str:
        matrix = fitz.Matrix(self.settings.ocr_render_zoom, self.settings.ocr_render_zoom)
        pixmap = page.get_pixmap(matrix=matrix, alpha=False)
        png_bytes = pixmap.tobytes("png")
        encoded = base64.b64encode(png_bytes).decode("utf-8")
        return f"data:image/png;base64,{encoded}"

    def _extract_text_from_image(self, image_data_url: str, page_number: int) -> str:
        prompt = (
            "Extract all readable text from this PDF page image. "
            "Keep Arabic text in Arabic. Keep English text in English. "
            "Preserve headings, bullet points, numbers, equations, and code-like text as much as possible. "
            "Return only the extracted text without explanations. "
            f"This is page {page_number}."
        )

        response = self.client.responses.create(
            model=self.settings.openai_ocr_model,
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": prompt},
                        {"type": "input_image", "image_url": image_data_url},
                    ],
                }
            ],
        )

        self.last_usage = self.last_usage + extract_openai_usage(
            getattr(response, "usage", None)
        )

        text = getattr(response, "output_text", None)
        return text.strip() if text else ""
