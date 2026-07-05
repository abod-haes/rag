import re
import unicodedata

from app.core.config import get_settings


WHITESPACE_RE = re.compile(r"\s+")
FONT_MARKER_RE = re.compile(r"/[A-Za-z][A-Za-z0-9_-]*")
HEX_GLYPH_RE = re.compile(r"(?:/?fe[0-9a-fA-F]{2}){3,}")


def clean_text_for_storage(text: str) -> str:
    text = text or ""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace(chr(0), " ").replace("\ufffd", " ")
    text = HEX_GLYPH_RE.sub(" ", text)
    text = FONT_MARKER_RE.sub(" ", text)

    cleaned_chars: list[str] = []
    for char in text:
        category = unicodedata.category(char)
        if category.startswith("C"):
            cleaned_chars.append(" ")
        else:
            cleaned_chars.append(char)

    text = "".join(cleaned_chars)
    text = WHITESPACE_RE.sub(" ", text)
    return text.strip()


def split_page_into_chunks(text: str, page_number: int) -> list[dict]:
    settings = get_settings()
    max_chars = settings.max_chunk_chars
    overlap = settings.chunk_overlap_chars
    text = clean_text_for_storage(text)

    if max_chars <= overlap:
        raise ValueError("MAX_CHUNK_CHARS must be greater than CHUNK_OVERLAP_CHARS")

    chunks: list[dict] = []
    start = 0
    local_index = 0

    while start < len(text):
        end = min(start + max_chars, len(text))
        chunk_text = clean_text_for_storage(text[start:end])

        if chunk_text:
            chunks.append(
                {
                    "page_number": page_number,
                    "content": chunk_text,
                    "local_index": local_index,
                }
            )
            local_index += 1

        if end == len(text):
            break

        start = max(0, end - overlap)

    return chunks


def build_chunks_from_pages(pages: list[dict]) -> list[dict]:
    all_chunks: list[dict] = []
    chunk_index = 0

    for page in pages:
        page_chunks = split_page_into_chunks(page["text"], page["page_number"])
        for chunk in page_chunks:
            chunk["chunk_index"] = chunk_index
            all_chunks.append(chunk)
            chunk_index += 1

    return all_chunks
