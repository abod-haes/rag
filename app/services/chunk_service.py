from app.core.config import get_settings


def clean_text_for_storage(text: str) -> str:
    return (
        text.replace("\x00", "")
        .replace("\ufffd", " ")
        .replace("\u0000", "")
        .strip()
    )


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
