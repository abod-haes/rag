import re
import unicodedata

from app.core.config import get_settings


SPACE_RE = re.compile(r"[ \t\f\v]+")
MULTI_NEWLINE_RE = re.compile(r"\n{3,}")
FONT_MARKER_RE = re.compile(r"/[A-Za-z][A-Za-z0-9_-]*")
HEX_GLYPH_RE = re.compile(r"(?:/?fe[0-9a-fA-F]{2}){3,}")
TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?؟؛])\s+")


EXERCISE_MARKERS = ("تمرين", "تدرب", "تدرّب", "سؤال", "مسألة", "حل الأسئلة")
EXAMPLE_MARKERS = ("مثال", "الحل", "نحو الحل", "برهان", "إثبات")
DEFINITION_MARKERS = ("تعريف", "نظرية", "قاعدة", "خاصية", "ملاحظة")


def clean_text_for_storage(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace(chr(0), " ").replace("\ufffd", " ")
    text = HEX_GLYPH_RE.sub(" ", text)
    text = FONT_MARKER_RE.sub(" ", text)

    cleaned_chars: list[str] = []
    for char in text:
        if char == "\n":
            cleaned_chars.append(char)
            continue

        category = unicodedata.category(char)
        cleaned_chars.append(" " if category.startswith("C") else char)

    normalized_lines = [
        SPACE_RE.sub(" ", line).strip()
        for line in "".join(cleaned_chars).split("\n")
    ]
    normalized_text = "\n".join(normalized_lines)
    normalized_text = MULTI_NEWLINE_RE.sub("\n\n", normalized_text)
    return normalized_text.strip()


def estimate_token_count(text: str) -> int:
    return len(TOKEN_RE.findall(text or ""))


def split_page_into_chunks(text: str, page_number: int) -> list[dict]:
    settings = get_settings()
    max_tokens = max(50, settings.max_chunk_tokens)
    overlap_tokens = max(0, min(settings.chunk_overlap_tokens, max_tokens - 1))
    text = clean_text_for_storage(text)

    if not text:
        return []

    units = _build_semantic_units(text, max_tokens)
    chunks: list[dict] = []
    current_units: list[str] = []
    current_tokens = 0
    local_index = 0

    for unit in units:
        unit_tokens = estimate_token_count(unit)
        separator_tokens = 1 if current_units else 0

        if current_units and current_tokens + separator_tokens + unit_tokens > max_tokens:
            chunk_text = clean_text_for_storage("\n\n".join(current_units))
            if chunk_text:
                chunks.append(
                    _build_chunk(
                        text=chunk_text,
                        page_number=page_number,
                        local_index=local_index,
                    )
                )
                local_index += 1

            available_overlap = max(0, max_tokens - unit_tokens - 1)
            current_units = _tail_for_overlap(
                current_units,
                min(overlap_tokens, available_overlap),
            )
            current_tokens = sum(
                estimate_token_count(item) for item in current_units
            )
            separator_tokens = 1 if current_units else 0

        current_units.append(unit)
        current_tokens += separator_tokens + unit_tokens

    if current_units:
        chunk_text = clean_text_for_storage("\n\n".join(current_units))
        if chunk_text:
            chunks.append(
                _build_chunk(
                    text=chunk_text,
                    page_number=page_number,
                    local_index=local_index,
                )
            )

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


def _build_semantic_units(text: str, max_tokens: int) -> list[str]:
    paragraphs = [
        part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()
    ]
    if len(paragraphs) <= 1:
        line_units = [line.strip() for line in text.split("\n") if line.strip()]
        if len(line_units) > 1:
            paragraphs = line_units

    units: list[str] = []
    for paragraph in paragraphs or [text]:
        if estimate_token_count(paragraph) <= max_tokens:
            units.append(paragraph)
            continue

        sentences = [
            part.strip()
            for part in SENTENCE_SPLIT_RE.split(paragraph)
            if part.strip()
        ]
        if len(sentences) <= 1:
            units.extend(_split_by_tokens(paragraph, max_tokens))
            continue

        sentence_group: list[str] = []
        sentence_group_tokens = 0
        for sentence in sentences:
            sentence_tokens = estimate_token_count(sentence)
            if sentence_tokens > max_tokens:
                if sentence_group:
                    units.append(" ".join(sentence_group))
                    sentence_group = []
                    sentence_group_tokens = 0
                units.extend(_split_by_tokens(sentence, max_tokens))
                continue

            if sentence_group and sentence_group_tokens + sentence_tokens > max_tokens:
                units.append(" ".join(sentence_group))
                sentence_group = []
                sentence_group_tokens = 0

            sentence_group.append(sentence)
            sentence_group_tokens += sentence_tokens

        if sentence_group:
            units.append(" ".join(sentence_group))

    return units


def _split_by_tokens(text: str, max_tokens: int) -> list[str]:
    tokens = TOKEN_RE.findall(text)
    return [
        " ".join(tokens[index : index + max_tokens])
        for index in range(0, len(tokens), max_tokens)
    ]


def _tail_for_overlap(units: list[str], overlap_tokens: int) -> list[str]:
    if overlap_tokens <= 0:
        return []

    tail: list[str] = []
    remaining = overlap_tokens
    for unit in reversed(units):
        tokens = TOKEN_RE.findall(unit)
        if not tokens:
            continue

        if len(tokens) <= remaining:
            tail.insert(0, unit)
            remaining -= len(tokens)
        else:
            tail.insert(0, " ".join(tokens[-remaining:]))
            remaining = 0

        if remaining <= 0:
            break

    return tail


def _build_chunk(*, text: str, page_number: int, local_index: int) -> dict:
    return {
        "page_number": page_number,
        "content": text,
        "local_index": local_index,
        "section_title": _guess_section_title(text),
        "content_type": _guess_content_type(text),
    }


def _guess_section_title(text: str) -> str | None:
    for candidate in [part.strip() for part in text.split("\n") if part.strip()]:
        if 3 <= len(candidate) <= 140 and estimate_token_count(candidate) <= 18:
            return candidate
    return None


def _guess_content_type(text: str) -> str:
    lowered = text.casefold()
    if any(marker in lowered for marker in EXERCISE_MARKERS):
        return "exercise"
    if any(marker in lowered for marker in EXAMPLE_MARKERS):
        return "worked_example"
    if any(marker in lowered for marker in DEFINITION_MARKERS):
        return "definition"
    return "text"
