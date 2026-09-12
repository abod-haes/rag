import re
import unicodedata
from dataclasses import dataclass

from app.core.config import get_settings


SPACE_RE = re.compile(r"[ \t\f\v]+")
MULTI_NEWLINE_RE = re.compile(r"\n{3,}")
FONT_MARKER_RE = re.compile(r"/[A-Za-z][A-Za-z0-9_-]*")
HEX_GLYPH_RE = re.compile(r"(?:/?fe[0-9a-fA-F]{2}){3,}")
TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?؟؛])\s+")
PAGE_NUMBER_RE = re.compile(r"^(?:صفحة\s*)?\d{1,4}$", re.IGNORECASE)


EXERCISE_MARKERS = ("تمرين", "تدرب", "تدرّب", "سؤال", "مسألة", "حل الأسئلة")
EXAMPLE_MARKERS = ("مثال", "الحل", "نحو الحل", "برهان", "إثبات")
DEFINITION_MARKERS = ("تعريف", "نظرية", "قاعدة", "خاصية", "ملاحظة")
CHAPTER_MARKERS = ("الفصل", "الوحدة", "الباب", "chapter", "unit")
LESSON_MARKERS = ("الدرس", "المبحث", "الموضوع", "lesson", "topic")


@dataclass
class HierarchyState:
    chapter_title: str | None = None
    lesson_title: str | None = None
    section_title: str | None = None


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
    configured_max = max(50, settings.max_chunk_tokens)
    max_tokens = max(
        50,
        min(configured_max, max(50, settings.contextual_chunk_max_tokens)),
    )
    configured_overlap = max(0, settings.chunk_overlap_tokens)
    overlap_tokens = max(
        0,
        min(
            configured_overlap,
            max(0, settings.contextual_chunk_overlap_tokens),
            max_tokens - 1,
        ),
    )
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
    settings = get_settings()
    all_chunks: list[dict] = []
    chunk_index = 0
    hierarchy = HierarchyState()

    for page in pages:
        page_number = int(page["page_number"])
        page_text = clean_text_for_storage(page["text"])
        if not page_text:
            continue

        _update_hierarchy_from_page(page_text, hierarchy)
        page_section = _guess_section_title(page_text)
        if page_section:
            hierarchy.section_title = page_section

        page_chunks = split_page_into_chunks(page_text, page_number)
        for chunk in page_chunks:
            raw_content = chunk["content"]
            chunk_section = chunk.get("section_title") or hierarchy.section_title
            hierarchy_path = _build_hierarchy_path(
                hierarchy=hierarchy,
                section_title=chunk_section,
                page_number=page_number,
            )
            content_type = chunk.get("content_type") or "text"
            chunk.update(
                {
                    "chunk_index": chunk_index,
                    "section_title": chunk_section,
                    "content": _contextualize_content(
                        content=raw_content,
                        hierarchy_path=hierarchy_path,
                        page_number=page_number,
                        content_type=content_type,
                    ),
                }
            )
            all_chunks.append(chunk)
            chunk_index += 1

        parent_text = _truncate_to_tokens(
            page_text,
            max(200, settings.parent_context_max_tokens),
        )
        if parent_text:
            parent_title = (
                hierarchy.section_title
                or hierarchy.lesson_title
                or hierarchy.chapter_title
                or f"صفحة {page_number}"
            )
            parent_path = _build_hierarchy_path(
                hierarchy=hierarchy,
                section_title=parent_title,
                page_number=page_number,
            )
            all_chunks.append(
                {
                    "page_number": page_number,
                    "content": _build_parent_context(
                        content=parent_text,
                        hierarchy_path=parent_path,
                        page_number=page_number,
                    ),
                    "local_index": -1,
                    "chunk_index": chunk_index,
                    "section_title": parent_title,
                    "content_type": "parent_context",
                }
            )
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
    for candidate in _heading_candidates(text):
        if _looks_like_heading(candidate):
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


def _update_hierarchy_from_page(text: str, hierarchy: HierarchyState) -> None:
    candidates = _heading_candidates(text)[:14]
    for candidate in candidates:
        lowered = candidate.casefold()
        if any(marker in lowered for marker in CHAPTER_MARKERS):
            hierarchy.chapter_title = candidate
            hierarchy.lesson_title = None
            hierarchy.section_title = candidate
            continue
        if any(marker in lowered for marker in LESSON_MARKERS):
            hierarchy.lesson_title = candidate
            hierarchy.section_title = candidate
            continue
        if hierarchy.section_title is None and _looks_like_heading(candidate):
            hierarchy.section_title = candidate


def _heading_candidates(text: str) -> list[str]:
    candidates: list[str] = []
    for line in text.split("\n"):
        candidate = SPACE_RE.sub(" ", line).strip(" -–—:؛")
        if not candidate or PAGE_NUMBER_RE.fullmatch(candidate):
            continue
        if len(candidate) > 160 or estimate_token_count(candidate) > 20:
            continue
        candidates.append(candidate)
    return candidates


def _looks_like_heading(candidate: str) -> bool:
    if len(candidate) < 3 or len(candidate) > 160:
        return False
    if estimate_token_count(candidate) > 20:
        return False
    if candidate.endswith(('.', '؟', '?', '!', '؛')):
        return False
    return True


def _build_hierarchy_path(
    *,
    hierarchy: HierarchyState,
    section_title: str | None,
    page_number: int,
) -> str:
    parts: list[str] = []
    for value in (
        hierarchy.chapter_title,
        hierarchy.lesson_title,
        section_title,
    ):
        normalized = SPACE_RE.sub(" ", value or "").strip()
        if normalized and normalized not in parts:
            parts.append(normalized)
    parts.append(f"صفحة {page_number}")
    return " > ".join(parts)


def _contextualize_content(
    *,
    content: str,
    hierarchy_path: str,
    page_number: int,
    content_type: str,
) -> str:
    return (
        "[سياق المقطع / Chunk context]\n"
        f"المسار: {hierarchy_path}\n"
        f"الصفحة: {page_number}\n"
        f"النوع: {content_type}\n"
        "[/سياق المقطع]\n\n"
        f"{content}"
    ).strip()


def _build_parent_context(
    *,
    content: str,
    hierarchy_path: str,
    page_number: int,
) -> str:
    return (
        "[سياق القسم / Parent context]\n"
        f"المسار: {hierarchy_path}\n"
        f"الصفحة: {page_number}\n"
        "[/سياق القسم]\n\n"
        f"{content}"
    ).strip()


def _truncate_to_tokens(text: str, max_tokens: int) -> str:
    tokens = TOKEN_RE.findall(text or "")
    if len(tokens) <= max_tokens:
        return text
    return " ".join(tokens[:max_tokens])
