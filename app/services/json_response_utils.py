import json
import re
from typing import Any


FENCED_JSON_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)
SOURCE_MARKER_RE = re.compile(
    r"\s*\[(?:S|s)\d+(?:\s*,\s*(?:page|p\.?|صفحة)?\s*\d+)?\]",
    re.IGNORECASE,
)


def parse_json_object(value: str) -> dict[str, Any] | None:
    text = (value or "").strip()
    if not text:
        return None

    fenced = FENCED_JSON_RE.search(text)
    if fenced:
        text = fenced.group(1).strip()

    candidates = [text]
    first = text.find("{")
    last = text.rfind("}")
    if first >= 0 and last > first:
        candidates.append(text[first : last + 1])

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def strip_source_markers(value: str) -> str:
    cleaned = SOURCE_MARKER_RE.sub("", value or "")
    cleaned = re.sub(r"[ \t]+\n", "\n", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def clamp_score(value: Any, default: float = 0.0) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return max(0.0, min(1.0, default))
    return max(0.0, min(1.0, score))
