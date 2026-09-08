import re
import uuid
from dataclasses import dataclass
from pathlib import Path

from app.core.config import get_settings
from app.db.database import dict_cursor, get_connection
from app.services.embedding_service import to_pgvector


TERM_RE = re.compile(r"[\w\u0600-\u06ff]+", re.UNICODE)
ARABIC_RE = re.compile(r"[\u0600-\u06ff]")

STOP_WORDS = {
    "من",
    "في",
    "على",
    "إلى",
    "الى",
    "عن",
    "ما",
    "ماذا",
    "هل",
    "هو",
    "هي",
    "هذا",
    "هذه",
    "ذلك",
    "اشرح",
    "اشرحلي",
    "وضح",
    "وضّح",
    "اعطيني",
    "عطيني",
    "the",
    "a",
    "an",
    "of",
    "in",
    "on",
    "to",
    "is",
    "are",
    "what",
    "how",
    "explain",
    "give",
    "me",
}

GENERIC_TERMS = {
    "درس",
    "الدرس",
    "كتاب",
    "الكتاب",
    "ملف",
    "الملف",
    "مادة",
    "المادة",
    "سؤال",
    "السؤال",
    "تمرين",
    "التمرين",
    "مثال",
    "المثال",
    "حل",
    "شرح",
    "اول",
    "أول",
    "الاول",
    "الأول",
    "تاني",
    "ثاني",
    "الثاني",
    "التالي",
    "lesson",
    "book",
    "file",
    "subject",
    "question",
    "exercise",
    "example",
}

SUBJECT_RULES: list[tuple[str, tuple[str, ...]]] = [
    (
        "الرياضيات",
        (
            "رياضيات",
            "جبر",
            "هندسة",
            "تفاضل",
            "تكامل",
            "متتاليات",
            "احتمالات",
            "مثلثات",
        ),
    ),
    ("الفيزياء", ("فيزياء", "ميكانيك", "كهرباء", "مغناطيس", "حركة", "طاقة")),
    ("الكيمياء", ("كيمياء", "عضوية", "تفاعلات", "ذرة", "جزيئات")),
    ("الأحياء", ("أحياء", "احياء", "بيولوجيا", "وراثة", "خلية", "مجهرية")),
    ("اللغة العربية", ("لغة عربية", "العربي", "نحو", "صرف", "بلاغة", "أدب عربي")),
    ("اللغة الإنكليزية", ("إنكليزي", "انكليزي", "English", "grammar", "vocabulary")),
    ("اللغة الفرنسية", ("فرنسي", "French", "français")),
    ("أمن المعلومات", ("أمن معلومات", "امن معلومات", "تشفير", "أمن شبكات", "cyber", "security")),
    ("نظم التشغيل", ("نظم تشغيل", "نظام تشغيل", "operating system", "process", "fork")),
    ("الشبكات", ("شبكات", "شبكة", "network", "routing", "tcp", "ip")),
    ("البرمجة", ("برمجة", "خوارزميات", "javascript", "python", "java", "c++", "react")),
    ("قواعد البيانات", ("قواعد بيانات", "قاعدة بيانات", "database", "sql", "postgresql")),
    ("التاريخ", ("تاريخ", "حضارة", "حرب عالمية")),
    ("الجغرافيا", ("جغرافيا", "مناخ", "تضاريس", "سكان")),
    ("الفلسفة", ("فلسفة", "منطق", "فكر فلسفي")),
    ("التربية الدينية", ("تربية دينية", "إسلامية", "اسلامية", "فقه", "حديث", "قرآن")),
]


class DocumentRoutingError(Exception):
    pass


@dataclass(frozen=True)
class DocumentRoutingResult:
    status: str
    selected_documents: list[dict]
    candidates: list[dict]
    # Kept with the old API name for compatibility. In clarification responses
    # this now contains every available document name, not inferred subjects.
    candidate_subjects: list[str]
    clarification_question: str | None = None

    @property
    def selected_document_ids(self) -> list[str]:
        return [item["documentId"] for item in self.selected_documents]


class DocumentRoutingService:
    def __init__(self) -> None:
        self.settings = get_settings()

    def route(
        self,
        *,
        query_embedding: list[float],
        query_text: str,
        user_id: str,
        project_id: str,
        explicit_document_ids: list[str] | None = None,
        active_document_ids: list[str] | None = None,
    ) -> DocumentRoutingResult:
        # Explicit document IDs come from the trusted Quizy backend after it has
        # resolved the student's curriculum. Curriculum files are uploaded by
        # admins, so their RAG user_id differs from the student's conversation
        # user_id. Keep the project as the tenant boundary while allowing those
        # explicitly selected ready documents to be used by the student.
        if explicit_document_ids:
            selected = self._select_explicit_documents(
                project_id=project_id,
                document_ids=explicit_document_ids,
            )
            return self._selected_result(selected, reason="explicit")

        documents = self._list_ready_documents(
            user_id=user_id,
            project_id=project_id,
        )

        if not documents:
            return DocumentRoutingResult(
                status="clarification",
                selected_documents=[],
                candidates=[],
                candidate_subjects=[],
                clarification_question=_clarification_question(
                    query_text=query_text,
                    no_documents=True,
                ),
            )

        # A user may answer a clarification by typing the displayed file name.
        # Prefer that explicit textual choice before semantic routing.
        name_selected = _select_by_document_name(documents, query_text)
        if name_selected:
            return self._selected_result(name_selected, reason="name_match")

        active_ids = {str(value) for value in (active_document_ids or [])}
        active_documents = [item for item in documents if item["id"] in active_ids]
        query_subject = infer_subject(query_text)

        if active_documents and _is_underspecified(query_text):
            return self._selected_result(
                active_documents[: self.settings.document_routing_max_documents],
                reason="conversation",
            )

        if len(documents) == 1:
            return self._selected_result(documents, reason="only_document")

        if _is_underspecified(query_text) and not query_subject:
            return self._clarification_result(
                documents=documents,
                query_text=query_text,
            )

        ranked = self._score_documents(
            documents=documents,
            query_embedding=query_embedding,
            query_text=query_text,
            user_id=user_id,
            project_id=project_id,
            active_ids=active_ids,
            query_subject=query_subject,
        )

        selected = self._choose_documents(ranked, query_subject=query_subject)
        if selected:
            public_ranked = [
                _public_document(
                    item,
                    score=float(item.get("score") or 0.0),
                    reason="candidate",
                )
                for item in ranked
            ]
            return DocumentRoutingResult(
                status="selected",
                selected_documents=[
                    _public_document(
                        item,
                        score=float(item.get("score") or 0.0),
                        reason="automatic",
                    )
                    for item in selected
                ],
                candidates=public_ranked,
                candidate_subjects=_document_names(documents),
            )

        return self._clarification_result(
            documents=ranked or documents,
            query_text=query_text,
        )

    def _selected_result(self, documents: list[dict], *, reason: str) -> DocumentRoutingResult:
        return DocumentRoutingResult(
            status="selected",
            selected_documents=[
                _public_document(
                    item,
                    score=float(item.get("score") or 1.0),
                    reason=reason,
                )
                for item in documents
            ],
            candidates=[],
            candidate_subjects=_document_names(documents),
        )

    def _clarification_result(
        self,
        *,
        documents: list[dict],
        query_text: str,
    ) -> DocumentRoutingResult:
        public_documents = [
            _public_document(
                item,
                score=float(item.get("score") or 0.0),
                reason="candidate",
            )
            for item in documents
        ]
        names = _document_names(documents)
        return DocumentRoutingResult(
            status="clarification",
            selected_documents=[],
            candidates=public_documents,
            candidate_subjects=names,
            clarification_question=_clarification_question(
                query_text=query_text,
                document_names=names,
            ),
        )

    def _list_ready_documents(
        self,
        *,
        user_id: str,
        project_id: str,
    ) -> list[dict]:
        with get_connection(cursor_factory=dict_cursor()) as (_, cursor):
            cursor.execute(
                """
                SELECT
                    d.id::text,
                    COALESCE(NULLIF(BTRIM(d.name), ''), d.file_name) AS name,
                    d.file_name,
                    sample.content AS sample_content
                FROM documents d
                LEFT JOIN LATERAL (
                    SELECT dc.content
                    FROM document_chunks dc
                    WHERE dc.document_id = d.id
                    ORDER BY dc.chunk_index
                    LIMIT 1
                ) sample ON TRUE
                WHERE d.user_id = %s
                  AND d.project_id = %s
                  AND d.status = 'ready'
                ORDER BY d.updated_at DESC, d.created_at DESC
                """,
                (user_id, project_id),
            )
            rows = list(cursor.fetchall())

        return self._hydrate_documents(rows)

    def _select_explicit_documents(
        self,
        *,
        project_id: str,
        document_ids: list[str],
    ) -> list[dict]:
        normalized_ids: list[str] = []
        for value in document_ids:
            try:
                normalized = str(uuid.UUID(str(value).strip()))
            except (ValueError, TypeError, AttributeError) as exc:
                raise DocumentRoutingError("Invalid documentId") from exc
            if normalized not in normalized_ids:
                normalized_ids.append(normalized)

        with get_connection(cursor_factory=dict_cursor()) as (_, cursor):
            cursor.execute(
                """
                SELECT
                    d.id::text,
                    COALESCE(NULLIF(BTRIM(d.name), ''), d.file_name) AS name,
                    d.file_name,
                    sample.content AS sample_content
                FROM documents d
                LEFT JOIN LATERAL (
                    SELECT dc.content
                    FROM document_chunks dc
                    WHERE dc.document_id = d.id
                    ORDER BY dc.chunk_index
                    LIMIT 1
                ) sample ON TRUE
                WHERE d.project_id = %s
                  AND d.id = ANY(%s::uuid[])
                  AND d.status = 'ready'
                """,
                (project_id, normalized_ids),
            )
            rows = list(cursor.fetchall())

        document_map = {
            item["id"]: item for item in self._hydrate_documents(rows)
        }
        missing = [value for value in normalized_ids if value not in document_map]
        if missing:
            raise DocumentRoutingError(
                "One or more documents were not found or are not ready"
            )
        return [document_map[value] for value in normalized_ids]

    def _hydrate_documents(self, rows: list[dict]) -> list[dict]:
        documents: list[dict] = []
        for row in rows:
            item = dict(row)
            item["subject"] = infer_subject(
                " ".join(
                    [
                        str(item.get("name") or ""),
                        str(item.get("file_name") or ""),
                        str(item.get("sample_content") or "")[:4000],
                    ]
                )
            ) or "غير محددة"
            documents.append(item)
        return documents

    def _score_documents(
        self,
        *,
        documents: list[dict],
        query_embedding: list[float],
        query_text: str,
        user_id: str,
        project_id: str,
        active_ids: set[str],
        query_subject: str | None,
    ) -> list[dict]:
        query_vector = to_pgvector(query_embedding)
        vector_scores = self._vector_scores(
            query_vector=query_vector,
            user_id=user_id,
            project_id=project_id,
        )
        lexical_scores = self._lexical_scores(
            query_text=query_text,
            user_id=user_id,
            project_id=project_id,
        )
        max_lexical = max(lexical_scores.values(), default=0.0)
        query_terms = _extract_terms(query_text)

        candidates: list[dict] = []
        for document in documents:
            item = dict(document)
            vector_score = vector_scores.get(item["id"], 0.0)
            lexical_score = (
                lexical_scores.get(item["id"], 0.0) / max_lexical
                if max_lexical > 0
                else 0.0
            )
            name_score = _overlap_score(
                query_terms,
                f"{item.get('name', '')} {item.get('file_name', '')}",
            )
            subject_score = (
                1.0
                if query_subject and item["subject"] == query_subject
                else 0.0
            )
            active_bonus = (
                self.settings.document_routing_active_boost
                if item["id"] in active_ids
                else 0.0
            )
            score = min(
                1.0,
                0.50 * vector_score
                + 0.20 * lexical_score
                + 0.15 * name_score
                + 0.15 * subject_score
                + active_bonus,
            )
            item.update(
                {
                    "score": score,
                    "vectorScore": vector_score,
                    "lexicalScore": lexical_score,
                    "nameScore": name_score,
                    "subjectScore": subject_score,
                }
            )
            candidates.append(item)

        candidates.sort(
            key=lambda item: (
                item["score"],
                item["subjectScore"],
                item["nameScore"],
                item["vectorScore"],
            ),
            reverse=True,
        )
        return candidates

    def _vector_scores(
        self,
        *,
        query_vector: str,
        user_id: str,
        project_id: str,
    ) -> dict[str, float]:
        limit = max(20, self.settings.document_routing_candidate_chunks)
        with get_connection(cursor_factory=dict_cursor()) as (_, cursor):
            cursor.execute(
                """
                SELECT
                    dc.document_id::text,
                    (dc.embedding <=> %s::vector) AS distance
                FROM document_chunks dc
                JOIN documents d ON d.id = dc.document_id
                WHERE dc.user_id = %s
                  AND dc.project_id = %s
                  AND d.status = 'ready'
                ORDER BY dc.embedding <=> %s::vector
                LIMIT %s
                """,
                (query_vector, user_id, project_id, query_vector, limit),
            )
            rows = cursor.fetchall()

        scores: dict[str, list[float]] = {}
        for row in rows:
            score = max(0.0, 1.0 - float(row["distance"]))
            scores.setdefault(row["document_id"], []).append(score)

        return {
            document_id: sum(values[:3]) / min(3, len(values))
            for document_id, values in scores.items()
        }

    def _lexical_scores(
        self,
        *,
        query_text: str,
        user_id: str,
        project_id: str,
    ) -> dict[str, float]:
        terms = _extract_terms(query_text)
        if not terms:
            return {}

        lexical_query = " | ".join(sorted(terms))
        with get_connection(cursor_factory=dict_cursor()) as (_, cursor):
            cursor.execute(
                """
                WITH query_data AS (
                    SELECT to_tsquery('simple', %s) AS query
                )
                SELECT
                    dc.document_id::text,
                    MAX(ts_rank_cd(dc.search_vector, query_data.query)) AS lexical_score
                FROM document_chunks dc
                JOIN documents d ON d.id = dc.document_id
                CROSS JOIN query_data
                WHERE dc.user_id = %s
                  AND dc.project_id = %s
                  AND d.status = 'ready'
                  AND dc.search_vector @@ query_data.query
                GROUP BY dc.document_id
                """,
                (lexical_query, user_id, project_id),
            )
            rows = cursor.fetchall()

        return {
            row["document_id"]: float(row["lexical_score"] or 0.0)
            for row in rows
        }

    def _choose_documents(
        self,
        candidates: list[dict],
        *,
        query_subject: str | None,
    ) -> list[dict]:
        if not candidates:
            return []

        top = candidates[0]
        if top["score"] < self.settings.document_routing_min_score:
            return []

        second = candidates[1] if len(candidates) > 1 else None
        if second is None:
            return [top]

        margin = top["score"] - second["score"]
        if margin >= self.settings.document_routing_ambiguity_margin:
            return [top]

        same_subject = [
            item
            for item in candidates
            if item["subject"] == top["subject"]
            and item["score"]
            >= top["score"] - self.settings.document_routing_ambiguity_margin
        ]
        if (
            top["subject"] != "غير محددة"
            and (query_subject == top["subject"] or top["subjectScore"] > 0)
        ):
            return same_subject[: self.settings.document_routing_max_documents]

        return []


def infer_subject(text: str) -> str | None:
    folded = (text or "").casefold()
    for subject, keywords in SUBJECT_RULES:
        if any(keyword.casefold() in folded for keyword in keywords):
            return subject
    return None


def _select_by_document_name(
    documents: list[dict],
    query_text: str,
) -> list[dict]:
    query = _normalize_match_text(query_text)
    if not query:
        return []

    matches: list[tuple[float, dict]] = []
    query_terms = _extract_terms(query_text)

    for document in documents:
        aliases = _document_aliases(document)
        best_score = 0.0
        for alias in aliases:
            normalized_alias = _normalize_match_text(alias)
            if not normalized_alias:
                continue

            if query == normalized_alias:
                best_score = max(best_score, 1.0)
                continue

            if len(normalized_alias) >= 6 and normalized_alias in query:
                best_score = max(best_score, 0.95)
                continue

            alias_terms = _extract_terms(alias)
            if alias_terms:
                overlap = len(query_terms & alias_terms) / len(alias_terms)
                best_score = max(best_score, overlap)

        if best_score > 0:
            matches.append((best_score, document))

    matches.sort(key=lambda item: item[0], reverse=True)
    if not matches:
        return []

    top_score, top_document = matches[0]
    second_score = matches[1][0] if len(matches) > 1 else 0.0
    if top_score >= 0.90 and top_score - second_score >= 0.10:
        return [top_document]
    if top_score == 1.0 and second_score < 1.0:
        return [top_document]
    return []


def _document_aliases(document: dict) -> list[str]:
    values = [
        str(document.get("name") or ""),
        str(document.get("file_name") or ""),
    ]
    file_name = str(document.get("file_name") or "")
    if file_name:
        values.append(Path(file_name).stem)
    return list(dict.fromkeys(value for value in values if value.strip()))


def _normalize_match_text(text: str) -> str:
    normalized = " ".join((text or "").casefold().split())
    normalized = normalized.removesuffix(".pdf")
    return normalized.strip()


def _extract_terms(text: str) -> set[str]:
    terms = {
        term.casefold()
        for term in TERM_RE.findall(text or "")
        if term.casefold() not in STOP_WORDS
    }
    return {term for term in terms if len(term) > 1 or term.isdigit()}


def _is_underspecified(text: str) -> bool:
    terms = _extract_terms(text)
    generic = {item.casefold() for item in GENERIC_TERMS}
    meaningful = {term for term in terms if term not in generic}
    return len(meaningful) == 0


def _overlap_score(query_terms: set[str], text: str) -> float:
    if not query_terms:
        return 0.0
    document_terms = _extract_terms(text)
    return len(query_terms & document_terms) / len(query_terms)


def _document_names(documents: list[dict]) -> list[str]:
    names: list[str] = []
    for item in documents:
        name = str(item.get("name") or item.get("file_name") or "").strip()
        if name and name not in names:
            names.append(name)
    return names


def _public_document(document: dict, *, score: float, reason: str) -> dict:
    return {
        "documentId": document["id"],
        "name": document.get("name") or document.get("file_name"),
        "fileName": document.get("file_name"),
        "subject": document.get("subject") or "غير محددة",
        "score": round(float(score), 4),
        "selectionReason": reason,
    }


def _clarification_question(
    *,
    query_text: str,
    document_names: list[str] | None = None,
    no_documents: bool = False,
) -> str:
    is_arabic = bool(ARABIC_RE.search(query_text or ""))
    if is_arabic:
        if no_documents:
            return "ما في ملفات جاهزة مفهرسة حاليًا. ارفع ملف أولًا وبعدين اسألني."
        return "أي ملف بدك تسأل منو؟ اختار اسم ملف من القائمة."

    if no_documents:
        return "There are no indexed documents ready yet. Upload a document first."
    return "Which document do you want to use? Choose a file name from the list."
