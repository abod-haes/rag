import re
from collections.abc import Iterable

from app.core.config import get_settings
from app.db.database import dict_cursor, get_connection
from app.services.embedding_service import to_pgvector


TERM_RE = re.compile(r"[\w\u0600-\u06ff]+", re.UNICODE)
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
    "وضح",
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
}


class RetrievalService:
    def __init__(self) -> None:
        self.settings = get_settings()

    def retrieve(
        self,
        *,
        query_embedding: list[float],
        query_text: str,
        user_id: str,
        project_id: str,
        document_ids: list[str] | None = None,
        top_k: int | None = None,
    ) -> list[dict]:
        limit = top_k or self.settings.top_k
        candidate_limit = max(limit, self.settings.retrieval_candidate_k)
        query_vector = to_pgvector(query_embedding)

        vector_rows = self._retrieve_vector_candidates(
            query_vector=query_vector,
            user_id=user_id,
            project_id=project_id,
            document_ids=document_ids,
            limit=candidate_limit,
        )
        lexical_rows = self._retrieve_lexical_candidates(
            query_text=query_text,
            query_vector=query_vector,
            user_id=user_id,
            project_id=project_id,
            document_ids=document_ids,
            limit=candidate_limit,
        )

        ranked = self._merge_and_rerank(
            query_text=query_text,
            vector_rows=vector_rows,
            lexical_rows=lexical_rows,
        )
        core_results = ranked[:limit]

        if not core_results or self.settings.neighbor_window <= 0:
            return core_results[: self.settings.max_context_chunks]

        return self._expand_neighbors(
            core_results=core_results,
            query_vector=query_vector,
            user_id=user_id,
            project_id=project_id,
        )

    def _retrieve_vector_candidates(
        self,
        *,
        query_vector: str,
        user_id: str,
        project_id: str,
        document_ids: list[str] | None,
        limit: int,
    ) -> list[dict]:
        document_filter = ""
        params: list = [query_vector, user_id, project_id]
        if document_ids:
            document_filter = "AND dc.document_id = ANY(%s::uuid[])"
            params.append(document_ids)
        params.extend([query_vector, limit])

        sql = f"""
            SELECT
                dc.document_id::text,
                COALESCE(NULLIF(BTRIM(d.name), ''), d.file_name) AS name,
                d.file_name,
                dc.content,
                dc.page_number,
                dc.chunk_index,
                dc.section_title,
                dc.content_type,
                (dc.embedding <=> %s::vector) AS distance
            FROM document_chunks dc
            JOIN documents d ON d.id = dc.document_id
            WHERE dc.user_id = %s
              AND dc.project_id = %s
              {document_filter}
            ORDER BY dc.embedding <=> %s::vector
            LIMIT %s;
        """

        with get_connection(cursor_factory=dict_cursor()) as (_, cursor):
            cursor.execute(sql, params)
            return list(cursor.fetchall())

    def _retrieve_lexical_candidates(
        self,
        *,
        query_text: str,
        query_vector: str,
        user_id: str,
        project_id: str,
        document_ids: list[str] | None,
        limit: int,
    ) -> list[dict]:
        if not _extract_terms(query_text):
            return []

        document_filter = ""
        params: list = [query_text, query_vector, user_id, project_id]
        if document_ids:
            document_filter = "AND dc.document_id = ANY(%s::uuid[])"
            params.append(document_ids)
        params.append(limit)

        sql = f"""
            WITH query_data AS (
                SELECT plainto_tsquery('simple', %s) AS query
            )
            SELECT
                dc.document_id::text,
                COALESCE(NULLIF(BTRIM(d.name), ''), d.file_name) AS name,
                d.file_name,
                dc.content,
                dc.page_number,
                dc.chunk_index,
                dc.section_title,
                dc.content_type,
                (dc.embedding <=> %s::vector) AS distance,
                ts_rank_cd(dc.search_vector, query_data.query) AS lexical_score
            FROM document_chunks dc
            JOIN documents d ON d.id = dc.document_id
            CROSS JOIN query_data
            WHERE dc.user_id = %s
              AND dc.project_id = %s
              {document_filter}
              AND dc.search_vector @@ query_data.query
            ORDER BY lexical_score DESC
            LIMIT %s;
        """

        with get_connection(cursor_factory=dict_cursor()) as (_, cursor):
            cursor.execute(sql, params)
            return list(cursor.fetchall())

    def _merge_and_rerank(
        self,
        *,
        query_text: str,
        vector_rows: list[dict],
        lexical_rows: list[dict],
    ) -> list[dict]:
        candidates: dict[tuple[str, int], dict] = {}

        for rank, row in enumerate(vector_rows, start=1):
            item = _row_to_candidate(row)
            item["vector_rank"] = rank
            item["lexical_score"] = 0.0
            candidates[_candidate_key(item)] = item

        for rank, row in enumerate(lexical_rows, start=1):
            key = (row["document_id"], int(row["chunk_index"]))
            item = candidates.get(key) or _row_to_candidate(row)
            item["lexical_rank"] = rank
            item["lexical_score"] = float(row.get("lexical_score") or 0.0)
            candidates[key] = item

        max_lexical = max(
            (float(item.get("lexical_score") or 0.0) for item in candidates.values()),
            default=0.0,
        )
        query_terms = _extract_terms(query_text)
        weights_total = max(
            0.0001,
            self.settings.vector_weight
            + self.settings.lexical_weight
            + self.settings.exact_match_weight,
        )

        ranked: list[dict] = []
        for item in candidates.values():
            vector_score = max(0.0, 1.0 - float(item["distance"]))
            lexical_score = (
                float(item.get("lexical_score") or 0.0) / max_lexical
                if max_lexical > 0
                else 0.0
            )
            exact_score = _exact_overlap_score(query_terms, item["content"])
            hybrid_score = (
                self.settings.vector_weight * vector_score
                + self.settings.lexical_weight * lexical_score
                + self.settings.exact_match_weight * exact_score
            ) / weights_total

            item.update(
                {
                    "vector_score": vector_score,
                    "lexical_score": lexical_score,
                    "exact_match_score": exact_score,
                    "score": hybrid_score,
                    "is_neighbor": False,
                }
            )

            if (
                hybrid_score >= self.settings.min_relevance_score
                or lexical_score > 0
                or exact_score >= 0.5
            ):
                ranked.append(item)

        ranked.sort(
            key=lambda item: (
                item["score"],
                item["vector_score"],
                item["lexical_score"],
            ),
            reverse=True,
        )
        return ranked

    def _expand_neighbors(
        self,
        *,
        core_results: list[dict],
        query_vector: str,
        user_id: str,
        project_id: str,
    ) -> list[dict]:
        window = self.settings.neighbor_window
        requested_pairs: set[str] = set()
        for item in core_results:
            for chunk_index in range(
                max(0, item["chunk_index"] - window),
                item["chunk_index"] + window + 1,
            ):
                requested_pairs.add(f"{item['document_id']}:{chunk_index}")

        if not requested_pairs:
            return core_results[: self.settings.max_context_chunks]

        sql = """
            SELECT
                dc.document_id::text,
                COALESCE(NULLIF(BTRIM(d.name), ''), d.file_name) AS name,
                d.file_name,
                dc.content,
                dc.page_number,
                dc.chunk_index,
                dc.section_title,
                dc.content_type,
                (dc.embedding <=> %s::vector) AS distance
            FROM document_chunks dc
            JOIN documents d ON d.id = dc.document_id
            WHERE dc.user_id = %s
              AND dc.project_id = %s
              AND (dc.document_id::text || ':' || dc.chunk_index::text) = ANY(%s::text[]);
        """

        with get_connection(cursor_factory=dict_cursor()) as (_, cursor):
            cursor.execute(
                sql,
                [query_vector, user_id, project_id, sorted(requested_pairs)],
            )
            rows = cursor.fetchall()

        row_map = {
            (row["document_id"], int(row["chunk_index"])): _row_to_candidate(row)
            for row in rows
        }
        core_map = {_candidate_key(item): item for item in core_results}
        expanded: list[dict] = []
        seen: set[tuple[str, int]] = set()

        for core in core_results:
            for chunk_index in range(
                max(0, core["chunk_index"] - window),
                core["chunk_index"] + window + 1,
            ):
                key = (core["document_id"], chunk_index)
                if key in seen:
                    continue

                if key in core_map:
                    item = dict(core_map[key])
                else:
                    neighbor = row_map.get(key)
                    if neighbor is None:
                        continue
                    item = dict(neighbor)
                    item.update(
                        {
                            "vector_score": max(0.0, 1.0 - item["distance"]),
                            "lexical_score": 0.0,
                            "exact_match_score": 0.0,
                            "score": core["score"] * 0.90,
                            "is_neighbor": True,
                            "neighbor_of_chunk_index": core["chunk_index"],
                        }
                    )

                expanded.append(item)
                seen.add(key)
                if len(expanded) >= self.settings.max_context_chunks:
                    return expanded

        return expanded


def _row_to_candidate(row: dict) -> dict:
    distance = float(row["distance"])
    return {
        "document_id": row["document_id"],
        "name": row["name"],
        "file_name": row["file_name"],
        "content": row["content"],
        "page_number": row["page_number"],
        "chunk_index": int(row["chunk_index"]),
        "section_title": row.get("section_title"),
        "content_type": row.get("content_type") or "text",
        "distance": distance,
    }


def _candidate_key(item: dict) -> tuple[str, int]:
    return item["document_id"], int(item["chunk_index"])


def _extract_terms(text: str) -> set[str]:
    terms = {
        term.casefold()
        for term in TERM_RE.findall(text or "")
        if term.casefold() not in STOP_WORDS
    }
    return {term for term in terms if len(term) > 1 or term.isdigit()}


def _exact_overlap_score(query_terms: set[str], content: str) -> float:
    if not query_terms:
        return 0.0
    content_terms = _extract_terms(content)
    return len(query_terms & content_terms) / len(query_terms)
