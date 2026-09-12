import re

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


class RetrievalV2Service:
    """High-recall candidate retrieval plus explicit parent/neighbor expansion.

    The first stage deliberately favors recall: it merges dense, lexical and exact
    signals and leaves precision to the semantic reranker. Parent-context chunks
    are never candidates by themselves; they are attached only after a child chunk
    has proved relevant, which implements the "index small, retrieve big" pattern.
    """

    def __init__(self) -> None:
        self.settings = get_settings()

    def retrieve_candidates(
        self,
        *,
        query_embedding: list[float],
        user_id: str,
        project_id: str,
        query_text: str,
        document_ids: list[str] | None = None,
        candidate_k: int | None = None,
    ) -> list[dict]:
        limit = max(
            self.settings.top_k,
            candidate_k
            or self.settings.semantic_reranker_candidate_k
            or self.settings.retrieval_candidate_k,
        )
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
        return self._merge_candidates(
            query_text=query_text,
            vector_rows=vector_rows,
            lexical_rows=lexical_rows,
        )[:limit]

    def expand_context(
        self,
        *,
        core_results: list[dict],
        query_embedding: list[float],
        project_id: str,
    ) -> list[dict]:
        if not core_results:
            return []

        max_context = max(1, self.settings.max_context_chunks)
        expanded: list[dict] = []
        seen: set[tuple[str, int, str]] = set()

        for core in core_results:
            if len(expanded) >= max_context:
                break

            direct = dict(core)
            direct.update(
                {
                    "is_neighbor": False,
                    "is_parent": False,
                    "retrieval_relation": "direct",
                }
            )
            self._append_unique(expanded, seen, direct, max_context)

            parent = self._load_parent_context(
                document_id=core["document_id"],
                page_number=core.get("page_number"),
                project_id=project_id,
                core=core,
            )
            if parent is not None:
                self._append_unique(expanded, seen, parent, max_context)
                continue

            for neighbor in self._load_legacy_neighbors(
                document_id=core["document_id"],
                chunk_index=int(core["chunk_index"]),
                project_id=project_id,
                query_embedding=query_embedding,
            ):
                self._append_unique(expanded, seen, neighbor, max_context)
                if len(expanded) >= max_context:
                    break

        return expanded[:max_context]

    def _retrieve_vector_candidates(
        self,
        *,
        query_vector: str,
        user_id: str,
        project_id: str,
        document_ids: list[str] | None,
        limit: int,
    ) -> list[dict]:
        if document_ids:
            scope_filter = "dc.project_id = %s AND dc.document_id = ANY(%s::uuid[])"
            params: list = [query_vector, project_id, document_ids, query_vector, limit]
        else:
            scope_filter = "dc.user_id = %s AND dc.project_id = %s"
            params = [query_vector, user_id, project_id, query_vector, limit]

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
            WHERE {scope_filter}
              AND d.status = 'ready'
              AND dc.content_type <> 'parent_context'
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
        terms = _extract_terms(query_text)
        if not terms:
            return []

        lexical_query = " | ".join(sorted(terms))
        if document_ids:
            scope_filter = "dc.project_id = %s AND dc.document_id = ANY(%s::uuid[])"
            params: list = [lexical_query, query_vector, project_id, document_ids, limit]
        else:
            scope_filter = "dc.user_id = %s AND dc.project_id = %s"
            params = [lexical_query, query_vector, user_id, project_id, limit]

        sql = f"""
            WITH query_data AS (
                SELECT to_tsquery('simple', %s) AS query
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
            WHERE {scope_filter}
              AND d.status = 'ready'
              AND dc.content_type <> 'parent_context'
              AND dc.search_vector @@ query_data.query
            ORDER BY lexical_score DESC
            LIMIT %s;
        """

        with get_connection(cursor_factory=dict_cursor()) as (_, cursor):
            cursor.execute(sql, params)
            return list(cursor.fetchall())

    def _merge_candidates(
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
            item["lexical_score_raw"] = 0.0
            candidates[_candidate_key(item)] = item

        for rank, row in enumerate(lexical_rows, start=1):
            key = (row["document_id"], int(row["chunk_index"]))
            item = candidates.get(key) or _row_to_candidate(row)
            item["lexical_rank"] = rank
            item["lexical_score_raw"] = float(row.get("lexical_score") or 0.0)
            candidates[key] = item

        max_lexical = max(
            (float(item.get("lexical_score_raw") or 0.0) for item in candidates.values()),
            default=0.0,
        )
        query_terms = _extract_terms(query_text)
        weights_total = max(
            0.0001,
            self.settings.vector_weight
            + self.settings.lexical_weight
            + self.settings.exact_match_weight,
        )
        recall_floor = max(0.05, self.settings.min_relevance_score * 0.40)

        ranked: list[dict] = []
        for item in candidates.values():
            vector_score = max(0.0, 1.0 - float(item["distance"]))
            lexical_score = (
                float(item.get("lexical_score_raw") or 0.0) / max_lexical
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
                    "hybrid_score": hybrid_score,
                    "is_neighbor": False,
                    "is_parent": False,
                }
            )

            if (
                hybrid_score >= recall_floor
                or lexical_score > 0
                or exact_score >= 0.25
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

    def _load_parent_context(
        self,
        *,
        document_id: str,
        page_number: int | None,
        project_id: str,
        core: dict,
    ) -> dict | None:
        if page_number is None:
            return None

        sql = """
            SELECT
                dc.document_id::text,
                COALESCE(NULLIF(BTRIM(d.name), ''), d.file_name) AS name,
                d.file_name,
                dc.content,
                dc.page_number,
                dc.chunk_index,
                dc.section_title,
                dc.content_type
            FROM document_chunks dc
            JOIN documents d ON d.id = dc.document_id
            WHERE dc.project_id = %s
              AND dc.document_id = %s::uuid
              AND dc.page_number = %s
              AND dc.content_type = 'parent_context'
              AND d.status = 'ready'
            ORDER BY dc.chunk_index
            LIMIT 1;
        """
        with get_connection(cursor_factory=dict_cursor()) as (_, cursor):
            cursor.execute(sql, (project_id, document_id, page_number))
            row = cursor.fetchone()

        if not row:
            return None

        semantic_score = float(core.get("rerank_score") or core.get("score") or 0.0)
        hybrid_score = float(core.get("hybrid_score") or core.get("score") or 0.0)
        return {
            "document_id": row["document_id"],
            "name": row["name"],
            "file_name": row["file_name"],
            "content": row["content"],
            "page_number": row["page_number"],
            "chunk_index": int(row["chunk_index"]),
            "section_title": row.get("section_title"),
            "content_type": row.get("content_type") or "parent_context",
            "distance": float(core.get("distance") or 1.0),
            "vector_score": float(core.get("vector_score") or 0.0),
            "lexical_score": float(core.get("lexical_score") or 0.0),
            "exact_match_score": float(core.get("exact_match_score") or 0.0),
            "hybrid_score": hybrid_score * 0.96,
            "score": float(core.get("score") or hybrid_score) * 0.96,
            "rerank_score": semantic_score * 0.96,
            "is_neighbor": False,
            "is_parent": True,
            "retrieval_relation": "parent_context",
        }

    def _load_legacy_neighbors(
        self,
        *,
        document_id: str,
        chunk_index: int,
        project_id: str,
        query_embedding: list[float],
    ) -> list[dict]:
        window = max(0, self.settings.neighbor_window)
        if window <= 0:
            return []

        search_radius = max(2, window * 2 + 1)
        query_vector = to_pgvector(query_embedding)
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
            WHERE dc.project_id = %s
              AND dc.document_id = %s::uuid
              AND dc.content_type <> 'parent_context'
              AND dc.chunk_index BETWEEN %s AND %s
              AND dc.chunk_index <> %s
              AND d.status = 'ready'
            ORDER BY ABS(dc.chunk_index - %s), dc.chunk_index
            LIMIT %s;
        """
        with get_connection(cursor_factory=dict_cursor()) as (_, cursor):
            cursor.execute(
                sql,
                (
                    query_vector,
                    project_id,
                    document_id,
                    max(0, chunk_index - search_radius),
                    chunk_index + search_radius,
                    chunk_index,
                    chunk_index,
                    window * 2,
                ),
            )
            rows = list(cursor.fetchall())

        neighbors: list[dict] = []
        for row in rows:
            item = _row_to_candidate(row)
            vector_score = max(0.0, 1.0 - item["distance"])
            item.update(
                {
                    "vector_score": vector_score,
                    "lexical_score": 0.0,
                    "exact_match_score": 0.0,
                    "hybrid_score": vector_score * 0.85,
                    "score": vector_score * 0.85,
                    "rerank_score": vector_score * 0.80,
                    "is_neighbor": True,
                    "is_parent": False,
                    "retrieval_relation": "neighboring_context",
                }
            )
            neighbors.append(item)
        return neighbors

    @staticmethod
    def _append_unique(
        target: list[dict],
        seen: set[tuple[str, int, str]],
        item: dict,
        limit: int,
    ) -> None:
        if len(target) >= limit:
            return
        key = (
            str(item["document_id"]),
            int(item["chunk_index"]),
            str(item.get("content_type") or "text"),
        )
        if key in seen:
            return
        seen.add(key)
        target.append(item)


def _row_to_candidate(row: dict) -> dict:
    distance = float(row.get("distance") or 0.0)
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
