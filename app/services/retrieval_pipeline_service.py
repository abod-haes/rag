from dataclasses import dataclass

from app.core.config import get_settings
from app.services.chat_service import ChatService
from app.services.embedding_service import EmbeddingService
from app.services.retrieval_gate_service import ACCEPT, RETRY, RetrievalGateDecision, RetrievalGateService
from app.services.retrieval_v2_service import RetrievalV2Service
from app.services.semantic_reranker_service import SemanticRerankerService
from app.services.usage_service import TokenUsage


@dataclass(frozen=True)
class RetrievalPipelineResult:
    chunks: list[dict]
    candidates: list[dict]
    gate: RetrievalGateDecision
    retry_count: int
    retrieval_query: str
    candidate_count: int
    chat_usage: TokenUsage
    additional_embedding_tokens: int
    used_model_reranker: bool

    @property
    def accepted(self) -> bool:
        return self.gate.status == ACCEPT and bool(self.chunks)

    def diagnostics(self) -> dict:
        return {
            "gateStatus": self.gate.status,
            "gateScore": round(float(self.gate.score), 4),
            "gateReason": self.gate.reason,
            "retryCount": self.retry_count,
            "retrievalQuery": self.retrieval_query,
            "candidateCount": self.candidate_count,
            "usedSemanticReranker": self.used_model_reranker,
        }


class RetrievalPipelineService:
    def __init__(
        self,
        *,
        chat_service: ChatService | None = None,
        embedding_service: EmbeddingService | None = None,
    ) -> None:
        self.settings = get_settings()
        self.chat_service = chat_service or ChatService()
        self.embedding_service = embedding_service or EmbeddingService()
        self.retriever = RetrievalV2Service()
        self.reranker = SemanticRerankerService(self.chat_service)
        self.gate = RetrievalGateService()

    def retrieve(
        self,
        *,
        query: str,
        query_embedding: list[float],
        user_id: str,
        project_id: str,
        document_ids: list[str] | None,
        history: list[dict] | None = None,
    ) -> RetrievalPipelineResult:
        chat_usage = TokenUsage()
        additional_embedding_tokens = 0
        retry_count = 0
        retrieval_query = query
        active_embedding = query_embedding

        candidates = self.retriever.retrieve_candidates(
            query_embedding=query_embedding,
            query_text=query,
            user_id=user_id,
            project_id=project_id,
            document_ids=document_ids,
        )
        rerank_result = self.reranker.rerank(query=query, candidates=candidates)
        chat_usage = chat_usage + rerank_result.usage
        ranked = rerank_result.candidates
        used_model_reranker = rerank_result.used_model_reranker
        decision = self.gate.assess(ranked, retry_count=0)

        if decision.status == RETRY and self.settings.retrieval_gate_max_retries > 0:
            retry_count = 1
            retry_query, rewrite_usage = self._rewrite_for_retrieval(
                query=query,
                history=history or [],
                candidates=ranked,
            )
            chat_usage = chat_usage + rewrite_usage

            if retry_query and retry_query.casefold().strip() != query.casefold().strip():
                retry_embedding_result = self.embedding_service.embed_query_with_usage(retry_query)
                additional_embedding_tokens += retry_embedding_result.usage.input_tokens
                active_embedding = retry_embedding_result.values
                retrieval_query = retry_query

                retry_candidates = self.retriever.retrieve_candidates(
                    query_embedding=retry_embedding_result.values,
                    query_text=retry_query,
                    user_id=user_id,
                    project_id=project_id,
                    document_ids=document_ids,
                )
                merged = _merge_candidate_sets(ranked, retry_candidates)
                second_rerank = self.reranker.rerank(query=query, candidates=merged)
                chat_usage = chat_usage + second_rerank.usage
                ranked = second_rerank.candidates
                used_model_reranker = (
                    used_model_reranker or second_rerank.used_model_reranker
                )

            decision = self.gate.assess(ranked, retry_count=retry_count)

        if decision.status != ACCEPT:
            return RetrievalPipelineResult(
                chunks=[],
                candidates=ranked,
                gate=decision,
                retry_count=retry_count,
                retrieval_query=retrieval_query,
                candidate_count=len(ranked),
                chat_usage=chat_usage,
                additional_embedding_tokens=additional_embedding_tokens,
                used_model_reranker=used_model_reranker,
            )

        core_results = ranked[: max(1, self.settings.semantic_reranker_top_k)]
        chunks = self.retriever.expand_context(
            core_results=core_results,
            query_embedding=active_embedding,
            project_id=project_id,
        )
        return RetrievalPipelineResult(
            chunks=chunks,
            candidates=ranked,
            gate=decision,
            retry_count=retry_count,
            retrieval_query=retrieval_query,
            candidate_count=len(ranked),
            chat_usage=chat_usage,
            additional_embedding_tokens=additional_embedding_tokens,
            used_model_reranker=used_model_reranker,
        )

    def _rewrite_for_retrieval(
        self,
        *,
        query: str,
        history: list[dict],
        candidates: list[dict],
    ) -> tuple[str, TokenUsage]:
        hints: list[str] = []
        for item in candidates[:4]:
            document = str(item.get("name") or item.get("file_name") or "").strip()
            section = str(item.get("section_title") or "").strip()
            if document or section:
                hints.append(f"- {document} | {section}".strip())

        history_text = "\n".join(
            f"{item.get('role', 'user')}: {str(item.get('content') or '').strip()}"
            for item in history[-4:]
            if str(item.get("content") or "").strip()
        )
        prompt = f"""
Rewrite the student's academic search query once to improve curriculum retrieval.
Do not answer the question.

Rules:
- Preserve the original meaning, language, mathematical symbols, numbers, and constraints.
- Add only useful alternate terminology or a more explicit formulation that is already implied by the query/history.
- Do not invent a chapter, formula, answer, or fact.
- The weak candidate titles below are hints about vocabulary only; never assume they are correct.
- Return one search query only, with no quotes, explanation, bullets, or JSON.

ORIGINAL QUERY:
{query}

RECENT CONTEXT:
{history_text or 'No additional context.'}

WEAK RETRIEVAL HINTS:
{chr(10).join(hints) if hints else 'No hints.'}
""".strip()
        raw, usage = self.chat_service.generate_answer_with_usage(prompt)
        rewritten = " ".join((raw or "").strip().strip('"').split())
        if len(rewritten) > 1000:
            rewritten = rewritten[:1000].rstrip()
        return (rewritten or query), usage


def _merge_candidate_sets(first: list[dict], second: list[dict]) -> list[dict]:
    merged: dict[tuple[str, int], dict] = {}
    for item in [*first, *second]:
        key = (str(item["document_id"]), int(item["chunk_index"]))
        existing = merged.get(key)
        if existing is None:
            merged[key] = dict(item)
            continue

        existing_hybrid = float(existing.get("hybrid_score") or existing.get("score") or 0.0)
        incoming_hybrid = float(item.get("hybrid_score") or item.get("score") or 0.0)
        if incoming_hybrid > existing_hybrid:
            merged[key] = dict(item)

    return sorted(
        merged.values(),
        key=lambda item: float(item.get("hybrid_score") or item.get("score") or 0.0),
        reverse=True,
    )
