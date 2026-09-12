from dataclasses import dataclass

from app.core.config import get_settings


ACCEPT = "accept"
RETRY = "retry"
ABSTAIN = "abstain"


@dataclass(frozen=True)
class RetrievalGateDecision:
    status: str
    score: float
    reason: str


class RetrievalGateService:
    def __init__(self) -> None:
        self.settings = get_settings()

    def assess(
        self,
        candidates: list[dict],
        *,
        retry_count: int = 0,
    ) -> RetrievalGateDecision:
        if not candidates:
            return RetrievalGateDecision(
                status=ABSTAIN,
                score=0.0,
                reason="no_candidates",
            )

        top = candidates[0]
        semantic_score = float(
            top.get("rerank_score")
            if top.get("rerank_score") is not None
            else top.get("score")
            or 0.0
        )
        hybrid_score = float(
            top.get("hybrid_score")
            if top.get("hybrid_score") is not None
            else top.get("score")
            or 0.0
        )
        exact_score = float(top.get("exact_match_score") or 0.0)
        lexical_score = float(top.get("lexical_score") or 0.0)

        strong_lexical_support = exact_score >= 0.65 or lexical_score >= 0.70
        if (
            semantic_score >= self.settings.retrieval_gate_accept_score
            or (
                semantic_score >= self.settings.retrieval_gate_retry_score
                and hybrid_score >= self.settings.retrieval_gate_min_hybrid_score
                and strong_lexical_support
            )
        ):
            return RetrievalGateDecision(
                status=ACCEPT,
                score=max(semantic_score, hybrid_score),
                reason="strong_support",
            )

        retries_allowed = retry_count < max(0, self.settings.retrieval_gate_max_retries)
        if retries_allowed and (
            semantic_score >= self.settings.retrieval_gate_retry_score
            or hybrid_score >= self.settings.retrieval_gate_min_hybrid_score
            or exact_score >= 0.20
            or lexical_score > 0
        ):
            return RetrievalGateDecision(
                status=RETRY,
                score=max(semantic_score, hybrid_score),
                reason="weak_but_plausible_support",
            )

        return RetrievalGateDecision(
            status=ABSTAIN,
            score=max(semantic_score, hybrid_score),
            reason="insufficient_support",
        )
