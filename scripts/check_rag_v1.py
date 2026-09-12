import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.api.chat_v5 import _fallback_intent
from app.api.chat_v6 import CURRICULUM, GENERAL_CHAT, _curriculum_fallback, router
from app.core.config import get_settings
from app.services.chunk_service import build_chunks_from_pages
from app.services.json_response_utils import parse_json_object, strip_source_markers
from app.services.retrieval_gate_service import (
    ACCEPT,
    ABSTAIN,
    RETRY,
    RetrievalGateService,
)
from app.services.semantic_reranker_service import SemanticRerankerService


def main() -> None:
    chunks = build_chunks_from_pages(
        [
            {
                "page_number": 12,
                "text": (
                    "الفصل الأول: التوابع\n"
                    "الدرس الثاني: التابع الأسي\n\n"
                    "تعريف التابع الأسي\n"
                    "يعرّف التابع الأسي ضمن شروطه المعطاة في الدرس.\n\n"
                    "مثال\n"
                    "نطبق القاعدة السابقة على المثال."
                ),
            }
        ]
    )
    child_chunks = [
        chunk for chunk in chunks if chunk["content_type"] != "parent_context"
    ]
    parent_chunks = [
        chunk for chunk in chunks if chunk["content_type"] == "parent_context"
    ]
    assert child_chunks, "contextual child chunks were not produced"
    assert parent_chunks, "parent context was not produced"
    assert "سياق المقطع" in child_chunks[0]["content"]
    assert "صفحة 12" in child_chunks[0]["content"]
    assert "الفصل الأول" in child_chunks[0]["content"]
    assert "سياق القسم" in parent_chunks[0]["content"]

    gate = RetrievalGateService()
    accept = gate.assess(
        [
            {
                "rerank_score": 0.92,
                "hybrid_score": 0.45,
                "exact_match_score": 0.4,
                "lexical_score": 0.5,
            }
        ]
    )
    retry = gate.assess(
        [
            {
                "rerank_score": 0.40,
                "hybrid_score": 0.20,
                "exact_match_score": 0.1,
                "lexical_score": 0.1,
            }
        ]
    )
    abstain = gate.assess(
        [
            {
                "rerank_score": 0.10,
                "hybrid_score": 0.05,
                "exact_match_score": 0.0,
                "lexical_score": 0.0,
            }
        ]
    )
    assert accept.status == ACCEPT
    assert retry.status == RETRY
    assert abstain.status == ABSTAIN

    parsed = parse_json_object(
        '```json\n{"answer":"جواب [S2]", "usedSourceIds":["S2"]}\n```'
    )
    assert parsed and parsed["usedSourceIds"] == ["S2"]
    assert (
        strip_source_markers("الجواب [S1] ومن المصدر [S2, page 18]")
        == "الجواب ومن المصدر"
    )

    reranker = SemanticRerankerService()
    scores = reranker._extract_scores(
        {
            "ranking": [
                {"candidateId": "C1", "score": 0.91},
                {"candidateId": "C2", "score": 0.25},
            ]
        },
        2,
    )
    assert scores["C1"] == 0.91
    assert scores["C2"] == 0.25

    # Obvious retrieval winners must not spend an extra chat-model call.
    should_rerank, reason = reranker.should_use_model_reranker(
        [
            {
                "hybrid_score": 0.76,
                "exact_match_score": 0.55,
                "lexical_score": 0.70,
            },
            {
                "hybrid_score": 0.53,
                "exact_match_score": 0.40,
                "lexical_score": 0.55,
            },
        ]
    )
    assert should_rerank is False
    assert reason == "clear_hybrid_winner"

    should_rerank, reason = reranker.should_use_model_reranker(
        [
            {
                "hybrid_score": 0.48,
                "exact_match_score": 0.90,
                "lexical_score": 0.65,
            },
            {
                "hybrid_score": 0.45,
                "exact_match_score": 0.50,
                "lexical_score": 0.55,
            },
        ]
    )
    assert should_rerank is False
    assert reason == "strong_exact_match"

    # Ambiguous candidates still get semantic reranking to protect answer quality.
    should_rerank, reason = reranker.should_use_model_reranker(
        [
            {
                "hybrid_score": 0.52,
                "exact_match_score": 0.45,
                "lexical_score": 0.60,
            },
            {
                "hybrid_score": 0.49,
                "exact_match_score": 0.42,
                "lexical_score": 0.58,
            },
        ]
    )
    assert should_rerank is True
    assert reason == "ambiguous_candidates"

    settings = get_settings()
    assert settings.semantic_reranker_adaptive_enabled is True
    assert settings.semantic_reranker_candidate_k <= 8
    assert settings.semantic_reranker_max_chars_per_candidate <= 800

    assert _fallback_intent("مرحبا", []) == GENERAL_CHAT
    assert _fallback_intent("اشرح قانون نيوتن الثاني", []) == CURRICULUM
    fallback = _curriculum_fallback("اشرحلي الفيزياء")
    assert "منهاجك" in fallback
    assert "المحتوى الدراسي المتاح" not in fallback

    ask_routes = [
        route
        for route in router.routes
        if getattr(route, "path", None) == "/api/chat/ask"
        and "POST" in (getattr(route, "methods", set()) or set())
    ]
    stream_routes = [
        route
        for route in router.routes
        if getattr(route, "path", None) == "/api/chat/stream"
        and "POST" in (getattr(route, "methods", set()) or set())
    ]
    assert len(ask_routes) == 1
    assert len(stream_routes) == 1

    print("RAG V1 smoke checks passed")


if __name__ == "__main__":
    main()
