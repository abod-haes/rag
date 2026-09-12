import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path


FALLBACK_MARKERS = (
    "لا يوجد جواب",
    "لم يتم العثور",
    "تعذر توليد",
    "مو موجودة ضمن منهاجك",
    "مو موجود ضمن منهاجك",
    "not found",
    "insufficient",
    "not in your current curriculum",
)
SOURCE_MARKER_RE = re.compile(r"\[(?:S|s)\d+(?:\s*,[^\]]+)?\]")


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate the RAG HTTP API")
    parser.add_argument("dataset", type=Path, help="Path to an evaluation JSON file")
    parser.add_argument(
        "--base-url",
        default=os.getenv("RAG_BASE_URL", "http://localhost:8000"),
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("RAG_API_KEY", "change-this-secret"),
    )
    parser.add_argument(
        "--user-id",
        default=os.getenv("RAG_USER_ID", "default-user"),
    )
    parser.add_argument(
        "--project-id",
        default=os.getenv("RAG_PROJECT_ID", "default-project"),
    )
    parser.add_argument("--minimum-pass-rate", type=float, default=0.80)
    args = parser.parse_args()

    cases = json.loads(args.dataset.read_text(encoding="utf-8"))
    enabled_cases = [case for case in cases if case.get("enabled", True)]
    if not enabled_cases:
        print("No enabled evaluation cases were found.")
        return 2

    passed = 0
    results: list[dict] = []
    for case in enabled_cases:
        result = evaluate_case(
            case=case,
            base_url=args.base_url.rstrip("/"),
            api_key=args.api_key,
            user_id=args.user_id,
            project_id=args.project_id,
        )
        results.append(result)
        passed += int(result["passed"])
        status = "PASS" if result["passed"] else "FAIL"
        print(f"[{status}] {result['id']}: {result['reason']}")

    pass_rate = passed / len(results)
    confidence_values = [
        float(result["confidence"])
        for result in results
        if result.get("confidence") is not None
    ]
    retry_values = [int(result.get("retrievalRetryCount") or 0) for result in results]
    marker_leaks = sum(bool(result.get("sourceMarkerLeak")) for result in results)
    routing_counts: dict[str, int] = {}
    for result in results:
        routing = str(result.get("routingStatus") or "unknown")
        routing_counts[routing] = routing_counts.get(routing, 0) + 1

    print(
        json.dumps(
            {
                "passed": passed,
                "total": len(results),
                "passRate": round(pass_rate, 4),
                "averageConfidence": (
                    round(sum(confidence_values) / len(confidence_values), 4)
                    if confidence_values
                    else None
                ),
                "averageRetrievalRetries": (
                    round(sum(retry_values) / len(retry_values), 4)
                    if retry_values
                    else 0.0
                ),
                "sourceMarkerLeaks": marker_leaks,
                "routingCounts": routing_counts,
                "results": results,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if pass_rate >= args.minimum_pass_rate else 1


def evaluate_case(
    *,
    case: dict,
    base_url: str,
    api_key: str,
    user_id: str,
    project_id: str,
) -> dict:
    body = {
        "question": case["question"],
        "documentIds": case.get("documentIds") or None,
    }
    payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        f"{base_url}/api/chat/ask",
        data=payload,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
            "X-API-Key": api_key,
            "X-User-Id": user_id,
            "X-Project-Id": project_id,
        },
    )

    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return {
            "id": case.get("id", "unknown"),
            "passed": False,
            "reason": f"HTTP {exc.code}: {exc.read().decode('utf-8', errors='replace')}",
        }
    except Exception as exc:
        return {
            "id": case.get("id", "unknown"),
            "passed": False,
            "reason": f"Request failed: {exc}",
        }

    answer = str(data.get("answer") or "")
    sources = data.get("sources") or []
    diagnostics = data.get("retrievalDiagnostics") or {}
    answer_folded = answer.casefold()

    expected_keywords = [
        str(keyword).casefold() for keyword in case.get("expectedKeywords", [])
    ]
    missing_keywords = [
        keyword for keyword in expected_keywords if keyword not in answer_folded
    ]

    forbidden_phrases = [
        str(phrase).casefold() for phrase in case.get("forbiddenPhrases", [])
    ]
    found_forbidden = [
        phrase for phrase in forbidden_phrases if phrase in answer_folded
    ]

    expected_pages = {int(page) for page in case.get("expectedSourcePages", [])}
    actual_pages = {
        int(source["pageNumber"])
        for source in sources
        if source.get("pageNumber") is not None
    }
    source_page_ok = not expected_pages or bool(expected_pages & actual_pages)

    should_answer = bool(case.get("shouldAnswer", True))
    has_fallback = any(marker in answer_folded for marker in FALLBACK_MARKERS)
    answer_behavior_ok = (should_answer and not has_fallback) or (
        not should_answer and has_fallback
    )

    source_marker_leak = bool(SOURCE_MARKER_RE.search(answer))
    source_marker_ok = not case.get("forbidSourceMarkers", True) or not source_marker_leak

    expected_routing = case.get("expectedRoutingStatus")
    routing_status = data.get("routingStatus")
    routing_ok = expected_routing is None or routing_status == expected_routing

    minimum_confidence = case.get("minimumConfidence")
    confidence = data.get("confidence")
    confidence_ok = (
        minimum_confidence is None
        or confidence is not None
        and float(confidence) >= float(minimum_confidence)
    )

    expected_grounding = case.get("expectedGroundingMode")
    grounding_mode = data.get("groundingMode")
    grounding_ok = expected_grounding is None or grounding_mode == expected_grounding

    max_retries = case.get("maxRetrievalRetries")
    retry_count = int(diagnostics.get("retryCount") or 0)
    retries_ok = max_retries is None or retry_count <= int(max_retries)

    require_sources = bool(case.get("requireSources", False))
    sources_ok = not require_sources or bool(sources)

    passed = (
        not missing_keywords
        and not found_forbidden
        and source_page_ok
        and answer_behavior_ok
        and source_marker_ok
        and routing_ok
        and confidence_ok
        and grounding_ok
        and retries_ok
        and sources_ok
    )
    reasons: list[str] = []
    if missing_keywords:
        reasons.append(f"missing keywords: {missing_keywords}")
    if found_forbidden:
        reasons.append(f"forbidden phrases: {found_forbidden}")
    if not source_page_ok:
        reasons.append(
            f"expected one of pages {sorted(expected_pages)}, got {sorted(actual_pages)}"
        )
    if not answer_behavior_ok:
        reasons.append("answer/fallback behavior did not match shouldAnswer")
    if not source_marker_ok:
        reasons.append("source marker leaked into student-visible answer")
    if not routing_ok:
        reasons.append(f"expected routing {expected_routing}, got {routing_status}")
    if not confidence_ok:
        reasons.append(
            f"confidence {confidence} is below required {minimum_confidence}"
        )
    if not grounding_ok:
        reasons.append(
            f"expected grounding {expected_grounding}, got {grounding_mode}"
        )
    if not retries_ok:
        reasons.append(f"retrieval retried {retry_count} times; max is {max_retries}")
    if not sources_ok:
        reasons.append("expected at least one used source")

    return {
        "id": case.get("id", "unknown"),
        "passed": passed,
        "reason": "; ".join(reasons) if reasons else "all checks passed",
        "routingStatus": routing_status,
        "groundingMode": grounding_mode,
        "confidence": confidence,
        "sourcePages": sorted(actual_pages),
        "sourceMarkerLeak": source_marker_leak,
        "retrievalGateStatus": diagnostics.get("gateStatus"),
        "retrievalGateScore": diagnostics.get("gateScore"),
        "retrievalRetryCount": retry_count,
        "retrievedSourceCount": data.get("retrievedSourceCount", 0),
        "estimatedCostUsd": (data.get("usage") or {}).get("estimatedCostUsd"),
    }


if __name__ == "__main__":
    sys.exit(main())
