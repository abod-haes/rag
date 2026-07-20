import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path


FALLBACK_MARKERS = (
    "لا يوجد جواب",
    "لم يتم العثور",
    "تعذر توليد",
    "not found",
    "insufficient",
)


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
    print(
        json.dumps(
            {
                "passed": passed,
                "total": len(results),
                "passRate": round(pass_rate, 4),
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

    passed = (
        not missing_keywords
        and not found_forbidden
        and source_page_ok
        and answer_behavior_ok
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

    return {
        "id": case.get("id", "unknown"),
        "passed": passed,
        "reason": "; ".join(reasons) if reasons else "all checks passed",
        "sourcePages": sorted(actual_pages),
        "retrievedSourceCount": data.get("retrievedSourceCount", 0),
        "estimatedCostUsd": (data.get("usage") or {}).get("estimatedCostUsd"),
    }


if __name__ == "__main__":
    sys.exit(main())
