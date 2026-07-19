from dataclasses import dataclass
from decimal import Decimal
from typing import Any


ONE_MILLION = Decimal("1000000")


@dataclass(frozen=True)
class TokenUsage:
    input_tokens: int = 0
    cached_input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0

    def __add__(self, other: "TokenUsage") -> "TokenUsage":
        return TokenUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            cached_input_tokens=(
                self.cached_input_tokens + other.cached_input_tokens
            ),
            output_tokens=self.output_tokens + other.output_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
        )


def extract_openai_usage(raw_usage: Any) -> TokenUsage:
    if raw_usage is None:
        return TokenUsage()

    input_tokens = _int_value(
        getattr(raw_usage, "input_tokens", None)
        or getattr(raw_usage, "prompt_tokens", None)
    )
    output_tokens = _int_value(getattr(raw_usage, "output_tokens", None))
    total_tokens = _int_value(getattr(raw_usage, "total_tokens", None))

    input_details = getattr(raw_usage, "input_tokens_details", None)
    cached_input_tokens = _int_value(
        getattr(input_details, "cached_tokens", None) if input_details else None
    )

    return TokenUsage(
        input_tokens=input_tokens,
        cached_input_tokens=cached_input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens or input_tokens + output_tokens,
    )


def extract_gemini_usage(raw_usage: Any) -> TokenUsage:
    if raw_usage is None:
        return TokenUsage()

    input_tokens = _int_value(getattr(raw_usage, "prompt_token_count", None))
    cached_input_tokens = _int_value(
        getattr(raw_usage, "cached_content_token_count", None)
    )
    output_tokens = _int_value(
        getattr(raw_usage, "candidates_token_count", None)
    )
    total_tokens = _int_value(getattr(raw_usage, "total_token_count", None))

    return TokenUsage(
        input_tokens=input_tokens,
        cached_input_tokens=cached_input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens or input_tokens + output_tokens,
    )


def estimate_embedding_cost_usd(
    input_tokens: int,
    *,
    provider: str,
    settings: Any,
) -> Decimal:
    if provider != "openai":
        return Decimal("0")

    return (
        Decimal(input_tokens)
        * settings.openai_embedding_price_per_million_tokens
        / ONE_MILLION
    )


def estimate_chat_cost_usd(
    usage: TokenUsage,
    *,
    provider: str,
    settings: Any,
    purpose: str = "chat",
) -> Decimal:
    if provider != "openai":
        return Decimal("0")

    if purpose == "ocr":
        input_price = settings.openai_ocr_input_price_per_million_tokens
        cached_input_price = (
            settings.openai_ocr_cached_input_price_per_million_tokens
        )
        output_price = settings.openai_ocr_output_price_per_million_tokens
    else:
        input_price = settings.openai_chat_input_price_per_million_tokens
        cached_input_price = (
            settings.openai_chat_cached_input_price_per_million_tokens
        )
        output_price = settings.openai_chat_output_price_per_million_tokens

    cached_tokens = min(usage.input_tokens, usage.cached_input_tokens)
    uncached_tokens = max(0, usage.input_tokens - cached_tokens)

    return (
        Decimal(uncached_tokens) * input_price
        + Decimal(cached_tokens) * cached_input_price
        + Decimal(usage.output_tokens) * output_price
    ) / ONE_MILLION


def decimal_to_json(value: Decimal) -> float:
    return float(value.quantize(Decimal("0.0000000001")))


def _int_value(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0
