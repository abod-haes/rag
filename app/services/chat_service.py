from collections.abc import Iterator

from google import genai
from openai import OpenAI

from app.core.config import get_settings
from app.services.usage_service import (
    TokenUsage,
    extract_gemini_usage,
    extract_openai_usage,
)


class GeminiChatService:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.provider = self.settings.ai_provider.lower().strip()
        self.last_usage = TokenUsage()

        if self.provider == "openai":
            if not self.settings.openai_api_key:
                raise RuntimeError("OPENAI_API_KEY is missing")
            self.openai_client = OpenAI(api_key=self.settings.openai_api_key)
            self.gemini_client = None
        elif self.provider == "gemini":
            if not self.settings.gemini_api_key:
                raise RuntimeError("GEMINI_API_KEY is missing")
            self.gemini_client = genai.Client(api_key=self.settings.gemini_api_key)
            self.openai_client = None
        else:
            raise RuntimeError("AI_PROVIDER must be either 'openai' or 'gemini'")

    def generate_answer(self, prompt: str) -> str:
        answer, _ = self.generate_answer_with_usage(prompt)
        return answer

    def generate_answer_with_usage(self, prompt: str) -> tuple[str, TokenUsage]:
        if self.provider == "openai":
            return self._generate_openai(prompt)

        return self._generate_gemini(prompt)

    def stream_answer(self, prompt: str) -> Iterator[str]:
        self.last_usage = TokenUsage()
        if self.provider == "openai":
            yield from self._stream_openai(prompt)
            return

        # Preserve Gemini compatibility even when native streaming is unavailable.
        answer, usage = self._generate_gemini(prompt)
        self.last_usage = usage
        if answer:
            yield answer

    def _generate_openai(self, prompt: str) -> tuple[str, TokenUsage]:
        assert self.openai_client is not None

        response = self.openai_client.responses.create(
            model=self.settings.openai_chat_model,
            input=prompt,
        )

        text = getattr(response, "output_text", None)
        usage = extract_openai_usage(getattr(response, "usage", None))
        if not text:
            return "لا يوجد جواب واضح في الملفات المرفوعة.", usage

        return text.strip(), usage

    def _stream_openai(self, prompt: str) -> Iterator[str]:
        assert self.openai_client is not None

        stream = self.openai_client.responses.create(
            model=self.settings.openai_chat_model,
            input=prompt,
            stream=True,
        )

        try:
            for event in stream:
                event_type = getattr(event, "type", "")
                if event_type == "response.completed":
                    response = getattr(event, "response", None)
                    self.last_usage = extract_openai_usage(
                        getattr(response, "usage", None)
                    )
                    continue

                if event_type != "response.output_text.delta":
                    continue

                delta = getattr(event, "delta", None)
                if delta:
                    yield delta
        finally:
            close = getattr(stream, "close", None)
            if callable(close):
                close()

    def _generate_gemini(self, prompt: str) -> tuple[str, TokenUsage]:
        assert self.gemini_client is not None

        response = self.gemini_client.models.generate_content(
            model=self.settings.gemini_chat_model,
            contents=prompt,
        )

        text = getattr(response, "text", None)
        usage = extract_gemini_usage(getattr(response, "usage_metadata", None))
        if not text:
            return "لا يوجد جواب واضح في الملفات المرفوعة.", usage

        return text.strip(), usage
