from google import genai
from openai import OpenAI

from app.core.config import get_settings


class GeminiChatService:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.provider = self.settings.ai_provider.lower().strip()

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
        if self.provider == "openai":
            return self._generate_openai(prompt)

        return self._generate_gemini(prompt)

    def _generate_openai(self, prompt: str) -> str:
        assert self.openai_client is not None

        response = self.openai_client.responses.create(
            model=self.settings.openai_chat_model,
            input=prompt,
        )

        text = getattr(response, "output_text", None)
        if not text:
            return "لا يوجد جواب واضح في الملفات المرفوعة."

        return text.strip()

    def _generate_gemini(self, prompt: str) -> str:
        assert self.gemini_client is not None

        response = self.gemini_client.models.generate_content(
            model=self.settings.gemini_chat_model,
            contents=prompt,
        )

        text = getattr(response, "text", None)
        if not text:
            return "لا يوجد جواب واضح في الملفات المرفوعة."

        return text.strip()
