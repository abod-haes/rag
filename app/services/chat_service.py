from google import genai

from app.core.config import get_settings


class GeminiChatService:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.client = genai.Client(api_key=self.settings.gemini_api_key)

    def generate_answer(self, prompt: str) -> str:
        response = self.client.models.generate_content(
            model=self.settings.gemini_chat_model,
            contents=prompt,
        )

        text = getattr(response, "text", None)
        if not text:
            return "لا يوجد جواب واضح في الملفات المرفوعة."

        return text.strip()
