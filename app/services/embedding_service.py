import numpy as np
from google import genai
from google.genai import types
from openai import OpenAI

from app.core.config import get_settings


class EmbeddingService:
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

    def embed_document(self, text: str) -> list[float]:
        return self._embed(text=text, task_type="RETRIEVAL_DOCUMENT")

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text=text, task_type="RETRIEVAL_QUERY")

    def _embed(self, text: str, task_type: str) -> list[float]:
        if self.provider == "openai":
            return self._embed_openai(text)

        return self._embed_gemini(text=text, task_type=task_type)

    def _embed_openai(self, text: str) -> list[float]:
        assert self.openai_client is not None

        response = self.openai_client.embeddings.create(
            model=self.settings.openai_embedding_model,
            input=text,
            dimensions=self.settings.embedding_dim,
            encoding_format="float",
        )

        return self._normalize(response.data[0].embedding)

    def _embed_gemini(self, text: str, task_type: str) -> list[float]:
        assert self.gemini_client is not None

        result = self.gemini_client.models.embed_content(
            model=self.settings.gemini_embedding_model,
            contents=text,
            config=types.EmbedContentConfig(
                task_type=task_type,
                output_dimensionality=self.settings.embedding_dim,
            ),
        )

        if not result.embeddings:
            raise RuntimeError("Gemini returned no embeddings")

        values = result.embeddings[0].values
        return self._normalize(values)

    @staticmethod
    def _normalize(values: list[float]) -> list[float]:
        vector = np.array(values, dtype=np.float32)
        norm = np.linalg.norm(vector)
        if norm == 0:
            return vector.tolist()
        return (vector / norm).tolist()


def to_pgvector(values: list[float]) -> str:
    return "[" + ",".join(str(float(value)) for value in values) + "]"
