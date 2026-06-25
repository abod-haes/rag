import numpy as np
from google import genai
from google.genai import types

from app.core.config import get_settings


class EmbeddingService:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.client = genai.Client(api_key=self.settings.gemini_api_key)

    def embed_document(self, text: str) -> list[float]:
        return self._embed(text=text, task_type="RETRIEVAL_DOCUMENT")

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text=text, task_type="RETRIEVAL_QUERY")

    def _embed(self, text: str, task_type: str) -> list[float]:
        result = self.client.models.embed_content(
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
