"""Embeddings. HashingEmbedder existe para o teste e o demo offline.

Não é um baseline de qualidade: vetores de hash não entram na comparação
publicada. Gemini é chamado um texto por vez porque gemini-embedding-2
agrega uma lista num único vetor.
"""

import hashlib
import math
import re

from rag_study.errors import ConfigError, ProviderError

_TOKEN = re.compile(r"\w+", flags=re.UNICODE)


class HashingEmbedder:
    def __init__(self, dimension: int = 64) -> None:
        if dimension < 1:
            raise ValueError("dimension precisa ser >= 1.")
        self.dimension = dimension

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [_hash_embed(text, self.dimension) for text in texts]


class GeminiEmbedder:
    def __init__(self, api_key: str, model: str) -> None:
        if not api_key.strip():
            raise ConfigError("GEMINI_API_KEY é obrigatória quando o provedor é gemini.")
        self._api_key = api_key
        self._model = model
        self._client = None

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        client = self._client_or_create()
        vectors: list[list[float]] = []
        for text in texts:
            response = client.models.embed_content(model=self._model, contents=text)
            embeddings = getattr(response, "embeddings", None) or []
            if not embeddings or embeddings[0].values is None:
                raise ProviderError("Gemini devolveu embedding vazio.")
            vectors.append([float(value) for value in embeddings[0].values])
        return vectors

    def _client_or_create(self):
        if self._client is None:
            try:
                from google import genai
            except ImportError as exc:
                raise ConfigError(
                    "Pacote google-genai ausente. Instale com: uv sync --extra gemini"
                ) from exc
            self._client = genai.Client(api_key=self._api_key)
        return self._client


class OpenAIEmbedder:
    def __init__(self, api_key: str, model: str) -> None:
        if not api_key.strip():
            raise ConfigError("OPENAI_API_KEY é obrigatória quando o provedor é openai.")
        self._api_key = api_key
        self._model = model
        self._client = None

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        client = self._client_or_create()
        response = client.embeddings.create(model=self._model, input=texts)
        ordered = sorted(response.data, key=lambda item: item.index)
        if len(ordered) != len(texts):
            raise ProviderError("OpenAI devolveu uma quantidade inesperada de embeddings.")
        return [[float(value) for value in item.embedding] for item in ordered]

    def _client_or_create(self):
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise ConfigError(
                    "Pacote openai ausente. Instale com: uv sync --extra openai"
                ) from exc
            self._client = OpenAI(api_key=self._api_key)
        return self._client


def _hash_embed(text: str, dimension: int) -> list[float]:
    vector = [0.0] * dimension
    tokens = _TOKEN.findall(text.lower())
    for token in tokens:
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], "big") % dimension
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[index] += sign
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0.0:
        return vector
    return [value / norm for value in vector]
