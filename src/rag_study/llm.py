"""LLMs de geração. O SDK só é importado dentro de complete(), nunca na importação.

Temperatura 0 deixa a comparação entre pipelines menos sensível à amostragem.
`last_usage` guarda a contagem do provedor quando ela vem na resposta. Preço,
latência, cache e nova tentativa ficam no harness, não aqui.
"""

from rag_study.errors import ConfigError, ProviderError
from rag_study.usage import TokenUsage, read_gemini_usage, read_openai_usage


class FakeLLM:
    """Grava o prompt e devolve um texto fixo. Usado em teste e no modo offline."""

    def __init__(self, reply: str = "resposta-fake") -> None:
        self.reply = reply
        self.prompts: list[str] = []
        self.last_usage: TokenUsage | None = None

    def complete(self, prompt: str) -> str:
        self.prompts.append(prompt)
        self.last_usage = None
        return self.reply


class GeminiLLM:
    def __init__(self, api_key: str, model: str) -> None:
        if not api_key.strip():
            raise ConfigError("GEMINI_API_KEY é obrigatória quando o provedor é gemini.")
        self._api_key = api_key
        self._model = model
        self._client = None
        self.last_usage: TokenUsage | None = None

    def complete(self, prompt: str) -> str:
        self.last_usage = None
        client = self._client_or_create()
        from google.genai import types

        response = client.models.generate_content(
            model=self._model,
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0),
        )
        text = getattr(response, "text", None)
        if not text:
            raise ProviderError("Gemini devolveu uma resposta vazia.")
        self.last_usage = read_gemini_usage(response)
        return text

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


class OpenAILLM:
    def __init__(self, api_key: str, model: str) -> None:
        if not api_key.strip():
            raise ConfigError("OPENAI_API_KEY é obrigatória quando o provedor é openai.")
        self._api_key = api_key
        self._model = model
        self._client = None
        self.last_usage: TokenUsage | None = None

    def complete(self, prompt: str) -> str:
        self.last_usage = None
        client = self._client_or_create()
        response = client.chat.completions.create(
            model=self._model,
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
        )
        text = response.choices[0].message.content
        if not text:
            raise ProviderError("OpenAI devolveu uma resposta vazia.")
        self.last_usage = read_openai_usage(response)
        return text

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
