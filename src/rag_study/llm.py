"""LLMs de geração. O SDK só é importado dentro de complete(), nunca na importação.

Temperatura 0 deixa a comparação entre pipelines menos sensível à amostragem.
A medição de custo e latência fica fora deste módulo.
"""

from rag_study.errors import ConfigError, ProviderError


class FakeLLM:
    """Grava o prompt e devolve um texto fixo. Usado em teste e no modo offline."""

    def __init__(self, reply: str = "resposta-fake") -> None:
        self.reply = reply
        self.prompts: list[str] = []

    def complete(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.reply


class GeminiLLM:
    def __init__(self, api_key: str, model: str) -> None:
        if not api_key.strip():
            raise ConfigError("GEMINI_API_KEY é obrigatória quando o provedor é gemini.")
        self._api_key = api_key
        self._model = model
        self._client = None

    def complete(self, prompt: str) -> str:
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

    def complete(self, prompt: str) -> str:
        client = self._client_or_create()
        response = client.chat.completions.create(
            model=self._model,
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
        )
        text = response.choices[0].message.content
        if not text:
            raise ProviderError("OpenAI devolveu uma resposta vazia.")
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
