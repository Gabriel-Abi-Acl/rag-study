"""Os clientes reais só montam a chamada HTTP. O teste substitui o SDK."""

import sys
import types

import pytest

from rag_study.embeddings import GeminiEmbedder, OpenAIEmbedder
from rag_study.errors import ProviderError
from rag_study.llm import GeminiLLM, OpenAILLM


def test_gemini_complete_and_embed_call_the_sdk_offline(monkeypatch):
    calls: dict = {}

    class Models:
        def generate_content(self, *, model, contents, config):
            calls["gen_model"] = model
            calls["contents"] = contents
            calls["temperature"] = config.temperature

            class Usage:
                prompt_token_count = 11
                candidates_token_count = 5
                thoughts_token_count = 0
                total_token_count = 16

            class Response:
                text = "ok-gemini"
                usage_metadata = Usage()

            return Response()

        def embed_content(self, *, model, contents):
            calls.setdefault("embed_models", []).append(model)
            calls.setdefault("embed_inputs", []).append(contents)

            class Embedding:
                values = [0.25, 0.5]

            class Response:
                embeddings = [Embedding()]

            return Response()

    class Client:
        def __init__(self, api_key: str) -> None:
            calls.setdefault("api_keys", []).append(api_key)
            self.models = Models()

    google = types.ModuleType("google")
    genai = types.ModuleType("google.genai")
    genai_types = types.ModuleType("google.genai.types")

    class GenerateContentConfig:
        def __init__(self, temperature: float) -> None:
            self.temperature = temperature

    genai.Client = Client
    genai_types.GenerateContentConfig = GenerateContentConfig
    google.genai = genai
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.genai", genai)
    monkeypatch.setitem(sys.modules, "google.genai.types", genai_types)

    llm = GeminiLLM(api_key="gemini-test", model="gemini-2.5-flash")
    assert llm.complete("ping") == "ok-gemini"
    assert llm.last_usage is not None
    assert llm.last_usage.prompt_tokens == 11
    assert llm.last_usage.completion_tokens == 5
    assert llm.last_usage.estimated is False
    vectors = GeminiEmbedder(api_key="gemini-test", model="gemini-embedding-001").embed(
        ["um", "dois"]
    )
    assert vectors == [[0.25, 0.5], [0.25, 0.5]]
    assert calls["gen_model"] == "gemini-2.5-flash"
    assert calls["contents"] == "ping"
    assert calls["temperature"] == 0
    assert calls["embed_inputs"] == ["um", "dois"]
    assert calls["api_keys"] == ["gemini-test", "gemini-test"]


def test_openai_complete_and_embed_keep_input_order(monkeypatch):
    class Message:
        content = "ok-openai"

    class Choice:
        message = Message()

    class Usage:
        prompt_tokens = 3
        completion_tokens = 4
        total_tokens = 7

    class ChatResponse:
        choices = [Choice()]
        usage = Usage()

    class EmbeddingRow:
        def __init__(self, index: int, embedding: list[float]) -> None:
            self.index = index
            self.embedding = embedding

    class EmbeddingResponse:
        data = [EmbeddingRow(1, [0.0, 1.0]), EmbeddingRow(0, [1.0, 0.0])]

    class Completions:
        def create(self, *, model, temperature, messages):
            assert model == "gpt-4o-mini"
            assert temperature == 0
            assert messages == [{"role": "user", "content": "ping"}]
            return ChatResponse()

    class Chat:
        completions = Completions()

    class Embeddings:
        def create(self, *, model, input):
            assert model == "text-embedding-3-small"
            assert input == ["a", "b"]
            return EmbeddingResponse()

    class Client:
        def __init__(self, api_key: str) -> None:
            assert api_key == "sk-test"
            self.chat = Chat()
            self.embeddings = Embeddings()

    openai = types.ModuleType("openai")
    openai.OpenAI = Client
    monkeypatch.setitem(sys.modules, "openai", openai)

    llm = OpenAILLM(api_key="sk-test", model="gpt-4o-mini")
    assert llm.complete("ping") == "ok-openai"
    assert llm.last_usage is not None
    assert llm.last_usage.prompt_tokens == 3
    assert llm.last_usage.completion_tokens == 4
    assert llm.last_usage.estimated is False
    vectors = OpenAIEmbedder(api_key="sk-test", model="text-embedding-3-small").embed(["a", "b"])
    assert vectors == [[1.0, 0.0], [0.0, 1.0]]


def test_empty_provider_text_raises():
    class Models:
        def generate_content(self, **kwargs):
            del kwargs

            class Response:
                text = ""

            return Response()

    class Client:
        def __init__(self, api_key: str) -> None:
            del api_key
            self.models = Models()

    google = types.ModuleType("google")
    genai = types.ModuleType("google.genai")
    genai_types = types.ModuleType("google.genai.types")
    genai_types.GenerateContentConfig = lambda temperature: None
    genai.Client = Client
    monkeypatch_modules = {
        "google": google,
        "google.genai": genai,
        "google.genai.types": genai_types,
    }
    for name, module in monkeypatch_modules.items():
        sys.modules[name] = module
    try:
        with pytest.raises(ProviderError):
            GeminiLLM(api_key="k", model="m").complete("ping")
    finally:
        for name in monkeypatch_modules:
            sys.modules.pop(name, None)
