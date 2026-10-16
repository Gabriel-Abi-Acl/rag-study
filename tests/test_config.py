"""Configuração por ambiente, chaves do .env.example e fábricas sem rede."""

from pathlib import Path

import pytest

from rag_study.config import ENV_KEYS, load_settings
from rag_study.embeddings import GeminiEmbedder, HashingEmbedder, OpenAIEmbedder
from rag_study.errors import ConfigError
from rag_study.factory import build_embedder, build_llm, build_pipelines
from rag_study.llm import FakeLLM, GeminiLLM, OpenAILLM

ROOT = Path(__file__).resolve().parents[1]


def test_defaults_point_at_gemini_and_recursive_chunks():
    settings = load_settings({})
    assert settings.llm_provider == "gemini"
    assert settings.embedding_provider == "gemini"
    assert settings.gemini_model == "gemini-2.5-flash"
    assert settings.chunk_strategy == "recursive"
    assert settings.chunk_size == 400
    assert settings.chunk_overlap == 50
    assert settings.top_k == 3
    assert settings.rerank_candidates == 8
    assert settings.corpus_path == Path("data/sample")
    assert settings.gemini_api_key == ""


def test_env_overrides_are_parsed():
    settings = load_settings(
        {
            "LLM_PROVIDER": "OpenAI",
            "EMBEDDING_PROVIDER": "fake",
            "OPENAI_API_KEY": "sk-test",
            "CHUNK_STRATEGY": "fixed",
            "CHUNK_SIZE": "128",
            "CHUNK_OVERLAP": "16",
            "TOP_K": "2",
            "RERANK_CANDIDATES": "5",
            "CORPUS_PATH": "data/other",
        }
    )
    assert settings.llm_provider == "openai"
    assert settings.embedding_provider == "fake"
    assert settings.openai_api_key == "sk-test"
    assert settings.chunk_strategy == "fixed"
    assert settings.chunk_size == 128
    assert settings.corpus_path == Path("data/other")


def test_invalid_values_raise_before_any_provider_call():
    with pytest.raises(ConfigError):
        load_settings({"LLM_PROVIDER": "anthropic"})
    with pytest.raises(ConfigError):
        load_settings({"CHUNK_SIZE": "abc"})
    with pytest.raises(ConfigError):
        load_settings({"CHUNK_SIZE": "20", "CHUNK_OVERLAP": "20"})
    with pytest.raises(ConfigError):
        load_settings({"TOP_K": "4", "RERANK_CANDIDATES": "2"})


def test_explicit_env_ignores_dotenv_file(tmp_path: Path):
    dotenv = tmp_path / ".env"
    dotenv.write_text("LLM_PROVIDER=openai\nGEMINI_API_KEY=segredo\n", encoding="utf-8")
    settings = load_settings({"LLM_PROVIDER": "fake"}, dotenv_path=dotenv)
    assert settings.llm_provider == "fake"
    assert settings.gemini_api_key == ""


def test_dotenv_fills_missing_keys_without_overriding_process_env(tmp_path: Path, monkeypatch):
    for key in ("LLM_PROVIDER", "EMBEDDING_PROVIDER", "GEMINI_MODEL"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("EMBEDDING_PROVIDER", "fake")
    dotenv = tmp_path / ".env"
    dotenv.write_text(
        "LLM_PROVIDER=fake\nEMBEDDING_PROVIDER=openai\nGEMINI_MODEL=custom\n",
        encoding="utf-8",
    )
    settings = load_settings(dotenv_path=dotenv)
    assert settings.llm_provider == "fake"
    assert settings.embedding_provider == "fake"
    assert settings.gemini_model == "custom"


def test_env_example_covers_the_known_keys():
    keys = []
    for line in (ROOT / ".env.example").read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        keys.append(stripped.split("=", 1)[0])
    assert keys == list(ENV_KEYS)


def test_factories_build_fakes_and_refuse_missing_keys():
    fake = load_settings({"LLM_PROVIDER": "fake", "EMBEDDING_PROVIDER": "fake"})
    assert isinstance(build_llm(fake), FakeLLM)
    assert isinstance(build_embedder(fake), HashingEmbedder)

    gemini = load_settings({"LLM_PROVIDER": "gemini", "EMBEDDING_PROVIDER": "gemini"})
    with pytest.raises(ConfigError):
        build_llm(gemini)
    with pytest.raises(ConfigError):
        build_embedder(gemini)

    openai = load_settings(
        {
            "LLM_PROVIDER": "openai",
            "EMBEDDING_PROVIDER": "openai",
            "OPENAI_API_KEY": "sk-test",
        }
    )
    assert isinstance(build_llm(openai), OpenAILLM)
    assert isinstance(build_embedder(openai), OpenAIEmbedder)
    keyed = load_settings(
        {
            "LLM_PROVIDER": "gemini",
            "EMBEDDING_PROVIDER": "gemini",
            "GEMINI_API_KEY": "test-key",
        }
    )
    assert isinstance(build_llm(keyed), GeminiLLM)
    assert isinstance(build_embedder(keyed), GeminiEmbedder)


def test_sample_corpus_indexes_offline(tmp_path: Path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "notas.txt").write_text(
        "A população de Vila Aurora é de 12450 habitantes.\n",
        encoding="utf-8",
    )
    (corpus / "ignorado.bin").write_bytes(b"\x00\x01")
    settings = load_settings(
        {
            "LLM_PROVIDER": "fake",
            "EMBEDDING_PROVIDER": "fake",
            "CORPUS_PATH": str(corpus),
            "CHUNK_STRATEGY": "fixed",
            "CHUNK_SIZE": "80",
            "CHUNK_OVERLAP": "0",
            "TOP_K": "1",
            "RERANK_CANDIDATES": "1",
        }
    )
    pipelines = build_pipelines(settings)
    answer = pipelines["naive_rag"].answer("população")
    assert answer.pipeline == "naive_rag"
    assert answer.contexts
    assert "12450" in answer.contexts[0]
    assert pipelines["no_context"].answer("população").contexts == ()


def test_checked_in_sample_loads():
    settings = load_settings(
        {
            "LLM_PROVIDER": "fake",
            "EMBEDDING_PROVIDER": "fake",
            "CORPUS_PATH": str(ROOT / "data" / "sample"),
        }
    )
    pipelines = build_pipelines(settings)
    answer = pipelines["rag_rerank"].answer("Qual é a população de Vila Aurora?")
    assert answer.pipeline == "rag_rerank"
    assert any("12.450" in context for context in answer.contexts)
