"""Configuração lida de variáveis de ambiente. Chaves nunca são impressas.

`load_settings(env)` ignora o arquivo .env de propósito: o teste passa um
mapa e não pode vazar credencial do ambiente do desenvolvedor.
"""

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from dotenv import dotenv_values

from rag_study.errors import ConfigError

LLM_PROVIDERS = ("gemini", "openai", "fake")
EMBEDDING_PROVIDERS = ("gemini", "openai", "fake")
CHUNK_STRATEGIES = ("fixed", "recursive")

ENV_KEYS = (
    "LLM_PROVIDER",
    "EMBEDDING_PROVIDER",
    "GEMINI_API_KEY",
    "GEMINI_MODEL",
    "GEMINI_EMBEDDING_MODEL",
    "OPENAI_API_KEY",
    "OPENAI_MODEL",
    "OPENAI_EMBEDDING_MODEL",
    "CHUNK_STRATEGY",
    "CHUNK_SIZE",
    "CHUNK_OVERLAP",
    "TOP_K",
    "RERANK_CANDIDATES",
    "CORPUS_PATH",
)


@dataclass(frozen=True)
class Settings:
    llm_provider: str
    embedding_provider: str
    gemini_api_key: str
    gemini_model: str
    gemini_embedding_model: str
    openai_api_key: str
    openai_model: str
    openai_embedding_model: str
    chunk_strategy: str
    chunk_size: int
    chunk_overlap: int
    top_k: int
    rerank_candidates: int
    corpus_path: Path


def load_settings(
    env: Mapping[str, str] | None = None,
    *,
    dotenv_path: Path | None = None,
) -> Settings:
    """Monta Settings. Com `env` explícito, o .env não entra no resultado."""
    source = _resolve_source(env, dotenv_path)
    chunk_size = _integer(source, "CHUNK_SIZE", default=400, minimum=1)
    chunk_overlap = _integer(source, "CHUNK_OVERLAP", default=50, minimum=0)
    if chunk_overlap >= chunk_size:
        raise ConfigError("CHUNK_OVERLAP precisa ser menor que CHUNK_SIZE.")
    top_k = _integer(source, "TOP_K", default=3, minimum=1)
    rerank_candidates = _integer(source, "RERANK_CANDIDATES", default=8, minimum=1)
    if rerank_candidates < top_k:
        raise ConfigError("RERANK_CANDIDATES precisa ser >= TOP_K.")
    return Settings(
        llm_provider=_choice(source, "LLM_PROVIDER", "gemini", LLM_PROVIDERS),
        embedding_provider=_choice(source, "EMBEDDING_PROVIDER", "gemini", EMBEDDING_PROVIDERS),
        gemini_api_key=_text(source, "GEMINI_API_KEY", ""),
        gemini_model=_text(source, "GEMINI_MODEL", "gemini-2.5-flash"),
        gemini_embedding_model=_text(source, "GEMINI_EMBEDDING_MODEL", "gemini-embedding-001"),
        openai_api_key=_text(source, "OPENAI_API_KEY", ""),
        openai_model=_text(source, "OPENAI_MODEL", "gpt-4o-mini"),
        openai_embedding_model=_text(source, "OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
        chunk_strategy=_choice(source, "CHUNK_STRATEGY", "recursive", CHUNK_STRATEGIES),
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        top_k=top_k,
        rerank_candidates=rerank_candidates,
        corpus_path=Path(_text(source, "CORPUS_PATH", "data/sample")),
    )


def _resolve_source(
    env: Mapping[str, str] | None,
    dotenv_path: Path | None,
) -> Mapping[str, str]:
    if env is not None:
        return env
    merged = dict(os.environ)
    path = Path(".env") if dotenv_path is None else dotenv_path
    if path.is_file():
        # Valor já exportado no processo ganha do arquivo.
        for key, value in dotenv_values(path).items():
            if value is not None:
                merged.setdefault(key, value)
    return merged


def _text(source: Mapping[str, str], key: str, default: str) -> str:
    raw = source.get(key)
    if raw is None or str(raw).strip() == "":
        return default
    return str(raw).strip()


def _choice(
    source: Mapping[str, str],
    key: str,
    default: str,
    allowed: tuple[str, ...],
) -> str:
    value = _text(source, key, default).lower()
    if value not in allowed:
        options = ", ".join(allowed)
        raise ConfigError(f"{key} inválida: {value}. Use: {options}.")
    return value


def _integer(
    source: Mapping[str, str],
    key: str,
    *,
    default: int,
    minimum: int,
) -> int:
    raw = source.get(key)
    if raw is None or str(raw).strip() == "":
        value = default
    else:
        try:
            value = int(str(raw).strip())
        except ValueError as exc:
            raise ConfigError(f"{key} precisa ser um inteiro.") from exc
    if value < minimum:
        raise ConfigError(f"{key} precisa ser >= {minimum}.")
    return value
