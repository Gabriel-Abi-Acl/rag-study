"""Cache em disco e nova tentativa. Um rerun não paga o mesmo prompt.

A chave é o sha256 do modelo e do prompt. 429 e RESOURCE_EXHAUSTED esperam
e tentam de novo. O intervalo entre chamadas segura o ritmo do tier gratuito.
"""

import hashlib
import json
import re
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

from rag_study.protocols import LLM
from rag_study.usage import TokenUsage, usage_from_provider

_KEY = re.compile(r"^[0-9a-f]{64}$")
T = TypeVar("T")


class CacheError(ValueError):
    """Arquivo de cache ilegível. Não é tratado como ausência."""


@dataclass(frozen=True)
class CachedCompletion:
    text: str
    usage: TokenUsage


class ResponseCache:
    def __init__(self, directory: Path) -> None:
        self.directory = directory

    def get(self, key: str) -> CachedCompletion | None:
        path = self._path(key)
        if not path.is_file():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise CacheError(f"cache ilegível: {path}") from exc
        if not isinstance(payload, dict) or not isinstance(payload.get("text"), str):
            raise CacheError(f"cache ilegível: {path}")
        return CachedCompletion(payload["text"], _usage_from_payload(payload, path))

    def put(self, key: str, text: str, usage: TokenUsage) -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "text": text,
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
            "total_tokens": usage.total_tokens,
            "estimated": usage.estimated,
        }
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        temporary.replace(path)

    def _path(self, key: str) -> Path:
        if not _KEY.fullmatch(key):
            raise CacheError("chave de cache inválida.")
        return self.directory / f"{key}.json"


class Pace:
    """Espaça o início das chamadas de rede. Acerto de cache não consome vaga."""

    def __init__(
        self,
        min_interval: float,
        sleep: Callable[[float], None],
        clock: Callable[[], float],
    ) -> None:
        if min_interval < 0:
            raise ValueError("intervalo precisa ser >= 0.")
        self.min_interval = min_interval
        self._sleep = sleep
        self._clock = clock
        self._next_allowed = 0.0

    def wait(self) -> None:
        if self.min_interval <= 0:
            return
        now = self._clock()
        delay = self._next_allowed - now
        if delay > 0:
            self._sleep(delay)
            now = self._clock()
        self._next_allowed = now + self.min_interval


class CachedLLM:
    """Envolve um LLM. Grava uso e marca se a resposta saiu do disco."""

    def __init__(
        self,
        inner: LLM,
        model: str,
        cache: ResponseCache | None,
        pace: Pace,
        *,
        max_attempts: int = 6,
        base_delay: float = 2.0,
        max_delay: float = 60.0,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if max_attempts < 1:
            raise ValueError("max_attempts precisa ser >= 1.")
        self.inner = inner
        self.model = model
        self.cache = cache
        self.pace = pace
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self._sleep = sleep
        self.last_usage: TokenUsage | None = None
        self.last_cached = False

    def complete(self, prompt: str) -> str:
        key = cache_key(self.model, prompt)
        if self.cache is not None:
            hit = self.cache.get(key)
            if hit is not None:
                self.last_cached = True
                self.last_usage = hit.usage
                return hit.text
        self.pace.wait()
        text = call_with_retry(
            lambda: self.inner.complete(prompt),
            attempts=self.max_attempts,
            base_delay=self.base_delay,
            max_delay=self.max_delay,
            sleep=self._sleep,
        )
        usage = usage_from_provider(self.inner, prompt, text)
        self.last_usage = usage
        self.last_cached = False
        if self.cache is not None:
            self.cache.put(key, text, usage)
        return text


def cache_key(model: str, prompt: str) -> str:
    digest = hashlib.sha256(f"{model}\n{prompt}".encode()).hexdigest()
    return digest


def call_with_retry(
    fn: Callable[[], T],
    *,
    attempts: int,
    base_delay: float,
    max_delay: float,
    sleep: Callable[[float], None],
) -> T:
    """Repete só erro de taxa ou indisponibilidade. O último erro sobe."""
    last: Exception | None = None
    for attempt in range(attempts):
        try:
            return fn()
        except Exception as exc:
            last = exc
            if not is_retryable(exc) or attempt == attempts - 1:
                raise
            delay = retry_after_seconds(exc)
            if delay is None:
                delay = min(max_delay, base_delay * (2**attempt))
            sleep(delay)
    if last is None:
        raise RuntimeError("nova tentativa sem chamada.")
    raise last


def is_retryable(exc: BaseException) -> bool:
    status = getattr(exc, "status_code", None)
    if status in (429, 500, 503):
        return True
    code = getattr(exc, "code", None)
    if code in (429, 500, 503):
        return True
    if isinstance(code, str) and code.upper() in {
        "RESOURCE_EXHAUSTED",
        "UNAVAILABLE",
        "DEADLINE_EXCEEDED",
    }:
        return True
    name = type(exc).__name__.lower()
    if "ratelimit" in name or "resourceexhausted" in name or "toomanyrequests" in name:
        return True
    message = str(exc).lower()
    markers = (
        "429",
        "resource_exhausted",
        "rate limit",
        "too many requests",
        "quota",
        "503",
        "unavailable",
    )
    return any(marker in message for marker in markers)


def retry_after_seconds(exc: BaseException) -> float | None:
    headers = _headers(exc)
    if headers is None:
        return None
    raw = headers.get("retry-after")
    if raw is None:
        raw = headers.get("Retry-After")
    if raw is None:
        return None
    try:
        delay = float(raw)
    except (TypeError, ValueError):
        return None
    if delay < 0:
        return None
    return delay


def _headers(exc: BaseException) -> Mapping[str, str] | None:
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None) if response is not None else None
    if headers is None:
        headers = getattr(exc, "headers", None)
    if headers is None:
        return None
    getter = getattr(headers, "get", None)
    if getter is None:
        return None
    return headers


def _usage_from_payload(payload: dict, path: Path) -> TokenUsage:
    try:
        prompt = int(payload["prompt_tokens"])
        completion = int(payload["completion_tokens"])
        total = int(payload["total_tokens"])
        estimated = payload["estimated"]
    except (KeyError, TypeError, ValueError) as exc:
        raise CacheError(f"cache ilegível: {path}") from exc
    if not isinstance(estimated, bool):
        raise CacheError(f"cache ilegível: {path}")
    return TokenUsage(prompt, completion, total, estimated)
