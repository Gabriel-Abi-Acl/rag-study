"""Cache e retry. Acerto de disco não chama o provedor. 429 espera e tenta de novo."""

from types import SimpleNamespace

import pytest

from rag_study.cache import CachedLLM, CacheError, Pace, ResponseCache, cache_key, is_retryable
from rag_study.errors import ProviderError
from rag_study.usage import TokenUsage


class Flaky:
    def __init__(self, failures: int, error: Exception) -> None:
        self.failures = failures
        self.error = error
        self.calls = 0
        self.last_usage = TokenUsage(4, 2, 6, False)

    def complete(self, prompt: str) -> str:
        del prompt
        self.calls += 1
        if self.calls <= self.failures:
            raise self.error
        return "pronta"


def test_rate_limit_retries_then_caches_the_success(tmp_path):
    delays: list[float] = []
    inner = Flaky(2, RuntimeError("429 RESOURCE_EXHAUSTED"))
    cache = ResponseCache(tmp_path / "cache")
    client = CachedLLM(
        inner,
        model="gemini-2.5-flash",
        cache=cache,
        pace=Pace(0, delays.append, lambda: 0),
        max_attempts=4,
        base_delay=2,
        max_delay=60,
        sleep=delays.append,
    )
    assert client.complete("oi") == "pronta"
    assert delays == [2, 4]
    assert inner.calls == 3
    assert client.last_cached is False
    assert client.complete("oi") == "pronta"
    assert inner.calls == 3
    assert client.last_cached is True
    assert client.last_usage == TokenUsage(4, 2, 6, False)


def test_retry_after_and_non_retryable_errors(tmp_path):
    delays: list[float] = []

    class RateLimit(RuntimeError):
        def __init__(self) -> None:
            super().__init__("slow down")
            self.status_code = 429
            self.response = SimpleNamespace(headers={"Retry-After": "7"})

    inner = Flaky(1, RateLimit())
    client = CachedLLM(
        inner,
        model="m",
        cache=None,
        pace=Pace(0, delays.append, lambda: 0),
        max_attempts=3,
        base_delay=2,
        sleep=delays.append,
    )
    assert client.complete("oi") == "pronta"
    assert delays == [7]

    fatal = Flaky(5, ProviderError("Gemini devolveu uma resposta vazia."))
    client.inner = fatal
    with pytest.raises(ProviderError):
        client.complete("outro")
    assert fatal.calls == 1
    assert is_retryable(RuntimeError("chave invalida")) is False
    assert is_retryable(RuntimeError("503 unavailable")) is True
    assert is_retryable(RuntimeError("bad request")) is False


def test_corrupt_cache_is_not_treated_as_a_miss(tmp_path):
    cache = ResponseCache(tmp_path / "cache")
    cache.put(cache_key("m", "oi"), "pronta", TokenUsage(1, 1, 2, True))
    path = cache.directory / f"{cache_key('m', 'oi')}.json"
    path.write_text("{", encoding="utf-8")
    with pytest.raises(CacheError, match="ilegível"):
        cache.get(cache_key("m", "oi"))


def test_pace_spaces_network_calls_and_not_cache_hits(tmp_path):
    now = {"t": 1000.0}
    delays: list[float] = []

    def clock() -> float:
        return now["t"]

    def sleep(seconds: float) -> None:
        delays.append(seconds)
        now["t"] += seconds

    class Once:
        def __init__(self) -> None:
            self.calls = 0
            self.last_usage = None

        def complete(self, prompt: str) -> str:
            del prompt
            self.calls += 1
            return "ok"

    inner = Once()
    cache = ResponseCache(tmp_path / "cache")
    client = CachedLLM(
        inner,
        model="m",
        cache=cache,
        pace=Pace(5, sleep, clock),
        max_attempts=1,
        sleep=sleep,
    )
    assert client.complete("a") == "ok"
    assert client.complete("b") == "ok"
    assert delays == [5]
    assert client.complete("a") == "ok"
    assert inner.calls == 2
    assert delays == [5]
