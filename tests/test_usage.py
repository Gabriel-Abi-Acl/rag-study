"""Custo sai da tabela. Sem linha do modelo o valor fica vazio, não zero."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from rag_study.usage import (
    ModelPrice,
    PriceError,
    TokenUsage,
    estimate_cost,
    estimate_tokens,
    load_price_table,
    read_gemini_usage,
    read_openai_usage,
    usage_from_provider,
)

ROOT_TABLE = Path(__file__).resolve().parents[1] / "data" / "eval" / "prices.v1.json"


def test_token_estimate_and_provider_usage():
    assert estimate_tokens("") == 0
    assert estimate_tokens("abcd") == 1
    assert estimate_tokens("abcde") == 2
    counted = TokenUsage(9, 4, 13, False)
    assert usage_from_provider(SimpleNamespace(last_usage=counted), "prompt", "texto") == counted
    estimated = usage_from_provider(SimpleNamespace(last_usage=None), "abcd", "abcd")
    assert estimated == TokenUsage(1, 1, 2, True)


def test_gemini_output_includes_thinking_tokens():
    with_total = SimpleNamespace(
        usage_metadata=SimpleNamespace(
            prompt_token_count=100,
            candidates_token_count=20,
            thoughts_token_count=30,
            total_token_count=150,
        )
    )
    usage = read_gemini_usage(with_total)
    assert usage == TokenUsage(100, 50, 150, False)
    without_total = SimpleNamespace(
        usage_metadata=SimpleNamespace(
            prompt_token_count=100,
            candidates_token_count=20,
            thoughts_token_count=30,
            total_token_count=None,
        )
    )
    assert read_gemini_usage(without_total) == TokenUsage(100, 50, 150, False)
    assert read_gemini_usage(SimpleNamespace(usage_metadata=None)) is None


def test_openai_usage_and_cost_formula():
    payload = SimpleNamespace(
        prompt_tokens=1_000_000,
        completion_tokens=2_000_000,
        total_tokens=3_000_000,
    )
    usage = read_openai_usage(SimpleNamespace(usage=payload))
    assert usage == TokenUsage(1_000_000, 2_000_000, 3_000_000, False)
    assert estimate_cost(usage, ModelPrice(0.30, 2.50)) == pytest.approx(5.30)
    assert estimate_cost(usage, None) is None
    assert read_openai_usage(SimpleNamespace(usage=None)) is None


def test_shipped_price_table_is_the_paid_list_and_rejects_a_bad_file(tmp_path):
    table = load_price_table(ROOT_TABLE)
    assert set(table) == {"gemini-2.5-flash", "gemini-2.5-flash-lite"}
    assert table["gemini-2.5-flash"] == ModelPrice(0.3, 2.5)
    assert table["gemini-2.5-flash-lite"] == ModelPrice(0.1, 0.4)
    missing = tmp_path / "prices.v1.json"
    with pytest.raises(PriceError, match="não encontrada"):
        load_price_table(missing)
    missing.write_text(
        '{"models": {"m": {"input_per_million_usd": -1, "output_per_million_usd": 1}}}',
        encoding="utf-8",
    )
    with pytest.raises(PriceError, match=">= 0"):
        load_price_table(missing)
