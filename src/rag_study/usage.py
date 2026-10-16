"""Tokens e custo estimado. A tabela é preço de lista, não a fatura do tier gratuito.

Sem contagem do provedor, a estimativa é um token a cada quatro caracteres.
Modelo ausente na tabela deixa o custo vazio em vez de inventar um preço.
"""

import json
from dataclasses import dataclass
from pathlib import Path


class PriceError(ValueError):
    """Tabela de preço ausente ou com número inválido."""


@dataclass(frozen=True)
class TokenUsage:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    estimated: bool


@dataclass(frozen=True)
class ModelPrice:
    input_per_million: float
    output_per_million: float


def estimate_tokens(text: str) -> int:
    """Aproxima tokens quando a API não devolve usage. Não é a contagem do modelo."""
    if text == "":
        return 0
    return max(1, (len(text) + 3) // 4)


def usage_from_provider(llm: object, prompt: str, completion: str) -> TokenUsage:
    """Usa `last_usage` se o cliente preencheu. Senão estima pelo texto."""
    raw = getattr(llm, "last_usage", None)
    if isinstance(raw, TokenUsage):
        return raw
    prompt_tokens = estimate_tokens(prompt)
    completion_tokens = estimate_tokens(completion)
    return TokenUsage(
        prompt_tokens,
        completion_tokens,
        prompt_tokens + completion_tokens,
        True,
    )


def add_usage(left: TokenUsage, right: TokenUsage) -> TokenUsage:
    return TokenUsage(
        left.prompt_tokens + right.prompt_tokens,
        left.completion_tokens + right.completion_tokens,
        left.total_tokens + right.total_tokens,
        left.estimated or right.estimated,
    )


def read_gemini_usage(response: object) -> TokenUsage | None:
    """Saída inclui thinking tokens. O total da API manda, quando existe."""
    meta = getattr(response, "usage_metadata", None)
    if meta is None:
        return None
    prompt = _count(meta, "prompt_token_count")
    candidates = _count(meta, "candidates_token_count")
    thoughts = _count(meta, "thoughts_token_count")
    total = _count(meta, "total_token_count")
    if prompt is None and candidates is None and thoughts is None and total is None:
        return None
    prompt_tokens = prompt or 0
    if total is not None and total >= prompt_tokens:
        completion = total - prompt_tokens
        total_tokens = total
    else:
        completion = (candidates or 0) + (thoughts or 0)
        total_tokens = prompt_tokens + completion
    return TokenUsage(prompt_tokens, completion, total_tokens, False)


def read_openai_usage(response: object) -> TokenUsage | None:
    usage = getattr(response, "usage", None)
    if usage is None:
        return None
    prompt = _count(usage, "prompt_tokens")
    completion = _count(usage, "completion_tokens")
    if prompt is None or completion is None:
        return None
    total = _count(usage, "total_tokens")
    total_tokens = prompt + completion if total is None else total
    return TokenUsage(prompt, completion, total_tokens, False)


def load_price_table(path: Path) -> dict[str, ModelPrice]:
    """Lê `models` do JSON. A nota do arquivo não entra no cálculo."""
    if not path.is_file():
        raise PriceError(f"tabela de preço não encontrada: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise PriceError(f"tabela de preço com JSON inválido: {path}") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("models"), dict):
        raise PriceError(f"tabela de preço precisa do objeto models: {path}")
    table: dict[str, ModelPrice] = {}
    for model, raw in payload["models"].items():
        if not isinstance(model, str) or not isinstance(raw, dict):
            raise PriceError(f"entrada de preço inválida: {model}")
        table[model] = ModelPrice(
            _price(raw, "input_per_million_usd", model),
            _price(raw, "output_per_million_usd", model),
        )
    return table


def estimate_cost(usage: TokenUsage, price: ModelPrice | None) -> float | None:
    """USD pela tabela. None se o modelo não tem preço: não se usa zero no lugar."""
    if price is None:
        return None
    return (
        usage.prompt_tokens * price.input_per_million
        + usage.completion_tokens * price.output_per_million
    ) / 1_000_000


def _count(obj: object, name: str) -> int | None:
    if isinstance(obj, dict):
        value = obj.get(name)
    else:
        value = getattr(obj, name, None)
    if value is None:
        return None
    return int(value)


def _price(raw: dict[str, object], field: str, model: str) -> float:
    value = raw.get(field)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise PriceError(f"{model}: {field} precisa ser um número.")
    number = float(value)
    if number < 0:
        raise PriceError(f"{model}: {field} precisa ser >= 0.")
    return number
