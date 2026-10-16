"""Juiz de correção e de fidelidade. O contrato é JSON, não a biblioteca RAGAS.

RAGAS arrasta LangChain e não roda no LLM fake da CI. Os dois prompts
pedem o mesmo sinal: equivalência ao gabarito, e afirmações sustentadas
só pelo contexto recuperado.
"""

import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from rag_study.protocols import LLM
from rag_study.usage import TokenUsage, add_usage, usage_from_provider

CORRECTNESS_TASK = "Julgamento de correção"
FAITHFULNESS_TASK = "Julgamento de fidelidade"


@dataclass(frozen=True)
class JudgeScore:
    correct: float | None
    faithfulness: float | None
    usage: TokenUsage | None
    model: str


class Judge(Protocol):
    def score(
        self,
        question: str,
        reference: str,
        answer: str,
        contexts: Sequence[str],
    ) -> JudgeScore:
        """Correção contra o gabarito e fidelidade ao contexto recuperado."""


class LLMJudge:
    """Duas chamadas por pergunta. Correção não vê o contexto. Fidelidade não vê o gabarito."""

    def __init__(self, llm: LLM, model: str) -> None:
        self._llm = llm
        self.model = model

    def score(
        self,
        question: str,
        reference: str,
        answer: str,
        contexts: Sequence[str],
    ) -> JudgeScore:
        correct_prompt = correctness_prompt(question, reference, answer)
        correct_text = self._llm.complete(correct_prompt)
        correct_usage = usage_from_provider(self._llm, correct_prompt, correct_text)
        faith_prompt = faithfulness_prompt(question, answer, contexts)
        faith_text = self._llm.complete(faith_prompt)
        faith_usage = usage_from_provider(self._llm, faith_prompt, faith_text)
        return JudgeScore(
            correct=parse_correctness(correct_text),
            faithfulness=parse_faithfulness(faith_text),
            usage=add_usage(correct_usage, faith_usage),
            model=self.model,
        )


def correctness_prompt(question: str, reference: str, answer: str) -> str:
    return (
        f"{CORRECTNESS_TASK}\n"
        "Julgue se a resposta equivale ao gabarito. Paráfrase vale. "
        "Não use conhecimento de fora do gabarito. "
        'Responda apenas JSON: {"correct": true} ou {"correct": false}.\n\n'
        f"Pergunta: {question}\n"
        f"Gabarito: {reference}\n"
        f"Resposta: {answer}"
    )


def faithfulness_prompt(question: str, answer: str, contexts: Sequence[str]) -> str:
    if contexts:
        blocks = [f"[{index}] {text}" for index, text in enumerate(contexts, start=1)]
        context = "\n\n".join(blocks)
    else:
        context = "(nenhum trecho recuperado)"
    return (
        f"{FAITHFULNESS_TASK}\n"
        "Liste as afirmações atômicas da resposta. Uma abstenção sem fato extra "
        "não gera afirmação. supported é true só se o contexto implica a afirmação. "
        "Contexto vazio não sustenta fato. "
        "faithful é a fração supported. Sem afirmações, faithful é 1. "
        'Responda apenas JSON: {"claims": [{"claim": "...", "supported": true}], '
        '"faithful": 1.0}.\n\n'
        f"Contexto:\n{context}\n\n"
        f"Pergunta: {question}\n"
        f"Resposta: {answer}"
    )


def parse_correctness(text: str) -> float | None:
    payload = extract_json_object(text)
    if payload is None:
        return None
    if "correct" in payload:
        return as_unit(payload["correct"])
    if "score" in payload:
        return as_unit(payload["score"])
    return None


def parse_faithfulness(text: str) -> float | None:
    """Preferência às claims. O campo faithful cobre a resposta sem lista."""
    payload = extract_json_object(text)
    if payload is None:
        return None
    claims = payload.get("claims")
    if isinstance(claims, list):
        score = _claims_score(claims)
        if score is not None:
            return score
    for key in ("faithful", "faithfulness"):
        if key in payload:
            return as_unit(payload[key])
    return None


def extract_json_object(text: str) -> dict[str, object] | None:
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        payload = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def as_unit(value: object) -> float | None:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, int | float):
        number = float(value)
        if 0.0 <= number <= 1.0:
            return number
    return None


def _claims_score(claims: list[object]) -> float | None:
    if not claims:
        return 1.0
    scores: list[float] = []
    for claim in claims:
        if not isinstance(claim, dict) or "supported" not in claim:
            return None
        unit = as_unit(claim["supported"])
        if unit is None:
            return None
        scores.append(unit)
    return sum(scores) / len(scores)
