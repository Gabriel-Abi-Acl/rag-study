"""O juiz pede JSON. Fidelidade não recebe o gabarito. Correção não recebe o contexto."""

import pytest

from rag_study.judge import (
    FAITHFULNESS_TASK,
    LLMJudge,
    parse_correctness,
    parse_faithfulness,
)
from rag_study.usage import TokenUsage


class CaptureLLM:
    def __init__(self) -> None:
        self.prompts: list[str] = []
        self.last_usage: TokenUsage | None = None

    def complete(self, prompt: str) -> str:
        self.prompts.append(prompt)
        self.last_usage = TokenUsage(3, 1, 4, True)
        if FAITHFULNESS_TASK in prompt:
            return (
                '{"claims": [{"claim": "a", "supported": true}, '
                '{"claim": "b", "supported": false}]}'
            )
        return '{"correct": true}'


def test_parser_reads_fenced_json_claims_and_rejects_out_of_range_scores():
    assert parse_correctness('```json\n{"correct": true}\n```') == 1
    assert parse_correctness('{"score": 0}') == 0
    assert parse_correctness('{"score": 2}') is None
    assert parse_correctness("não sei") is None
    assert parse_faithfulness(
        '{"claims": [{"claim": "a", "supported": true}, {"claim": "b", "supported": false}]}'
    ) == pytest.approx(0.5)
    assert parse_faithfulness('{"claims": []}') == 1
    assert parse_faithfulness('{"claims": [{"claim": "a"}], "faithful": 0.25}') == 0.25


def test_judge_splits_correctness_from_faithfulness_and_sums_usage():
    llm = CaptureLLM()
    score = LLMJudge(llm, "gemini-2.5-flash").score(
        "Q?",
        "GABARITO-SECRETO",
        "resposta",
        ["CTX-UNICO"],
    )
    assert score.correct == 1
    assert score.faithfulness == 0.5
    assert score.model == "gemini-2.5-flash"
    assert score.usage == TokenUsage(6, 2, 8, True)
    assert "GABARITO-SECRETO" in llm.prompts[0]
    assert "CTX-UNICO" not in llm.prompts[0]
    assert "GABARITO-SECRETO" not in llm.prompts[1]
    assert "CTX-UNICO" in llm.prompts[1]
    assert "resposta" in llm.prompts[1]
