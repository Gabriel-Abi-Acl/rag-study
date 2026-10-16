"""O runner grava o JSONL offline. O corpus do gabarito é o que entra no índice.

A resposta fake não é um resultado do estudo. O teste trava o contrato do arquivo.
"""

from pathlib import Path

import pytest

from rag_study.cache import ResponseCache
from rag_study.config import load_settings
from rag_study.dataset import ABSTENTION_ANSWER, EvalDataset, EvalQuestion
from rag_study.errors import ConfigError
from rag_study.judge import JudgeScore
from rag_study.loader import load_corpus
from rag_study.run import main, run_evaluation
from rag_study.usage import ModelPrice, TokenUsage, estimate_cost, estimate_tokens

ROOT = Path(__file__).resolve().parents[1]


class Router:
    """A resposta depende da pergunta, não do trecho recuperado que o prompt também carrega."""

    def __init__(self) -> None:
        self.prompts: list[str] = []
        self.last_usage = None

    def complete(self, prompt: str) -> str:
        self.prompts.append(prompt)
        question = prompt.rsplit("Pergunta:", 1)[-1]
        if "PIB" in question:
            return "O PIB é de 9 milhões."
        if "CEP" in question:
            return "Não sei."
        if "população" in question:
            return "Cerca de 12.540 habitantes."
        if "prefeita" in question:
            return "A prefeita é Helena Duarte."
        return "Não sei."


class ScriptedJudge:
    def __init__(self) -> None:
        self.calls = 0

    def score(
        self,
        question: str,
        reference: str,
        answer: str,
        contexts: tuple[str, ...],
    ) -> JudgeScore:
        del question, reference, answer, contexts
        self.calls += 1
        return JudgeScore(1.0, 0.25, TokenUsage(8, 2, 10, False), "judge-test")


def _dataset(corpus: Path) -> EvalDataset:
    questions = (
        EvalQuestion(
            "q001",
            "Quem é a prefeita eleita?",
            "Helena Duarte",
            ("nota.txt",),
            "factual",
        ),
        EvalQuestion("q002", "Qual é a população estimada?", "12.450", ("nota.txt",), "numeric"),
        EvalQuestion("q003", "Qual é o PIB municipal?", ABSTENTION_ANSWER, (), "unanswerable"),
        EvalQuestion("q004", "Qual é o CEP?", ABSTENTION_ANSWER, (), "unanswerable"),
    )
    return EvalDataset(
        "1",
        corpus / "questions.v1.jsonl",
        corpus,
        questions,
        tuple(load_corpus(corpus)),
    )


def _settings(corpus: Path):
    return load_settings(
        {
            "LLM_PROVIDER": "fake",
            "EMBEDDING_PROVIDER": "fake",
            "CORPUS_PATH": str(corpus / "ausente"),
            "CHUNK_STRATEGY": "fixed",
            "CHUNK_SIZE": "500",
            "CHUNK_OVERLAP": "0",
            "TOP_K": "1",
            "RERANK_CANDIDATES": "1",
            "JUDGE_PROVIDER": "off",
            "REQUEST_INTERVAL_S": "0",
        }
    )


def _prices() -> dict[str, ModelPrice]:
    return {"fake": ModelPrice(1.0, 2.0), "judge-test": ModelPrice(3.0, 4.0)}


def test_run_records_metrics_cost_and_cache(tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "nota.txt").write_text(
        "A população estimada de Vila Aurora é de 12.450 habitantes.\n"
        "A prefeita eleita chama-se Helena Duarte.\n",
        encoding="utf-8",
    )
    dataset = _dataset(corpus)
    router = Router()
    judge = ScriptedJudge()
    cache_dir = tmp_path / "cache"
    out = tmp_path / "naive.jsonl"
    settings = _settings(corpus)
    with pytest.raises(ConfigError, match="pipeline inválida"):
        run_evaluation(dataset, "nope", out, settings, llm=router, min_interval=0)
    assert router.prompts == []

    rows = run_evaluation(
        dataset,
        "naive_rag",
        out,
        settings,
        llm=router,
        judge=judge,
        cache=ResponseCache(cache_dir),
        prices=_prices(),
        min_interval=0,
        max_attempts=2,
    )
    assert [row["id"] for row in rows] == ["q001", "q002", "q003", "q004"]
    assert rows[0]["answer"] == "A prefeita é Helena Duarte."
    assert rows[0]["correct_match"] == 1
    assert rows[0]["correct"] == 1
    assert rows[0]["retrieved_chunk_ids"] == ["nota.txt:0"]
    assert rows[0]["k"] == 1
    assert rows[0]["recall_at_k"] == 1
    assert rows[0]["hallucinated"] is None
    assert rows[0]["cached"] is False
    assert rows[0]["latency_ms"] >= 0
    assert rows[0]["model"] == "fake"
    assert rows[0]["tokens_estimated"] is True
    prompt_tokens = estimate_tokens(router.prompts[0])
    completion_tokens = estimate_tokens(rows[0]["answer"])
    assert rows[0]["prompt_tokens"] == prompt_tokens
    assert rows[0]["completion_tokens"] == completion_tokens
    expected_cost = estimate_cost(
        TokenUsage(prompt_tokens, completion_tokens, prompt_tokens + completion_tokens, True),
        ModelPrice(1, 2),
    )
    assert rows[0]["cost_usd"] == pytest.approx(expected_cost)
    assert rows[1]["answer"] == "Cerca de 12.540 habitantes."
    assert rows[1]["correct_match"] == 0
    assert rows[1]["correct_judge"] == 1
    assert rows[1]["correct"] == 1
    assert rows[1]["faithfulness"] == 0.25
    assert rows[1]["judge_cost_usd"] == pytest.approx((8 * 3 + 2 * 4) / 1_000_000)
    assert rows[2]["hallucinated"] is True
    assert rows[2]["abstained"] is False
    assert rows[2]["correct_judge"] == 1
    assert rows[2]["correct"] == 0
    assert rows[2]["recall_at_k"] is None
    assert rows[3]["answer"] == "Não sei."
    assert rows[3]["hallucinated"] is False
    assert rows[3]["abstained"] is True
    assert rows[3]["correct"] == 1
    assert out.read_text(encoding="utf-8").count("\n") == 4

    calls = len(router.prompts)
    second = run_evaluation(
        dataset,
        "naive_rag",
        tmp_path / "naive-2.jsonl",
        settings,
        llm=router,
        judge=judge,
        cache=ResponseCache(cache_dir),
        prices=_prices(),
        min_interval=0,
    )
    assert len(router.prompts) == calls
    assert all(row["cached"] is True for row in second)
    assert second[0]["prompt_tokens"] == rows[0]["prompt_tokens"]
    assert judge.calls == 8

    closed = run_evaluation(
        dataset,
        "no_context",
        tmp_path / "closed.jsonl",
        settings,
        llm=router,
        cache=ResponseCache(cache_dir),
        prices=_prices(),
        min_interval=0,
    )
    assert closed[0]["retrieved_chunk_ids"] == []
    assert closed[0]["recall_at_k"] == 0
    assert closed[2]["recall_at_k"] is None
    assert closed[0]["faithfulness"] is None
    assert closed[0]["cached"] is False


def test_cli_smoke_does_not_price_an_unknown_model(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("LLM_PROVIDER", "fake")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "fake")
    out = tmp_path / "smoke.jsonl"
    code = main(
        [
            "--pipeline",
            "no_context",
            "--out",
            str(out),
            "--judge",
            "off",
            "--interval",
            "0",
            "--limit",
            "1",
            "--cache",
            str(tmp_path / "cache"),
            "--prices",
            str(ROOT / "data" / "eval" / "prices.v1.json"),
        ]
    )
    assert code == 0
    captured = capsys.readouterr()
    assert f"gravado: {out}" in captured.out
    assert "perguntas: 1" in captured.out
    assert "acerto" not in captured.out
    line = out.read_text(encoding="utf-8").strip()
    assert '"id": "q001"' in line
    assert '"model": "fake"' in line
    assert '"cost_usd": null' in line
    assert '"retrieved_chunk_ids": []' in line

    second = tmp_path / "smoke-2.jsonl"
    assert (
        main(
            [
                "--pipeline",
                "no_context",
                "--out",
                str(second),
                "--judge",
                "off",
                "--interval",
                "0",
                "--limit",
                "1",
                "--cache",
                str(tmp_path / "cache"),
                "--prices",
                str(ROOT / "data" / "eval" / "prices.v1.json"),
            ]
        )
        == 0
    )
    assert '"cached": true' in second.read_text(encoding="utf-8")
    assert (
        main(
            [
                "--pipeline",
                "naive_rag",
                "--out",
                str(tmp_path / "bad.jsonl"),
                "--judge",
                "off",
                "--interval",
                "0",
                "--limit",
                "0",
                "--prices",
                str(tmp_path / "ausente.json"),
            ]
        )
        == 1
    )
    assert "tabela de preço" in capsys.readouterr().err
