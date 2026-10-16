"""A tabela agrega o JSONL recebido. Célula vazia é métrica indefinida, não zero."""

import pytest

from rag_study.aggregate import (
    AggregateError,
    load_rows,
    main,
    render_csv,
    render_markdown,
    summarize,
)


def _row(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "schema": "eval-run.v1",
        "id": "q001",
        "pipeline": "naive_rag",
        "type": "factual",
        "correct": 1.0,
        "faithfulness": None,
        "hallucinated": None,
        "recall_at_k": 1.0,
        "latency_ms": 10.0,
        "cached": False,
        "prompt_tokens": 10,
        "completion_tokens": 4,
        "cost_usd": 1.0,
        "judge_prompt_tokens": None,
        "judge_completion_tokens": None,
        "judge_cost_usd": None,
    }
    base.update(overrides)
    return base


def _sample() -> list[dict[str, object]]:
    return [
        _row(
            id="q001",
            faithfulness=1.0,
            judge_prompt_tokens=1,
            judge_completion_tokens=1,
            judge_cost_usd=0.5,
        ),
        _row(
            id="q002",
            correct=0.0,
            recall_at_k=0.5,
            latency_ms=100.0,
            cached=True,
            prompt_tokens=6,
            completion_tokens=2,
            cost_usd=2.0,
        ),
        _row(
            id="q003",
            type="unanswerable",
            correct=0.0,
            faithfulness=1.0,
            hallucinated=True,
            recall_at_k=None,
            latency_ms=20.0,
            prompt_tokens=3,
            completion_tokens=1,
            cost_usd=None,
        ),
        _row(
            id="q001",
            pipeline="no_context",
            recall_at_k=0.0,
            latency_ms=5.0,
            prompt_tokens=4,
            completion_tokens=1,
            cost_usd=None,
        ),
    ]


def _find(rows, pipeline: str, question_type: str):
    found = [row for row in rows if row.pipeline == pipeline and row.type == question_type]
    assert len(found) == 1
    return found[0]


def test_summary_math_per_pipeline_and_type():
    rows = summarize(_sample())
    factual = _find(rows, "naive_rag", "factual")
    assert factual.n == 2
    assert factual.accuracy == pytest.approx(0.5)
    assert factual.faithfulness == pytest.approx(1)
    assert factual.hallucination_rate is None
    assert factual.recall_at_k == pytest.approx(0.75)
    assert factual.latency_ms_mean == pytest.approx(10)
    assert factual.prompt_tokens == 17
    assert factual.completion_tokens == 7
    assert factual.cost_usd == pytest.approx(3.5)
    assert factual.n_cached == 1

    missing = _find(rows, "naive_rag", "unanswerable")
    assert missing.hallucination_rate == pytest.approx(1)
    assert missing.recall_at_k is None
    assert missing.cost_usd is None
    assert missing.accuracy == 0

    overall = _find(rows, "naive_rag", "all")
    assert overall.n == 3
    assert overall.accuracy == pytest.approx(1 / 3)
    assert overall.faithfulness == pytest.approx(1)
    assert overall.hallucination_rate == pytest.approx(1)
    assert overall.recall_at_k == pytest.approx(0.75)
    assert overall.latency_ms_mean == pytest.approx(15)
    assert overall.prompt_tokens == 20
    assert overall.completion_tokens == 8
    assert overall.cost_usd == pytest.approx(3.5)
    assert overall.n_cached == 1

    closed = _find(rows, "no_context", "factual")
    assert closed.recall_at_k == 0
    assert closed.latency_ms_mean == pytest.approx(5)
    assert [row.pipeline for row in rows if row.type == "all"] == ["no_context", "naive_rag"]


def test_cached_only_rows_leave_latency_empty():
    summary = summarize([_row(cached=True, latency_ms=999, faithfulness=None, cost_usd=None)])
    assert summary[0].latency_ms_mean is None
    assert summary[0].faithfulness is None
    assert summary[0].cost_usd is None
    csv_text = render_csv(summary)
    assert "latencia_ms" in csv_text.splitlines()[0]
    assert ",," in csv_text


def test_markdown_and_csv_render_the_sample_without_filling_blanks():
    rendered = summarize(_sample())
    markdown = render_markdown(rendered)
    csv_text = render_csv(rendered)
    assert markdown.startswith("Agregação dos arquivos informados.")
    assert "resultado publicado" in markdown
    factual = (
        "| naive_rag | factual | 2 | 0.5000 | 1.0000 |  | 0.7500 | 10.00 | 17 | 7 | 3.500000 | 1 |"
    )
    assert factual in markdown
    assert "naive_rag,factual,2,0.5000,1.0000,,0.7500,10.00,17,7,3.500000,1" in csv_text
    assert "0.3333" in markdown


def test_duplicate_question_unknown_schema_and_cli_files(tmp_path, capsys):
    with pytest.raises(AggregateError, match="pergunta repetida"):
        summarize([_row(id="q001"), _row(id="q001")])
    with pytest.raises(AggregateError, match="schema desconhecido"):
        summarize([_row(schema="outro")])
    with pytest.raises(AggregateError, match="campo ausente"):
        summarize([{"id": "q001"}])

    path = tmp_path / "run.jsonl"
    path.write_text(
        '{"schema": "eval-run.v1", "id": "q001", "pipeline": "naive_rag", "type": "factual", '
        '"correct": 1, "faithfulness": null, "hallucinated": null, "recall_at_k": 1, '
        '"latency_ms": 10, "cached": false, "prompt_tokens": 2, "completion_tokens": 1, '
        '"cost_usd": null, "judge_prompt_tokens": null, "judge_completion_tokens": null, '
        '"judge_cost_usd": null}\n',
        encoding="utf-8",
    )
    assert load_rows([path])[0]["id"] == "q001"
    markdown_path = tmp_path / "summary.md"
    csv_path = tmp_path / "summary.csv"
    assert main([str(path), "--md", str(markdown_path), "--csv", str(csv_path)]) == 0
    assert "naive_rag" in markdown_path.read_text(encoding="utf-8")
    assert "acerto" in csv_path.read_text(encoding="utf-8")
    captured = capsys.readouterr()
    assert "Agregação dos arquivos informados." in captured.out
    assert main([str(tmp_path / "ausente.jsonl")]) == 1
    assert "não encontrado" in capsys.readouterr().err
