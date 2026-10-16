"""Agrega JSONL de execuções. A tabela descreve os arquivos recebidos, não o estudo.

Célula vazia significa métrica indefinida naquele recorte. Custo nulo não
vira zero: a soma só existe quando alguma linha tem preço.
"""

import argparse
import csv
import io
import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from rag_study.dataset import QUESTION_TYPES
from rag_study.pipelines import PIPELINE_ORDER
from rag_study.run import RUN_SCHEMA

_REQUIRED = (
    "id",
    "pipeline",
    "type",
    "correct",
    "faithfulness",
    "hallucinated",
    "recall_at_k",
    "latency_ms",
    "cached",
    "prompt_tokens",
    "completion_tokens",
    "cost_usd",
    "judge_prompt_tokens",
    "judge_completion_tokens",
    "judge_cost_usd",
)


class AggregateError(ValueError):
    """JSONL fora do contrato. A agregação não completa lacuna com chute."""


@dataclass(frozen=True)
class SummaryRow:
    pipeline: str
    type: str
    n: int
    accuracy: float | None
    faithfulness: float | None
    hallucination_rate: float | None
    recall_at_k: float | None
    latency_ms_mean: float | None
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float | None
    n_cached: int


def load_rows(paths: Sequence[Path]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in paths:
        if not path.is_file():
            raise AggregateError(f"arquivo não encontrado: {path}")
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise AggregateError(f"{path}:{number}: JSON inválido") from exc
            if not isinstance(payload, dict):
                raise AggregateError(f"{path}:{number}: cada linha precisa ser um objeto")
            rows.append(payload)
    if not rows:
        raise AggregateError("nenhuma linha para agregar")
    return rows


def summarize(rows: Sequence[Mapping[str, object]]) -> list[SummaryRow]:
    """Uma linha por pipeline e tipo presente, mais o total `all` da pipeline."""
    _validate(rows)
    type_order = [name for name in QUESTION_TYPES if any(row["type"] == name for row in rows)]
    extras = sorted({str(row["type"]) for row in rows} - set(QUESTION_TYPES))
    type_order.extend(extras)
    summaries: list[SummaryRow] = []
    for pipeline in _pipeline_order(rows):
        owned = [row for row in rows if row["pipeline"] == pipeline]
        for name in type_order:
            subset = [row for row in owned if row["type"] == name]
            if subset:
                summaries.append(_group(pipeline, name, subset))
        summaries.append(_group(pipeline, "all", owned))
    return summaries


def render_markdown(rows: Sequence[SummaryRow]) -> str:
    header = (
        "| pipeline | tipo | n | acerto | fidelidade | alucinacao | recall_at_k "
        "| latencia_ms | tokens_prompt | tokens_saida | custo_usd | n_cache |"
    )
    separator = "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"
    lines = [
        "Agregação dos arquivos informados. Não é um resultado publicado do estudo.",
        "",
        header,
        separator,
    ]
    lines.extend(_markdown_line(row) for row in rows)
    return "\n".join(lines) + "\n"


def render_csv(rows: Sequence[SummaryRow]) -> str:
    fieldnames = [
        "pipeline",
        "tipo",
        "n",
        "acerto",
        "fidelidade",
        "alucinacao",
        "recall_at_k",
        "latencia_ms",
        "tokens_prompt",
        "tokens_saida",
        "custo_usd",
        "n_cache",
    ]
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow(
            {
                "pipeline": row.pipeline,
                "tipo": row.type,
                "n": row.n,
                "acerto": _fmt_rate(row.accuracy),
                "fidelidade": _fmt_rate(row.faithfulness),
                "alucinacao": _fmt_rate(row.hallucination_rate),
                "recall_at_k": _fmt_rate(row.recall_at_k),
                "latencia_ms": _fmt_latency(row.latency_ms_mean),
                "tokens_prompt": row.prompt_tokens,
                "tokens_saida": row.completion_tokens,
                "custo_usd": _fmt_cost(row.cost_usd),
                "n_cache": row.n_cached,
            }
        )
    return buffer.getvalue()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Agrega JSONL de avaliação em tabela markdown e CSV. Não publica o estudo.",
    )
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--md", type=Path, default=None)
    parser.add_argument("--csv", type=Path, default=None)
    args = parser.parse_args(argv)
    try:
        summaries = summarize(load_rows(args.inputs))
    except AggregateError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    markdown = render_markdown(summaries)
    csv_text = render_csv(summaries)
    if args.md is not None:
        args.md.parent.mkdir(parents=True, exist_ok=True)
        args.md.write_text(markdown, encoding="utf-8")
    if args.csv is not None:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        args.csv.write_text(csv_text, encoding="utf-8")
    print(markdown, end="")
    return 0


def _validate(rows: Sequence[Mapping[str, object]]) -> None:
    seen: set[tuple[object, object]] = set()
    for index, row in enumerate(rows, start=1):
        missing = [field for field in _REQUIRED if field not in row]
        if missing:
            names = ", ".join(missing)
            raise AggregateError(f"linha {index}: campo ausente: {names}")
        schema = row.get("schema")
        if schema not in (None, RUN_SCHEMA):
            raise AggregateError(f"linha {index}: schema desconhecido: {schema}")
        key = (row["pipeline"], row["id"])
        if key in seen:
            raise AggregateError(f"pergunta repetida: {key[0]} {key[1]}")
        seen.add(key)


def _pipeline_order(rows: Sequence[Mapping[str, object]]) -> list[str]:
    present = {str(row["pipeline"]) for row in rows}
    ordered = [name for name in PIPELINE_ORDER if name in present]
    ordered.extend(sorted(present - set(PIPELINE_ORDER)))
    return ordered


def _group(pipeline: str, question_type: str, rows: Sequence[Mapping[str, object]]) -> SummaryRow:
    costs = _numbers(rows, "cost_usd") + _numbers(rows, "judge_cost_usd")
    latencies = [
        float(row["latency_ms"])
        for row in rows
        if row["cached"] is False and isinstance(row["latency_ms"], int | float)
    ]
    return SummaryRow(
        pipeline=pipeline,
        type=question_type,
        n=len(rows),
        accuracy=_mean(_numbers(rows, "correct")),
        faithfulness=_mean(_numbers(rows, "faithfulness")),
        hallucination_rate=_mean([1.0 if flag else 0.0 for flag in _flags(rows)]),
        recall_at_k=_mean(_numbers(rows, "recall_at_k")),
        latency_ms_mean=_mean(latencies),
        prompt_tokens=_token_sum(rows, "prompt_tokens", "judge_prompt_tokens"),
        completion_tokens=_token_sum(rows, "completion_tokens", "judge_completion_tokens"),
        cost_usd=None if not costs else sum(costs),
        n_cached=sum(1 for row in rows if row["cached"] is True),
    )


def _numbers(rows: Sequence[Mapping[str, object]], field: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = row[field]
        if isinstance(value, bool) or not isinstance(value, int | float):
            continue
        values.append(float(value))
    return values


def _flags(rows: Sequence[Mapping[str, object]]) -> list[bool]:
    return [bool(row["hallucinated"]) for row in rows if isinstance(row["hallucinated"], bool)]


def _token_sum(rows: Sequence[Mapping[str, object]], answer_field: str, judge_field: str) -> int:
    total = 0
    for row in rows:
        total += _as_int(row[answer_field]) + _as_int(row[judge_field])
    return total


def _as_int(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return 0
    return int(value)


def _mean(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _fmt_rate(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.4f}"


def _fmt_latency(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.2f}"


def _fmt_cost(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.6f}"


def _markdown_line(row: SummaryRow) -> str:
    cells = [
        row.pipeline,
        row.type,
        str(row.n),
        _fmt_rate(row.accuracy),
        _fmt_rate(row.faithfulness),
        _fmt_rate(row.hallucination_rate),
        _fmt_rate(row.recall_at_k),
        _fmt_latency(row.latency_ms_mean),
        str(row.prompt_tokens),
        str(row.completion_tokens),
        _fmt_cost(row.cost_usd),
        str(row.n_cached),
    ]
    return "| " + " | ".join(cells) + " |"


if __name__ == "__main__":
    raise SystemExit(main())
