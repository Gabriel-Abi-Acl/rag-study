"""Executa uma pipeline no gabarito e grava uma linha JSONL por pergunta.

O arquivo guarda resposta, chunks, latência, tokens e custo. Não imprime
acerto nem tabela: número de qualidade fica para o relatório do experimento.
"""

import argparse
import json
import sys
import time
from collections.abc import Callable, Mapping
from dataclasses import replace
from pathlib import Path

from rag_study.cache import CachedLLM, CacheError, Pace, ResponseCache
from rag_study.config import Settings, load_settings
from rag_study.dataset import DatasetError, EvalDataset, load_eval_dataset
from rag_study.errors import ConfigError, ProviderError
from rag_study.factory import (
    build_judge,
    build_llm,
    build_pipelines,
    generation_model_name,
    judge_model_name,
)
from rag_study.judge import Judge, LLMJudge
from rag_study.metrics import (
    choose_correct,
    correctness_match,
    hallucination_flag,
    recall_at_k,
)
from rag_study.pipelines import PIPELINE_ORDER
from rag_study.protocols import LLM
from rag_study.usage import ModelPrice, PriceError, TokenUsage, estimate_cost, load_price_table

RUN_SCHEMA = "eval-run.v1"


def run_evaluation(
    dataset: EvalDataset,
    pipeline_name: str,
    output: Path,
    settings: Settings,
    *,
    llm: LLM | None = None,
    judge: Judge | None = None,
    use_settings_judge: bool = False,
    cache: ResponseCache | None = None,
    prices: Mapping[str, ModelPrice] | None = None,
    limit: int | None = None,
    model_name: str | None = None,
    min_interval: float | None = None,
    max_attempts: int | None = None,
    base_delay: float = 2.0,
    max_delay: float = 60.0,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.perf_counter,
) -> list[dict[str, object]]:
    """Responde o recorte do gabarito. O corpus indexado é o do dataset."""
    if pipeline_name not in PIPELINE_ORDER:
        allowed = ", ".join(PIPELINE_ORDER)
        raise ConfigError(f"pipeline inválida: {pipeline_name}. Use: {allowed}.")
    if limit is not None and limit < 1:
        raise ConfigError("limit precisa ser >= 1.")
    interval = settings.request_interval_s if min_interval is None else min_interval
    attempts = settings.llm_max_retries if max_attempts is None else max_attempts
    if interval < 0:
        raise ConfigError("REQUEST_INTERVAL_S precisa ser >= 0.")
    if attempts < 1:
        raise ConfigError("LLM_MAX_RETRIES precisa ser >= 1.")
    settings = replace(settings, corpus_path=dataset.corpus_dir)
    pace = Pace(interval, sleep, clock)
    model = generation_model_name(settings) if model_name is None else model_name
    inner = build_llm(settings) if llm is None else llm
    client = _cached(inner, model, cache, pace, attempts, base_delay, max_delay, sleep)
    if judge is None and use_settings_judge:
        judge = _settings_judge(settings, cache, pace, attempts, base_delay, max_delay, sleep)
    pipeline = build_pipelines(settings, llm=client)[pipeline_name]
    questions = dataset.questions if limit is None else dataset.questions[:limit]
    table = {} if prices is None else prices
    output.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    with output.open("w", encoding="utf-8") as handle:
        for question in questions:
            print(f"{question.id} {pipeline_name}", file=sys.stderr)
            row = _evaluate_question(
                question,
                pipeline_name,
                pipeline,
                client,
                judge,
                model,
                table,
                clock,
            )
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            rows.append(row)
    return rows


def main(argv: list[str] | None = None) -> int:
    """CLI da avaliação. Código 1 em config, gabarito, preço ou provedor."""
    parser = argparse.ArgumentParser(
        description="Responde o gabarito com uma pipeline e grava JSONL. Não agrega métrica.",
    )
    parser.add_argument("--pipeline", required=True, choices=PIPELINE_ORDER)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--questions", type=Path, default=None)
    parser.add_argument("--corpus", type=Path, default=None)
    parser.add_argument("--judge", choices=("gemini", "openai", "fake", "off"), default=None)
    parser.add_argument("--cache", type=Path, default=None)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--prices", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--interval", type=float, default=None)
    parser.add_argument("--retries", type=int, default=None)
    args = parser.parse_args(argv)
    try:
        settings = load_settings()
        if args.judge is not None:
            settings = replace(settings, judge_provider=args.judge)
        if args.interval is not None:
            settings = replace(settings, request_interval_s=args.interval)
        if args.retries is not None:
            settings = replace(settings, llm_max_retries=args.retries)
        dataset = load_eval_dataset(args.questions, args.corpus)
        price_path = settings.price_table if args.prices is None else args.prices
        prices = load_price_table(price_path)
        cache_dir = settings.eval_cache_dir if args.cache is None else args.cache
        cache = None if args.no_cache else ResponseCache(cache_dir)
        rows = run_evaluation(
            dataset,
            args.pipeline,
            args.out,
            settings,
            cache=cache,
            prices=prices,
            limit=args.limit,
            use_settings_judge=True,
        )
    except (DatasetError, ConfigError, PriceError, CacheError, ProviderError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(f"gravado: {args.out}")
    print(f"pipeline: {args.pipeline}")
    print(f"perguntas: {len(rows)}")
    return 0


def _evaluate_question(question, pipeline_name, pipeline, client, judge, model, table, clock):
    started = clock()
    answer = pipeline.answer(question.question)
    latency_ms = (clock() - started) * 1000
    usage = client.last_usage
    if not isinstance(usage, TokenUsage):
        usage = TokenUsage(0, 0, 0, True)
    correct_judge = None
    faithfulness = None
    judge_model = None
    judge_usage = None
    if judge is not None:
        score = judge.score(
            question.question,
            question.reference_answer,
            answer.text,
            answer.contexts,
        )
        correct_judge = score.correct
        faithfulness = score.faithfulness
        judge_model = score.model
        judge_usage = score.usage
    correct_match = correctness_match(
        question.type,
        answer.text,
        question.reference_answer,
        question.question,
    )
    flag = hallucination_flag(question.type, answer.text, question.question)
    price = table.get(model)
    judge_price = table.get(judge_model) if judge_model is not None else None
    return {
        "schema": RUN_SCHEMA,
        "id": question.id,
        "pipeline": pipeline_name,
        "type": question.type,
        "question": question.question,
        "reference_answer": question.reference_answer,
        "source_doc_ids": list(question.source_doc_ids),
        "answer": answer.text,
        "retrieved_chunk_ids": list(answer.chunk_ids),
        "k": len(answer.chunk_ids),
        "latency_ms": round(latency_ms, 3),
        "cached": bool(client.last_cached),
        "model": model,
        "prompt_tokens": usage.prompt_tokens,
        "completion_tokens": usage.completion_tokens,
        "total_tokens": usage.total_tokens,
        "tokens_estimated": usage.estimated,
        "cost_usd": estimate_cost(usage, price),
        "judge_model": judge_model,
        "judge_prompt_tokens": None if judge_usage is None else judge_usage.prompt_tokens,
        "judge_completion_tokens": None if judge_usage is None else judge_usage.completion_tokens,
        "judge_tokens_estimated": None if judge_usage is None else judge_usage.estimated,
        "judge_cost_usd": None if judge_usage is None else estimate_cost(judge_usage, judge_price),
        "correct_match": correct_match,
        "correct_judge": correct_judge,
        "correct": choose_correct(question.type, correct_match, correct_judge),
        "faithfulness": faithfulness,
        "hallucinated": flag,
        "abstained": None if flag is None else not flag,
        "recall_at_k": recall_at_k(question.source_doc_ids, answer.chunk_ids),
    }


def _cached(inner, model, cache, pace, attempts, base_delay, max_delay, sleep) -> CachedLLM:
    if isinstance(inner, CachedLLM):
        return inner
    return CachedLLM(
        inner,
        model=model,
        cache=cache,
        pace=pace,
        max_attempts=attempts,
        base_delay=base_delay,
        max_delay=max_delay,
        sleep=sleep,
    )


def _settings_judge(
    settings,
    cache,
    pace,
    attempts,
    base_delay,
    max_delay,
    sleep,
) -> LLMJudge | None:
    inner = build_judge(settings)
    if inner is None:
        return None
    model = judge_model_name(settings)
    client = _cached(inner, model, cache, pace, attempts, base_delay, max_delay, sleep)
    return LLMJudge(client, model)


if __name__ == "__main__":
    raise SystemExit(main())
