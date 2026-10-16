"""Demo local das três pipelines. Não imprime métrica nem comparação de qualidade.

Serve para checar se a chave, o corpus e o índice sobem. A avaliação
versionada fica em `python -m rag_study.run`.
"""

import argparse

from rag_study.config import load_settings
from rag_study.factory import build_pipelines
from rag_study.pipelines import PIPELINE_ORDER


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Roda a mesma pergunta nas três pipelines. Não é uma avaliação.",
    )
    parser.add_argument(
        "question",
        nargs="?",
        default="Qual é a população de Vila Aurora?",
    )
    args = parser.parse_args(argv)
    settings = load_settings()
    pipelines = build_pipelines(settings)
    for name in PIPELINE_ORDER:
        answer = pipelines[name].answer(args.question)
        print(f"[{answer.pipeline}]")
        print(answer.text)
        print(f"contextos: {len(answer.contexts)}")
        if answer.sources:
            print(f"fontes: {', '.join(dict.fromkeys(answer.sources))}")
        print()
    return 0
