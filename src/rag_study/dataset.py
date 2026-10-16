"""Gabarito versionado do estudo e a checagem offline contra o corpus.

O arquivo é `data/eval/questions.vN.jsonl`. A validação não chama modelo
nem rede: só confere schema, citação e se o gabarito factual ou numérico
está escrito nos documentos citados.
"""

import argparse
import json
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from rag_study.loader import load_corpus
from rag_study.types import Document

QUESTION_TYPES = ("factual", "numeric", "multi_hop", "unanswerable")
ABSTENTION_ANSWER = "Não consta no corpus."
_FIELDS = ("id", "question", "reference_answer", "source_doc_ids", "type")
_SUPPORT_TYPES = ("factual", "numeric")
_MIN_BY_TYPE = {"factual": 10, "numeric": 8, "multi_hop": 8, "unanswerable": 4}
_VERSION_RE = re.compile(r"questions\.v(\d+)\.jsonl")
_ID_RE = re.compile(r"q\d{3}")
_QUESTIONS_NAME = "questions.v1.jsonl"


class DatasetError(ValueError):
    """JSONL ilegível ou fora do nome versionado. O texto lista cada falha."""

    def __init__(self, errors: list[str]) -> None:
        self.errors = errors
        super().__init__("\n".join(errors))


@dataclass(frozen=True)
class EvalQuestion:
    """Uma linha do gabarito. `type` repete a chave do JSONL."""

    id: str
    question: str
    reference_answer: str
    source_doc_ids: tuple[str, ...]
    type: str


@dataclass(frozen=True)
class EvalDataset:
    """Perguntas já ligadas ao corpus que `load_corpus` enxerga."""

    version: str
    questions_path: Path
    corpus_dir: Path
    questions: tuple[EvalQuestion, ...]
    documents: tuple[Document, ...]


def repo_root() -> Path:
    """Raiz do checkout. O marcador é o jsonl, para o teste não depender do cwd."""
    anchor = Path(__file__).resolve()
    marker = Path("data") / "eval" / _QUESTIONS_NAME
    candidates = [anchor.parents[2], Path.cwd()] if len(anchor.parents) > 2 else [Path.cwd()]
    for candidate in candidates:
        if (candidate / marker).is_file():
            return candidate
    return candidates[0]


def default_questions_path() -> Path:
    return repo_root() / "data" / "eval" / _QUESTIONS_NAME


def default_corpus_dir() -> Path:
    return repo_root() / "data" / "sample"


def load_questions(path: Path) -> tuple[EvalQuestion, ...]:
    """Lê o JSONL. Falha com todas as linhas ruins, não só a primeira."""
    _version_from_name(path)
    if not path.is_file():
        raise DatasetError([f"gabarito não encontrado: {path}"])
    errors: list[str] = []
    questions: list[EvalQuestion] = []
    seen: set[str] = set()
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        question, line_errors = _parse_line(line, number)
        errors.extend(line_errors)
        if question is None:
            continue
        if question.id in seen:
            errors.append(f"linha {number}: id duplicado: {question.id}")
        seen.add(question.id)
        questions.append(question)
    if errors:
        raise DatasetError(errors)
    if not questions:
        raise DatasetError(["gabarito sem perguntas"])
    return tuple(questions)


def load_eval_dataset(
    questions_path: Path | None = None,
    corpus_dir: Path | None = None,
) -> EvalDataset:
    """Junta o JSONL ao corpus. O diretório padrão é `data/sample`, não o .env."""
    path = default_questions_path() if questions_path is None else questions_path
    corpus = default_corpus_dir() if corpus_dir is None else corpus_dir
    version = _version_from_name(path)
    questions = load_questions(path)
    try:
        documents = tuple(load_corpus(corpus))
    except FileNotFoundError as exc:
        raise DatasetError([str(exc)]) from exc
    return EvalDataset(version, path, corpus, questions, documents)


def normalize_text(text: str) -> str:
    return " ".join(text.casefold().split())


def reference_supported(answer: str, texts: Sequence[str]) -> bool:
    """O gabarito factual ou numérico precisa ocorrer nos textos citados.

    Caixa e espaços são ignorados. Números valem como estão grafados no
    corpus (`12.450`, `R$ 4,50`). Uma quebra de linha separa documentos
    para um nome partido entre arquivos não contar como apoio.
    """
    needle = normalize_text(answer)
    if not needle or not texts:
        return False
    haystack = "\n".join(normalize_text(text) for text in texts)
    return needle in haystack


def validate_dataset(dataset: EvalDataset) -> list[str]:
    """Regras do desenho: citação existente, apoio literal, tipo e tamanho.

    Lista vazia significa que o arquivo pode entrar no harness depois.
    Não mede resposta de modelo.
    """
    errors: list[str] = []
    documents = dataset.documents
    questions = dataset.questions
    if not 15 <= len(documents) <= 30:
        errors.append(f"corpus com {len(documents)} documentos; a faixa é 15 a 30")
    by_id: dict[str, Document] = {}
    for document in documents:
        if document.doc_id in by_id:
            errors.append(f"doc_id duplicado no corpus: {document.doc_id}")
        by_id[document.doc_id] = document
    if not 45 <= len(questions) <= 60:
        errors.append(f"total de perguntas fora da faixa de 45 a 60: {len(questions)}")
    counts = Counter(question.type for question in questions)
    for name, minimum in _MIN_BY_TYPE.items():
        if counts[name] < minimum:
            errors.append(f"tipo {name}: {counts[name]} perguntas; mínimo {minimum}")
    for question in questions:
        errors.extend(_validate_question(question, by_id))
    return errors


def main(argv: list[str] | None = None) -> int:
    """Valida o gabarito padrão, ou os caminhos passados em argv. Código 1 se falhar."""
    parser = argparse.ArgumentParser(
        description="Valida o gabarito versionado contra o corpus. Não chama modelo nem rede.",
    )
    parser.add_argument("--questions", type=Path, default=None)
    parser.add_argument("--corpus", type=Path, default=None)
    args = parser.parse_args(argv)
    questions_path = default_questions_path() if args.questions is None else args.questions
    corpus_dir = default_corpus_dir() if args.corpus is None else args.corpus
    try:
        dataset = load_eval_dataset(questions_path, corpus_dir)
    except DatasetError as exc:
        _print_errors(exc.errors)
        return 1
    errors = validate_dataset(dataset)
    if errors:
        _print_errors(errors)
        return 1
    print(f"dataset v{dataset.version}")
    print(f"perguntas: {len(dataset.questions)}")
    print(f"documentos: {len(dataset.documents)}")
    print("ok")
    return 0


def _version_from_name(path: Path) -> str:
    match = _VERSION_RE.fullmatch(path.name)
    if not match:
        message = f"arquivo de gabarito precisa se chamar questions.vN.jsonl: {path.name}"
        raise DatasetError([message])
    return match.group(1)


def _parse_line(line: str, number: int) -> tuple[EvalQuestion | None, list[str]]:
    prefix = f"linha {number}"
    try:
        payload = json.loads(line)
    except json.JSONDecodeError:
        return None, [f"{prefix}: JSON inválido"]
    if not isinstance(payload, dict):
        return None, [f"{prefix}: cada linha precisa ser um objeto"]
    errors: list[str] = []
    missing = [field for field in _FIELDS if field not in payload]
    extra = sorted(set(payload) - set(_FIELDS))
    errors.extend(f"{prefix}: campo obrigatório ausente: {field}" for field in missing)
    errors.extend(f"{prefix}: campo desconhecido: {field}" for field in extra)
    if missing:
        return None, errors
    question_id = _text_field(payload, "id", prefix, errors)
    question_text = _text_field(payload, "question", prefix, errors)
    answer = _text_field(payload, "reference_answer", prefix, errors)
    sources = _sources_field(payload, prefix, errors)
    question_type = _type_field(payload, prefix, errors)
    if question_id is not None and not _ID_RE.fullmatch(question_id):
        errors.append(f"{prefix}: id precisa casar com qNNN")
        question_id = None
    if (
        errors
        or question_id is None
        or question_text is None
        or answer is None
        or sources is None
        or question_type is None
    ):
        return None, errors
    return (
        EvalQuestion(question_id, question_text, answer, sources, question_type),
        [],
    )


def _text_field(
    payload: dict[str, object],
    field: str,
    prefix: str,
    errors: list[str],
) -> str | None:
    value = payload[field]
    if not isinstance(value, str) or not value.strip():
        errors.append(f"{prefix}: {field} precisa ser texto não vazio")
        return None
    return value.strip()


def _type_field(payload: dict[str, object], prefix: str, errors: list[str]) -> str | None:
    value = payload["type"]
    if not isinstance(value, str) or value not in QUESTION_TYPES:
        allowed = ", ".join(QUESTION_TYPES)
        errors.append(f"{prefix}: type inválido. Use: {allowed}.")
        return None
    return value


def _sources_field(
    payload: dict[str, object],
    prefix: str,
    errors: list[str],
) -> tuple[str, ...] | None:
    value = payload["source_doc_ids"]
    items_are_text = isinstance(value, list) and all(
        isinstance(item, str) and item.strip() for item in value
    )
    if not items_are_text:
        errors.append(f"{prefix}: source_doc_ids precisa ser lista de textos")
        return None
    sources = tuple(item.strip() for item in value)
    if len(sources) != len(set(sources)):
        errors.append(f"{prefix}: source_doc_id repetido")
        return None
    return sources


def _validate_question(question: EvalQuestion, by_id: dict[str, Document]) -> list[str]:
    errors: list[str] = []
    missing = [doc_id for doc_id in question.source_doc_ids if doc_id not in by_id]
    for doc_id in missing:
        errors.append(f"{question.id}: source_doc_id ausente no corpus: {doc_id}")
    texts = [by_id[doc_id].text for doc_id in question.source_doc_ids if doc_id in by_id]
    if question.type in _SUPPORT_TYPES:
        if len(question.source_doc_ids) != 1:
            errors.append(f"{question.id}: {question.type} cita um único documento")
        elif not missing and not reference_supported(question.reference_answer, texts):
            errors.append(f"{question.id}: gabarito fora do documento citado")
    elif question.type == "multi_hop":
        if len(question.source_doc_ids) < 2:
            errors.append(f"{question.id}: multi_hop cita pelo menos dois documentos")
        elif not missing and _contained_in_every_doc(question.reference_answer, texts):
            errors.append(
                f"{question.id}: gabarito multi_hop cabe inteiro em cada documento citado"
            )
    elif question.type == "unanswerable":
        if question.source_doc_ids:
            errors.append(f"{question.id}: unanswerable não cita documento")
        if question.reference_answer != ABSTENTION_ANSWER:
            errors.append(f"{question.id}: unanswerable usa o gabarito fixo de abstenção")
    return errors


def _contained_in_every_doc(answer: str, texts: Sequence[str]) -> bool:
    needle = normalize_text(answer)
    if not needle or not texts:
        return False
    return all(needle in normalize_text(text) for text in texts)


def _print_errors(errors: list[str]) -> None:
    print("dataset inválido:")
    for error in errors:
        print(f"- {error}")


if __name__ == "__main__":
    raise SystemExit(main())
