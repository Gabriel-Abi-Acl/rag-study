"""Contrato do gabarito v1: schema, citação, apoio literal e desenho do corpus.

Não abre rede. Lê `data/eval` e `data/sample`, ou arquivos em tmp_path.
"""

from collections import Counter
from pathlib import Path

import pytest

from rag_study.dataset import (
    ABSTENTION_ANSWER,
    QUESTION_TYPES,
    DatasetError,
    EvalDataset,
    EvalQuestion,
    load_eval_dataset,
    load_questions,
    main,
    reference_supported,
    validate_dataset,
)
from rag_study.types import Document

EXPECTED_COUNTS = {"factual": 18, "numeric": 14, "multi_hop": 12, "unanswerable": 6}
SHARED_LIBRARY_PASSAGE = (
    "A biblioteca fica na Rua do Cedro, 45 e empresta livros para moradores com cadastro. "
    "O balcão de empréstimo abre de terça a sexta."
)
RECYCLING_RULE = "A coleta de lixo reciclável ocorre às quartas-feiras."


def test_schema_of_shipped_questions():
    dataset = load_eval_dataset()
    assert dataset.version == "1"
    assert len(dataset.questions) == 50
    assert [question.id for question in dataset.questions] == [f"q{i:03d}" for i in range(1, 51)]
    for question in dataset.questions:
        assert question.type in QUESTION_TYPES
        assert question.question.strip() == question.question
        assert question.reference_answer.strip() == question.reference_answer


def test_ids_are_unique():
    questions = load_questions(load_eval_dataset().questions_path)
    ids = [question.id for question in questions]
    assert len(ids) == len(set(ids))


def test_every_source_doc_id_exists_in_the_corpus():
    dataset = load_eval_dataset()
    doc_ids = {document.doc_id for document in dataset.documents}
    for question in dataset.questions:
        missing = [doc_id for doc_id in question.source_doc_ids if doc_id not in doc_ids]
        assert missing == []


def test_factual_and_numeric_answers_are_supported_by_cited_docs():
    dataset = load_eval_dataset()
    by_id = {document.doc_id: document.text for document in dataset.documents}
    checked = 0
    for question in dataset.questions:
        if question.type not in ("factual", "numeric"):
            continue
        texts = [by_id[doc_id] for doc_id in question.source_doc_ids]
        assert reference_supported(question.reference_answer, texts), question.id
        checked += 1
    assert checked == EXPECTED_COUNTS["factual"] + EXPECTED_COUNTS["numeric"]


def test_type_distribution_matches_v1():
    dataset = load_eval_dataset()
    counts = Counter(question.type for question in dataset.questions)
    assert dict(counts) == EXPECTED_COUNTS
    assert set(counts) == set(QUESTION_TYPES)


def test_multi_hop_and_unanswerable_follow_the_citation_rules():
    dataset = load_eval_dataset()
    by_id = {document.doc_id: document.text for document in dataset.documents}
    for question in dataset.questions:
        if question.type == "multi_hop":
            assert len(question.source_doc_ids) >= 2
            texts = [by_id[doc_id] for doc_id in question.source_doc_ids]
            assert not all(reference_supported(question.reference_answer, [text]) for text in texts)
        if question.type == "unanswerable":
            assert question.source_doc_ids == ()
            assert question.reference_answer == ABSTENTION_ANSWER
        if question.type in ("factual", "numeric"):
            assert len(question.source_doc_ids) == 1


def test_shipped_dataset_passes_the_validator():
    assert validate_dataset(load_eval_dataset()) == []


def test_validation_script_accepts_the_shipped_dataset(capsys):
    assert main([]) == 0
    output = capsys.readouterr().out
    assert "dataset v1" in output
    assert "perguntas: 50" in output
    assert "documentos: 24" in output
    assert output.endswith("ok\n")


def test_corpus_shape_distractors_and_near_duplicates():
    dataset = load_eval_dataset()
    by_id = {document.doc_id: document.text for document in dataset.documents}
    assert 15 <= len(by_id) <= 30
    lengths = [len(text) for text in by_id.values()]
    assert sum(length < 500 for length in lengths) >= 3
    assert sum(length > 3500 for length in lengths) >= 2
    assert SHARED_LIBRARY_PASSAGE in by_id["biblioteca.md"]
    assert SHARED_LIBRARY_PASSAGE in by_id["blog_biblioteca.md"]
    assert "1978" in by_id["biblioteca.md"]
    assert "1978" not in by_id["blog_biblioteca.md"]
    assert "1981" in by_id["blog_biblioteca.md"]
    assert RECYCLING_RULE in by_id["residuos.md"]
    assert RECYCLING_RULE in by_id["ata_camara.txt"]
    assert "12.450" in by_id["populacao.md"]
    assert "12.450" not in by_id["rumor_populacao.md"]
    assert "12.450" not in by_id["serra_verde.txt"]
    assert "20.000" in by_id["rumor_populacao.md"]
    assert "12.540" in by_id["serra_verde.txt"]
    assert "R$ 4,50" in by_id["transporte.md"]
    assert "R$ 4,50" not in by_id["ata_camara.txt"]
    assert "R$ 5,00" in by_id["ata_camara.txt"]
    cited = {doc_id for question in dataset.questions for doc_id in question.source_doc_ids}
    assert "serra_verde.txt" not in cited
    for term in ("PIB", "UTI neonatal", "xadrez", "caixa d'água", "ciclovia"):
        assert term.casefold() not in "\n".join(by_id.values()).casefold()


def test_support_does_not_join_words_across_documents():
    assert reference_supported("Helena Duarte", ["A prefeita chama-se Helena Duarte."])
    assert reference_supported("12.450", ["A população estimada é de 12.450 habitantes."])
    assert not reference_supported("Helena Duarte", ["Helena", "Duarte"])
    assert not reference_supported("12.450", ["A população é de 12.540 habitantes."])


def test_schema_rejects_bad_type_duplicate_id_and_wrong_filename(tmp_path):
    path = tmp_path / "questions.v1.jsonl"
    path.write_text(
        "\n".join(
            [
                '{"id": "q001", "question": "Qual?", "reference_answer": "x", '
                '"source_doc_ids": [], "type": "factual"}',
                '{"id": "q001", "question": "Outra?", "reference_answer": "y", '
                '"source_doc_ids": [], "type": "factual"}',
                '{"id": "q002", "question": "Tipo?", "reference_answer": "z", '
                '"source_doc_ids": [], "type": "nope"}',
                "nao-json",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(DatasetError) as caught:
        load_questions(path)
    text = "\n".join(caught.value.errors)
    assert "type inválido" in text
    assert "id duplicado" in text
    assert "JSON inválido" in text
    with pytest.raises(DatasetError, match="questions.vN.jsonl"):
        load_questions(tmp_path / "perguntas.jsonl")


def test_validator_flags_missing_doc_unsupported_answer_and_bad_distribution():
    questions = (
        EvalQuestion("q001", "Onde fica?", "Marte", ("ausente.txt",), "factual"),
        EvalQuestion("q002", "Quanto é?", "99", ("a.txt",), "numeric"),
        EvalQuestion("q003", "Quem?", "Ana", ("a.txt", "b.txt"), "multi_hop"),
        EvalQuestion("q004", "Qual o PIB?", "inventei", ("a.txt",), "unanswerable"),
    )
    documents = (
        Document("a.txt", "a.txt", "Ana mora aqui. O valor é 10."),
        Document("b.txt", "b.txt", "Ana mora aqui. O valor é 10."),
    )
    errors = validate_dataset(
        EvalDataset("1", Path("questions.v1.jsonl"), Path("corpus"), questions, documents)
    )
    joined = "\n".join(errors)
    assert "ausente.txt" in joined
    assert "q002: gabarito fora do documento citado" in joined
    assert "q003: gabarito multi_hop cabe inteiro em cada documento citado" in joined
    assert "q004: unanswerable não cita documento" in joined
    assert "abstenção" in joined
    assert "fora da faixa" in joined


def test_validation_script_rejects_a_broken_file(tmp_path, capsys):
    path = tmp_path / "questions.v1.jsonl"
    path.write_text("{}\n", encoding="utf-8")
    assert main(["--questions", str(path), "--corpus", str(tmp_path)]) == 1
    assert "dataset inválido:" in capsys.readouterr().out
