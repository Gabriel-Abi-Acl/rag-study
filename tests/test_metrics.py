"""Contrato das métricas determinísticas: match, abstenção, recall e escolha do juiz.

Não chama modelo. O juiz só entra quando o teste passa a nota dele.
"""

import pytest

from rag_study.metrics import (
    abstained,
    choose_correct,
    correctness_match,
    hallucination_flag,
    normalized_match,
    recall_at_k,
)


def test_factual_and_numeric_match_ignores_case_and_keeps_number_spelling():
    assert normalized_match("A prefeita é Helena Duarte.", "Helena Duarte")
    assert normalized_match("helena   duarte", "Helena Duarte")
    assert normalized_match("A população é de 12.450 habitantes.", "12.450")
    assert not normalized_match("Cerca de 12.540 habitantes.", "12.450")
    assert not normalized_match("A população é de 12450 habitantes.", "12.450")
    assert not normalized_match("ribeirao das laranjeiras", "Ribeirão das Laranjeiras")
    assert correctness_match("factual", "Helena Duarte", "Helena Duarte") == 1
    assert correctness_match("numeric", "12.540", "12.450") == 0
    paraphrase = correctness_match(
        "multi_hop",
        "só Helena Duarte",
        "Helena Duarte rejeitou a tarifa",
    )
    assert paraphrase == 0


def test_abstention_accepts_a_short_refusal_and_rejects_an_added_fact():
    assert abstained("")
    assert abstained("Não consta no corpus.")
    assert abstained("Não sei.")
    assert abstained("Nao sei")
    assert abstained("Não há informação disponível.")
    assert abstained("Não há menção a ciclovia.", "Existe ciclovia em Vila Aurora?")
    assert not abstained("Não há menção a ciclovia.")
    assert not abstained("Não sei. O PIB é 9 milhões.", "Qual é o PIB?")
    assert not abstained("Não sei, talvez Carlos.", "Quem é o prefeito?")
    assert not abstained("A população é 12.450.")


def test_hallucination_is_only_defined_for_unanswerable_questions():
    assert hallucination_flag("unanswerable", "O PIB é 9 milhões.", "Qual é o PIB?") is True
    assert hallucination_flag("unanswerable", "Não sei.", "Qual é o PIB?") is False
    assert hallucination_flag("factual", "Não sei.", "Quem é a prefeita?") is None
    assert correctness_match("unanswerable", "Não consta no corpus.", "Não consta no corpus.") == 1
    invented = correctness_match(
        "unanswerable",
        "O CEP é 00000-000.",
        "Não consta no corpus.",
        "Qual é o CEP?",
    )
    assert invented == 0


def test_judge_cannot_redefine_abstention():
    assert choose_correct("factual", 0, 1) == 1
    assert choose_correct("numeric", 1, 0) == 0
    assert choose_correct("multi_hop", 0, None) == 0
    assert choose_correct("unanswerable", 0, 1) == 0
    assert choose_correct("unanswerable", 1, 0) == 1


def test_recall_at_k_uses_gold_documents_and_is_undefined_without_them():
    assert recall_at_k(["a.txt", "b.txt"], ["a.txt:0", "a.txt:1"]) == pytest.approx(0.5)
    assert recall_at_k(["a.txt"], []) == 0
    assert recall_at_k([], ["a.txt:0"]) is None
    assert recall_at_k(["pasta/a.txt"], ["pasta/a.txt:2"]) == 1
    assert recall_at_k(["a.txt"], ["a.txt"]) == 1
