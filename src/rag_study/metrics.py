"""Métricas determinísticas do harness. O juiz, se houver, fica em rag_study.judge.

O match factual reusa a normalização do gabarito: caixa e espaços, sem
trocar a grafia do número. Alucinação aqui é só responder unanswerable.
"""

import re
import unicodedata
from collections.abc import Sequence

from rag_study.dataset import ABSTENTION_ANSWER, normalize_text

# Frases de recusa, da mais longa para a mais curta na hora de apagar.
_REFUSAL_CUES = (
    "nao tenho essa informacao",
    "nao e possivel responder",
    "informacao nao disponivel",
    "nao tenho informacao",
    "nao ha informacao",
    "nao posso responder",
    "nao foi mencionada",
    "nao foi mencionado",
    "nao foi encontrado",
    "nao ha mencao",
    "nao ha registro",
    "nao ha dados",
    "sem informacao",
    "nao menciona",
    "nao encontrei",
    "nao disponho",
    "desconheco",
    "nao consta",
    "nao sei",
)
_FILLER = {
    "o",
    "a",
    "os",
    "as",
    "um",
    "uma",
    "de",
    "do",
    "da",
    "dos",
    "das",
    "que",
    "e",
    "em",
    "no",
    "na",
    "nos",
    "nas",
    "para",
    "com",
    "se",
    "ao",
    "aos",
    "por",
    "ou",
    "mas",
    "nao",
    "ha",
    "isso",
    "essa",
    "esse",
    "este",
    "esta",
    "eu",
    "me",
    "muito",
    "apenas",
    "so",
    "talvez",
    "provavelmente",
    "cerca",
    "acho",
    "parece",
    "dizer",
    "responder",
    "informar",
    "corpus",
    "contexto",
    "documento",
    "documentos",
    "informacao",
    "dados",
    "resposta",
    "pergunta",
    "trecho",
    "trechos",
    "fonte",
    "fontes",
    "encontrado",
    "encontrada",
    "disponivel",
    "registro",
    "registros",
    "suporte",
    "base",
    "texto",
    "textos",
    "mencao",
    "consta",
}
_TOKEN = re.compile(r"\w+", flags=re.UNICODE)


def fold_accents(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text.casefold())
    stripped = "".join(char for char in decomposed if not unicodedata.combining(char))
    return " ".join(stripped.split())


def normalized_match(answer: str, reference: str) -> bool:
    """O gabarito normalizado cabe na resposta. 12.450 não casa com 12.540."""
    needle = normalize_text(reference)
    if not needle:
        return False
    return needle in normalize_text(answer)


def abstained(answer: str, question: str = "") -> bool:
    """Recusa sem fato novo, inclusive resposta vazia.

    Palavra que já está na pergunta não conta; dígito ou palavra restante é resposta.
    """
    if answer.strip() == "":
        return True
    if fold_accents(answer) == fold_accents(ABSTENTION_ANSWER):
        return True
    folded = fold_accents(answer)
    if not any(cue in folded for cue in _REFUSAL_CUES):
        return False
    return not _claim_tokens(folded, question)


def correctness_match(
    question_type: str,
    answer: str,
    reference: str,
    question: str = "",
) -> float:
    """1 ou 0. unanswerable usa abstenção. Os outros usam o match normalizado."""
    if question_type == "unanswerable":
        return 1.0 if abstained(answer, question) else 0.0
    return 1.0 if normalized_match(answer, reference) else 0.0


def choose_correct(
    question_type: str,
    correct_match: float,
    correct_judge: float | None,
) -> float:
    """O juiz pode aceitar paráfrase. Ele não redefine abstenção."""
    if question_type == "unanswerable" or correct_judge is None:
        return correct_match
    return correct_judge


def hallucination_flag(question_type: str, answer: str, question: str = "") -> bool | None:
    """Só unanswerable entra na taxa. Nos outros tipos o campo fica nulo."""
    if question_type != "unanswerable":
        return None
    return not abstained(answer, question)


def recall_at_k(
    source_doc_ids: Sequence[str],
    retrieved_chunk_ids: Sequence[str],
) -> float | None:
    """Documentos de ouro cobertos pelos chunks. Ouro vazio não tem recall.

    O id do chunk é `doc_id:indice`. O doc_id é o que vem antes do último `:`.
    """
    gold = set(source_doc_ids)
    if not gold:
        return None
    retrieved = {_doc_id(chunk_id) for chunk_id in retrieved_chunk_ids}
    return len(gold & retrieved) / len(gold)


def _claim_tokens(folded_answer: str, question: str) -> list[str]:
    text = folded_answer
    for cue in sorted(_REFUSAL_CUES, key=len, reverse=True):
        text = text.replace(cue, " ")
    question_tokens = {token for token in _TOKEN.findall(fold_accents(question)) if len(token) >= 3}
    claims: list[str] = []
    for token in _TOKEN.findall(text):
        if token in _FILLER or token in question_tokens:
            continue
        if any(char.isdigit() for char in token) or len(token) >= 3:
            claims.append(token)
    return claims


def _doc_id(chunk_id: str) -> str:
    if ":" not in chunk_id:
        return chunk_id
    return chunk_id.rsplit(":", 1)[0]
