"""Três formas de responder a mesma pergunta.

no_context não vê o corpus. naive_rag concatena o top_k do retriever.
rag_rerank pede uma lista maior e só então chama o reranker. O prompt final
usa a ordem devolvida pelo reranker, truncada de novo em top_k.
"""

from collections.abc import Sequence

from rag_study.protocols import LLM, Reranker
from rag_study.retriever import Retriever
from rag_study.types import Answer, ScoredChunk


class NoContextPipeline:
    name = "no_context"

    def __init__(self, llm: LLM) -> None:
        self._llm = llm

    def answer(self, question: str) -> Answer:
        prompt = f"Responda à pergunta. Se não souber, diga que não sabe.\n\nPergunta: {question}"
        return Answer(
            text=self._llm.complete(prompt),
            pipeline=self.name,
            contexts=(),
            sources=(),
        )


class NaiveRagPipeline:
    name = "naive_rag"

    def __init__(self, llm: LLM, retriever: Retriever, top_k: int) -> None:
        self._llm = llm
        self._retriever = retriever
        self._top_k = top_k

    def answer(self, question: str) -> Answer:
        hits = self._retriever.retrieve(question, self._top_k)
        return _complete_with_context(self._llm, self.name, question, hits)


class RagRerankPipeline:
    name = "rag_rerank"

    def __init__(
        self,
        llm: LLM,
        retriever: Retriever,
        reranker: Reranker,
        top_k: int,
        candidate_k: int,
    ) -> None:
        self._llm = llm
        self._retriever = retriever
        self._reranker = reranker
        self._top_k = top_k
        self._candidate_k = candidate_k

    def answer(self, question: str) -> Answer:
        candidates = self._retriever.retrieve(question, self._candidate_k)
        ranked = self._reranker.rerank(question, candidates, self._top_k)
        # O reranker promete top_k; o corte extra segura uma implementação que ignore o limite.
        return _complete_with_context(self._llm, self.name, question, ranked[: self._top_k])


def _complete_with_context(
    llm: LLM,
    pipeline: str,
    question: str,
    hits: Sequence[ScoredChunk],
) -> Answer:
    blocks = [
        f"[{index}] fonte: {hit.chunk.source}\n{hit.chunk.text}"
        for index, hit in enumerate(hits, start=1)
    ]
    context = "\n\n".join(blocks) if blocks else "(nenhum trecho recuperado)"
    prompt = (
        "Responda usando somente o contexto. "
        "Se o contexto não tiver a resposta, diga que não sabe.\n\n"
        f"Contexto:\n{context}\n\n"
        f"Pergunta: {question}"
    )
    return Answer(
        text=llm.complete(prompt),
        pipeline=pipeline,
        contexts=tuple(hit.chunk.text for hit in hits),
        sources=tuple(hit.chunk.source for hit in hits),
    )
