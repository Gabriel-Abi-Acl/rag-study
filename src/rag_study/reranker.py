"""Rerank ainda não muda a ordem. A interface existe para o PR do cross-encoder.

IdentityReranker não pode ser citado como ganho experimental: ele só corta
a lista no top_k e preserva a ordem do retriever.
"""

from collections.abc import Sequence

from rag_study.types import ScoredChunk


class IdentityReranker:
    def rerank(
        self,
        query: str,
        candidates: Sequence[ScoredChunk],
        top_k: int,
    ) -> list[ScoredChunk]:
        del query
        if top_k < 1:
            raise ValueError("top_k precisa ser >= 1.")
        return list(candidates[:top_k])
