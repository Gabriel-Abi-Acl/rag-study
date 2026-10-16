"""Índice local em memória com similaridade de cosseno.

O corpus deste estudo cabe em RAM. A interface `add`/`query` é o ponto
para um adapter Chroma ou FAISS, se um PR futuro precisar de persistência.
"""

import math
from collections.abc import Sequence

from rag_study.types import Chunk, ScoredChunk


class InMemoryVectorStore:
    def __init__(self) -> None:
        self._items: list[tuple[Chunk, tuple[float, ...]]] = []
        self._dimension: int | None = None

    def add(self, items: Sequence[tuple[Chunk, Sequence[float]]]) -> None:
        for chunk, vector in items:
            values = tuple(float(value) for value in vector)
            if self._dimension is None:
                self._dimension = len(values)
            if len(values) != self._dimension:
                raise ValueError("Todos os vetores do índice precisam ter a mesma dimensão.")
            self._items.append((chunk, values))

    def query(self, vector: Sequence[float], top_k: int) -> list[ScoredChunk]:
        if top_k < 1:
            raise ValueError("top_k precisa ser >= 1.")
        if self._dimension is not None and len(vector) != self._dimension:
            raise ValueError("A consulta tem dimensão diferente do índice.")
        scored = [
            ScoredChunk(chunk=chunk, score=cosine_similarity(stored, vector))
            for chunk, stored in self._items
        ]
        # Empate: índice do chunk e id, para a ordem não depender da inserção.
        scored.sort(key=lambda item: (-item.score, item.chunk.index, item.chunk.chunk_id))
        return scored[:top_k]


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise ValueError("Vetores com dimensões diferentes.")
    dot = 0.0
    norm_left = 0.0
    norm_right = 0.0
    for a, b in zip(left, right, strict=True):
        dot += a * b
        norm_left += a * a
        norm_right += b * b
    if norm_left == 0.0 or norm_right == 0.0:
        return 0.0
    return dot / (math.sqrt(norm_left) * math.sqrt(norm_right))
