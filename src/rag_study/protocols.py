"""Contratos dos provedores. O teste injeta fakes; a CLI escolhe Gemini ou OpenAI.

O reranker é protocolo de propósito: neste PR a implementação não reordena.
"""

from collections.abc import Sequence
from typing import Protocol

from rag_study.types import ScoredChunk


class LLM(Protocol):
    def complete(self, prompt: str) -> str:
        """Devolve o texto do modelo para um único prompt já montado."""


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Um vetor por texto, na mesma ordem. Lista vazia devolve lista vazia."""


class Reranker(Protocol):
    def rerank(
        self,
        query: str,
        candidates: Sequence[ScoredChunk],
        top_k: int,
    ) -> list[ScoredChunk]:
        """Ordena candidatos para a pergunta e devolve no máximo top_k."""
