"""Tipos trocados entre loader, índice, retriever e pipelines.

Ficam imutáveis para o teste poder comparar ordem sem o store alterar o chunk.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Document:
    doc_id: str
    source: str
    text: str


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    source: str
    text: str
    index: int


@dataclass(frozen=True)
class ScoredChunk:
    chunk: Chunk
    score: float


@dataclass(frozen=True)
class Answer:
    """Resposta de uma pipeline. contexts e sources seguem a ordem enviada ao LLM."""

    text: str
    pipeline: str
    contexts: tuple[str, ...]
    sources: tuple[str, ...]
