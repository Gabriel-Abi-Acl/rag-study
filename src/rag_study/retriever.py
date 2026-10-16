"""Recuperação densa: embede a pergunta e pede os vizinhos ao store.

Não aplica rerank. A pipeline rag_rerank faz isso num passo separado,
para o teste distinguir ordem do índice e ordem do reranker.
"""

from rag_study.protocols import Embedder
from rag_study.types import ScoredChunk
from rag_study.vector_store import InMemoryVectorStore


class Retriever:
    def __init__(self, store: InMemoryVectorStore, embedder: Embedder) -> None:
        self._store = store
        self._embedder = embedder

    def retrieve(self, query: str, top_k: int) -> list[ScoredChunk]:
        vector = self._embedder.embed([query])[0]
        return self._store.query(vector, top_k)
