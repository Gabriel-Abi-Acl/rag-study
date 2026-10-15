"""Ordem do índice: maior cosseno primeiro, empate pelo índice do chunk."""

import pytest

from rag_study.embeddings import HashingEmbedder
from rag_study.retriever import Retriever
from rag_study.types import Chunk, ScoredChunk
from rag_study.vector_store import InMemoryVectorStore, cosine_similarity


def _chunk(text: str, index: int) -> Chunk:
    return Chunk(
        chunk_id=f"doc:{index}",
        doc_id="doc",
        source="doc.txt",
        text=text,
        index=index,
    )


def test_query_returns_highest_cosine_first_and_limits_top_k():
    store = InMemoryVectorStore()
    store.add(
        [
            (_chunk("longe", 0), [0.0, 1.0]),
            (_chunk("perto", 1), [1.0, 0.0]),
        ]
    )
    hits = store.query([0.9, 0.1], top_k=1)
    assert [hit.chunk.text for hit in hits] == ["perto"]
    assert hits[0].score > 0.9


def test_tie_breaks_by_chunk_index_then_id():
    store = InMemoryVectorStore()
    store.add(
        [
            (_chunk("segundo", 1), [1.0, 0.0]),
            (_chunk("primeiro", 0), [1.0, 0.0]),
        ]
    )
    hits = store.query([1.0, 0.0], top_k=2)
    assert [hit.chunk.index for hit in hits] == [0, 1]
    assert all(hit.score == pytest.approx(1.0) for hit in hits)


def test_zero_vector_scores_zero_and_dimension_mismatch_raises():
    assert cosine_similarity([0.0, 0.0], [1.0, 0.0]) == 0.0
    store = InMemoryVectorStore()
    store.add([(_chunk("a", 0), [1.0, 0.0])])
    with pytest.raises(ValueError):
        store.query([1.0, 0.0, 0.0], top_k=1)
    with pytest.raises(ValueError):
        store.query([1.0, 0.0], top_k=0)


def test_retriever_embeds_the_query_and_keeps_store_order():
    class ScriptedEmbedder:
        def embed(self, texts: list[str]) -> list[list[float]]:
            assert texts == ["pergunta"]
            return [[1.0, 0.0]]

    store = InMemoryVectorStore()
    store.add(
        [
            (_chunk("longe", 0), [0.0, 1.0]),
            (_chunk("perto", 1), [0.8, 0.2]),
        ]
    )
    hits = Retriever(store, ScriptedEmbedder()).retrieve("pergunta", top_k=2)
    assert [hit.chunk.text for hit in hits] == ["perto", "longe"]
    assert isinstance(hits[0], ScoredChunk)


def test_hashing_embedder_is_deterministic_and_groups_shared_tokens():
    embedder = HashingEmbedder(dimension=64)
    base = embedder.embed(["gatos bebem leite"])[0]
    assert embedder.embed(["gatos bebem leite"])[0] == base
    close = embedder.embed(["leite dos gatos"])[0]
    far = embedder.embed(["trens de carga no planalto"])[0]
    assert cosine_similarity(base, close) > cosine_similarity(base, far)
