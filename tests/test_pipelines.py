"""As três pipelines usam o mesmo LLM, mas só as de RAG veem o corpus.

rag_rerank segue a ordem do reranker, não a ordem crua do retriever.
"""

from rag_study.llm import FakeLLM
from rag_study.pipelines import NaiveRagPipeline, NoContextPipeline, RagRerankPipeline
from rag_study.reranker import IdentityReranker
from rag_study.types import Chunk, ScoredChunk


def _hit(text: str, index: int, score: float) -> ScoredChunk:
    return ScoredChunk(
        chunk=Chunk(
            chunk_id=f"doc:{index}",
            doc_id="doc",
            source="doc.txt",
            text=text,
            index=index,
        ),
        score=score,
    )


class RecordingRetriever:
    def __init__(self, hits: list[ScoredChunk]) -> None:
        self.hits = hits
        self.calls: list[tuple[str, int]] = []

    def retrieve(self, query: str, top_k: int) -> list[ScoredChunk]:
        self.calls.append((query, top_k))
        return self.hits[:top_k]


class ReverseReranker:
    def rerank(self, query: str, candidates: list[ScoredChunk], top_k: int) -> list[ScoredChunk]:
        del query
        return list(reversed(candidates))[:top_k]


def test_no_context_hides_corpus_from_the_prompt():
    llm = FakeLLM(reply="sem-contexto")
    answer = NoContextPipeline(llm).answer("Qual a população?")
    assert answer.pipeline == "no_context"
    assert answer.contexts == ()
    assert answer.chunk_ids == ()
    assert answer.text == "sem-contexto"
    assert "Qual a população?" in llm.prompts[0]
    assert "12.450" not in llm.prompts[0]


def test_naive_rag_puts_retrieved_chunks_in_score_order():
    llm = FakeLLM()
    retriever = RecordingRetriever([_hit("fato-a", 0, 0.2), _hit("fato-b", 1, 0.9)])
    answer = NaiveRagPipeline(llm, retriever, top_k=2).answer("pergunta")
    assert retriever.calls == [("pergunta", 2)]
    assert answer.pipeline == "naive_rag"
    assert answer.contexts == ("fato-a", "fato-b")
    assert answer.chunk_ids == ("doc:0", "doc:1")
    prompt = llm.prompts[0]
    assert prompt.index("fato-a") < prompt.index("fato-b")
    assert "Responda usando somente o contexto." in prompt


def test_rag_rerank_uses_reranker_order_and_wider_candidate_pool():
    llm = FakeLLM()
    retriever = RecordingRetriever(
        [_hit("primeiro", 0, 0.9), _hit("segundo", 1, 0.5), _hit("terceiro", 2, 0.1)]
    )
    answer = RagRerankPipeline(
        llm,
        retriever,
        ReverseReranker(),
        top_k=2,
        candidate_k=3,
    ).answer("pergunta")
    assert retriever.calls == [("pergunta", 3)]
    assert answer.pipeline == "rag_rerank"
    assert answer.contexts == ("terceiro", "segundo")
    assert answer.chunk_ids == ("doc:2", "doc:1")
    prompt = llm.prompts[0]
    assert prompt.index("terceiro") < prompt.index("segundo")
    assert "primeiro" not in prompt


def test_identity_reranker_preserves_order_and_pipeline_still_caps_top_k():
    hits = [_hit("a", 0, 1.0), _hit("b", 1, 0.5), _hit("c", 2, 0.2)]
    assert [hit.chunk.text for hit in IdentityReranker().rerank("q", hits, top_k=2)] == ["a", "b"]

    class GreedyReranker:
        def rerank(self, query, candidates, top_k):
            del query, top_k
            return list(candidates)

    llm = FakeLLM()
    answer = RagRerankPipeline(
        llm,
        RecordingRetriever(hits),
        GreedyReranker(),
        top_k=1,
        candidate_k=3,
    ).answer("pergunta")
    assert answer.contexts == ("a",)
    assert answer.chunk_ids == ("doc:0",)


def test_empty_retrieval_still_calls_the_llm():
    llm = FakeLLM()
    answer = NaiveRagPipeline(llm, RecordingRetriever([]), top_k=3).answer("pergunta")
    assert answer.contexts == ()
    assert answer.chunk_ids == ()
    assert "(nenhum trecho recuperado)" in llm.prompts[0]
