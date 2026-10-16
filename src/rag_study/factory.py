"""Monta provedores e as três pipelines a partir de Settings.

O índice é recriado a cada processo. Não há cache de embedding em disco
neste PR, então um demo offline com provider fake não chama rede.
"""

from rag_study.chunker import chunk_documents
from rag_study.config import Settings
from rag_study.embeddings import GeminiEmbedder, HashingEmbedder, OpenAIEmbedder
from rag_study.errors import ConfigError
from rag_study.llm import FakeLLM, GeminiLLM, OpenAILLM
from rag_study.loader import load_corpus
from rag_study.pipelines import NaiveRagPipeline, NoContextPipeline, RagRerankPipeline
from rag_study.protocols import LLM, Embedder
from rag_study.reranker import IdentityReranker
from rag_study.retriever import Retriever
from rag_study.vector_store import InMemoryVectorStore


def build_llm(settings: Settings) -> LLM:
    if settings.llm_provider == "fake":
        return FakeLLM()
    if settings.llm_provider == "gemini":
        return GeminiLLM(api_key=settings.gemini_api_key, model=settings.gemini_model)
    if settings.llm_provider == "openai":
        return OpenAILLM(api_key=settings.openai_api_key, model=settings.openai_model)
    raise ConfigError(f"LLM_PROVIDER inválida: {settings.llm_provider}.")


def build_embedder(settings: Settings) -> Embedder:
    if settings.embedding_provider == "fake":
        return HashingEmbedder()
    if settings.embedding_provider == "gemini":
        return GeminiEmbedder(
            api_key=settings.gemini_api_key,
            model=settings.gemini_embedding_model,
        )
    if settings.embedding_provider == "openai":
        return OpenAIEmbedder(
            api_key=settings.openai_api_key,
            model=settings.openai_embedding_model,
        )
    raise ConfigError(f"EMBEDDING_PROVIDER inválida: {settings.embedding_provider}.")


def generation_model_name(settings: Settings) -> str:
    """Nome cobrado na tabela de preço. fake não é um modelo de provedor."""
    if settings.llm_provider == "openai":
        return settings.openai_model
    if settings.llm_provider == "gemini":
        return settings.gemini_model
    return "fake"


def judge_model_name(settings: Settings) -> str:
    """Modelo do juiz. Vazio segue o modelo de geração do provedor escolhido."""
    if settings.judge_model:
        return settings.judge_model
    if settings.judge_provider == "openai":
        return settings.openai_model
    if settings.judge_provider == "gemini":
        return settings.gemini_model
    return "fake"


def build_judge(settings: Settings) -> LLM | None:
    """Juiz opcional. off não abre cliente. A chave continua só no ambiente."""
    provider = settings.judge_provider
    if provider == "off":
        return None
    model = judge_model_name(settings)
    if provider == "fake":
        return FakeLLM()
    if provider == "gemini":
        return GeminiLLM(api_key=settings.gemini_api_key, model=model)
    if provider == "openai":
        return OpenAILLM(api_key=settings.openai_api_key, model=model)
    raise ConfigError(f"JUDGE_PROVIDER inválida: {provider}.")


def build_pipelines(
    settings: Settings,
    llm: LLM | None = None,
) -> dict[str, NoContextPipeline | NaiveRagPipeline | RagRerankPipeline]:
    """Indexa o corpus e devolve no_context, naive_rag e rag_rerank.

    `llm` pronto deixa cache e retry do harness fora desta fábrica.
    """
    if llm is None:
        llm = build_llm(settings)
    embedder = build_embedder(settings)
    store = InMemoryVectorStore()
    documents = load_corpus(settings.corpus_path)
    chunks = chunk_documents(
        documents,
        settings.chunk_strategy,
        settings.chunk_size,
        settings.chunk_overlap,
    )
    if chunks:
        vectors = embedder.embed([chunk.text for chunk in chunks])
        store.add(list(zip(chunks, vectors, strict=True)))
    retriever = Retriever(store, embedder)
    return {
        "no_context": NoContextPipeline(llm),
        "naive_rag": NaiveRagPipeline(llm, retriever, settings.top_k),
        "rag_rerank": RagRerankPipeline(
            llm,
            retriever,
            IdentityReranker(),
            settings.top_k,
            settings.rerank_candidates,
        ),
    }
