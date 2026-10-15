# rag-study

Estudo técnico que compara três formas de um LLM responder perguntas sobre um corpus fechado: sem contexto, com RAG ingênuo e com RAG mais rerank. Este repositório ainda não tem números. Os PRs seguintes medem correção, fidelidade ao contexto, taxa de alucinação, custo e latência.

O corpus de exemplo é sintético (município fictício de Vila Aurora). Não há chave de API no git.

## Objetivo

Medir, no mesmo conjunto de perguntas com gabarito, a diferença entre:

| pipeline | o que entra no prompt |
| --- | --- |
| `no_context` | só a pergunta |
| `naive_rag` | pergunta mais os `top_k` trechos do índice denso |
| `rag_rerank` | pergunta mais os `top_k` trechos depois de um reranker |

Provedor padrão de geração e de embedding: Gemini, pelo tier gratuito. OpenAI é opcional, pela mesma interface. Teste e CI usam `LLM_PROVIDER=fake` e `EMBEDDING_PROVIDER=fake` e não abrem rede.

## Hipótese

Hipótese a testar, ainda sem evidência neste repositório: num corpus fechado, o RAG ingênuo reduz alucinação factual em relação à resposta sem contexto, e um reranker melhora a precisão dos trechos usados em relação ao RAG ingênuo. Custo e latência devem subir com a recuperação e, de novo, com o rerank. Confirmar ou refutar isso é o trabalho dos PRs 2 a 4.

O `IdentityReranker` deste PR não reordena. Ele não serve como evidência a favor ou contra a hipótese.

## Metodologia (planejada)

1. Corpus público ou sintético, versionado. O placeholder atual está em `data/sample/`.
2. Perguntas com gabarito e trechos de apoio (PR2). Sem gabarito não há correção nem alucinação comparável.
3. As três pipelines acima, com o mesmo modelo, a mesma temperatura (0) e o mesmo corpus.
4. Métricas (PR3), por RAGAS ou implementação equivalente e auditável:
   - answer correctness
   - faithfulness / groundedness
   - taxa de alucinação
   - custo (tokens ou preço de tabela) e latência
5. Repetições, tabela e gráficos só depois que o harness existir (PR4). Nenhum gráfico entra no README antes disso.

Decisões deste PR, de propósito limitadas:

- Índice local em memória, similaridade de cosseno. A classe `InMemoryVectorStore` é o lugar de um adapter Chroma ou FAISS se o corpus deixar de caber em RAM.
- Chunking `fixed` (janela de caracteres com overlap) e `recursive` (parágrafo, linha, frase, palavra, depois caractere).
- Rerank é um protocolo. A implementação atual devolve os candidatos na ordem do retriever, cortados em `top_k`.
- Embeddings de hash (`fake`) só existem para teste e demo offline. Não entram na comparação publicada.

## Como rodar

Requisito: Python 3.11+ e [uv](https://docs.astral.sh/uv/).

```bash
uv sync --extra gemini
cp .env.example .env
```

Preencha `GEMINI_API_KEY` no `.env`. Para OpenAI, use `uv sync --extra openai` e `LLM_PROVIDER=openai`.

Na raiz do repositório:

```bash
uv run python -m rag_study "Qual é a população de Vila Aurora?"
```

O comando indexa `CORPUS_PATH` (padrão `data/sample`) e imprime a resposta de cada pipeline. Isso não é uma avaliação: não há gabarito, não há métrica e o rerank ainda é identidade.

Offline, sem chave:

```bash
LLM_PROVIDER=fake EMBEDDING_PROVIDER=fake uv run python -m rag_study
```

Testes e lint, os mesmos da CI:

```bash
uv sync
uv run ruff check .
uv run ruff format --check .
uv run pytest
```

## Arquitetura

```
loader → chunker → embedder → InMemoryVectorStore
                                    ↓
pergunta → Retriever → (Reranker) → LLM → Answer
```

| módulo | papel |
| --- | --- |
| `rag_study.llm` / `rag_study.embeddings` | Gemini, OpenAI ou fake. O SDK só é importado na primeira chamada. |
| `rag_study.loader` | `.txt` e `.md` |
| `rag_study.chunker` | `fixed` e `recursive` |
| `rag_study.vector_store` | cosseno em memória |
| `rag_study.retriever` | embedding da pergunta + `top_k` |
| `rag_study.reranker` | interface; `IdentityReranker` neste PR |
| `rag_study.pipelines` | `no_context`, `naive_rag`, `rag_rerank` |
| `rag_study.config` | variáveis do `.env.example` |

## Roadmap

- **PR2 — dataset.** Perguntas com gabarito e trechos relevantes, em cima de corpus público ou sintético. Sem resposta de modelo neste PR.
- **PR3 — avaliação.** Harness das métricas (correção, fidelidade, alucinação, custo, latência), ainda sem tabela de experimento fechada.
- **PR4 — experimentos.** Rodar as três condições, gráficos e relatório no README. Só então entram números.

## O que este PR não faz

Não roda juiz, não estima alucinação e não compara provedores. Qualquer número de qualidade citado antes do PR4 está fora deste estudo.
