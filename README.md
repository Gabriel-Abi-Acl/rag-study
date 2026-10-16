# rag-study

Estudo técnico que compara três formas de um LLM responder perguntas sobre um corpus fechado: sem contexto, com RAG ingênuo e com RAG mais rerank. O corpus e o gabarito estão versionados. Este repositório ainda não tem números de qualidade, custo ou latência. Os PRs seguintes medem correção, fidelidade ao contexto, taxa de alucinação, custo e latência.

O corpus é sintético (município fictício de Vila Aurora). Não há chave de API no git.

## Objetivo

Medir, no mesmo conjunto de perguntas com gabarito, a diferença entre:

| pipeline | o que entra no prompt |
| --- | --- |
| `no_context` | só a pergunta |
| `naive_rag` | pergunta mais os `top_k` trechos do índice denso |
| `rag_rerank` | pergunta mais os `top_k` trechos depois de um reranker |

Provedor padrão de geração e de embedding: Gemini, pelo tier gratuito. OpenAI é opcional, pela mesma interface. Teste e CI usam `LLM_PROVIDER=fake` e `EMBEDDING_PROVIDER=fake` e não abrem rede.

## Hipótese

Hipótese a testar, ainda sem evidência neste repositório: num corpus fechado, o RAG ingênuo reduz alucinação factual em relação à resposta sem contexto, e um reranker melhora a precisão dos trechos usados em relação ao RAG ingênuo. Custo e latência devem subir com a recuperação e, de novo, com o rerank. Confirmar ou refutar isso é o trabalho dos PRs 3 e 4. O dataset é o instrumento da medida; não é a medida.

O `IdentityReranker` não reordena. Ele não serve como evidência a favor ou contra a hipótese.

## Metodologia (planejada)

1. Corpus sintético versionado em `data/sample/`. Vila Aurora não é um município real; o texto foi escrito para este estudo.
2. 50 perguntas com gabarito em `data/eval/questions.v1.jsonl`. Sem gabarito não há correção nem alucinação comparável. Este PR não gera resposta de modelo.
3. As três pipelines acima, com o mesmo modelo, a mesma temperatura (0) e o mesmo corpus.
4. Métricas (PR3), por RAGAS ou implementação equivalente e auditável:
   - answer correctness
   - faithfulness / groundedness
   - taxa de alucinação
   - custo (tokens ou preço de tabela) e latência
5. Repetições, tabela e gráficos só depois que o harness existir (PR4). Nenhum gráfico entra no README antes disso.

Decisões já fixadas no código, de propósito limitadas:

- Índice local em memória, similaridade de cosseno. A classe `InMemoryVectorStore` é o lugar de um adapter Chroma ou FAISS se o corpus deixar de caber em RAM.
- Chunking `fixed` (janela de caracteres com overlap) e `recursive` (parágrafo, linha, frase, palavra, depois caractere).
- Rerank é um protocolo. A implementação atual devolve os candidatos na ordem do retriever, cortados em `top_k`.
- Embeddings de hash (`fake`) só existem para teste e demo offline. Não entram na comparação publicada.

## Dataset

O corpus de avaliação fica em `data/sample/` (24 arquivos `.txt` e `.md`). O `doc_id` é o caminho relativo que `load_corpus` já usa, por exemplo `governo.txt`. O gabarito versionado é `data/eval/questions.v1.jsonl`: a versão está no nome do arquivo (`v1`), não num campo solto. Cada linha é um objeto com cinco campos:

| campo | papel |
| --- | --- |
| `id` | identificador único, `q001` a `q050` |
| `question` | pergunta em português |
| `reference_answer` | gabarito |
| `source_doc_ids` | documentos que sustentam a resposta, ou lista vazia |
| `type` | `factual`, `numeric`, `multi_hop` ou `unanswerable` |

Contagem desta versão, propriedade do arquivo e não resultado de modelo: 18 factuais, 14 numéricas, 12 multi-hop e 6 sem resposta no corpus.

O corpus é sintético de propósito. Fato que já esteja no pré-treino deixa o baseline `no_context` acertar por memória, e a comparação com RAG deixa de isolar o efeito do contexto. Os textos têm tamanho desigual: fichas curtas (algumas com menos de 500 caracteres), textos médios e duas peças longas (`historia.txt`, `ata_camara.txt`).

Regras de tipo:

- `factual` e `numeric` citam um único documento. O gabarito é um trecho literal desse documento. A checagem normaliza caixa e espaços e exige a substring. Número entra com a grafia da ficha (`12.450`, `R$ 4,50`, `860 metros`), sem converter milhar nem trocar vírgula por ponto.
- `multi_hop` cita dois ou mais documentos. O gabarito não cabe inteiro em cada um deles. Em parte dos itens a resposta é um nome que está num arquivo e o elo está no outro (o conselho diz "prefeita eleita em 2024"; o governo nomeia Helena Duarte). Em outros, a frase do gabarito junta fatos que não aparecem juntos (tarifa rejeitada na ata e tarifa vigente no transporte).
- `unanswerable` não cita documento. O gabarito é sempre `Não consta no corpus.` A pergunta parece local (PIB, UTI neonatal, CEP, xadrez, caixa d'água, ciclovia), mas o fato não foi escrito. A frase marca ausência de suporte. Este PR não define como uma métrica futura vai comparar a saída do modelo com ela.

Distratores, para a recuperação não ter só a frase certa:

- `serra_verde.txt` descreve outro município fictício, com números e nomes próximos (população `12.540`, prefeita Helena Martins, biblioteca em 1979, tarifa `R$ 4,80`). Não entra em `source_doc_ids`.
- `blog_biblioteca.md` repete o parágrafo de endereço e empréstimo da ficha da biblioteca e troca o ano oficial `1978` por `1981`, e o acervo `18.600` por "cerca de 20 mil". A pergunta do ano oficial cita `biblioteca.md`; a pergunta sobre o relato cita o blog.
- `rumor_populacao.md` afirma `20.000` habitantes e não repete a estimativa oficial `12.450`.
- `ata_camara.txt` copia a frase cadastral da coleta reciclável e registra proposta rejeitada de tarifa `R$ 5,00`. A tarifa vigente `R$ 4,50` está em `transporte.md`.

O mesmo ano da biblioteca aparece duas vezes de propósito: `q019` pergunta direto a ficha (`numeric`) e `q035` exige juntar ficha, conselho e nome da prefeita (`multi_hop`).

Schema e validação estão em `rag_study.dataset` (dataclasses, o mesmo estilo do resto do pacote). O comando não chama rede nem modelo:

```bash
uv run python -m rag_study.dataset
```

Ele confere o nome versionado do arquivo, o schema de cada linha, ids únicos, `source_doc_ids` existentes no corpus, apoio literal das respostas factuais e numéricas, a regra de multi-hop, o gabarito fixo das perguntas sem resposta no corpus e a faixa de tamanho (cerca de 50 perguntas, 15 a 30 documentos, os quatro tipos presentes). Sai com código 1 se algo falhar. Os mesmos invariantes estão em `tests/test_eval_dataset.py`.

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

O comando indexa `CORPUS_PATH` (padrão `data/sample`, o corpus do estudo) e imprime a resposta de cada pipeline. Isso não é uma avaliação: não compara com o gabarito, não calcula métrica e o rerank ainda é identidade.

Offline, sem chave:

```bash
LLM_PROVIDER=fake EMBEDDING_PROVIDER=fake uv run python -m rag_study
```

Validar o dataset, também offline:

```bash
uv run python -m rag_study.dataset
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

O gabarito não entra nesse caminho. Ele só existe para a avaliação futura.

| módulo | papel |
| --- | --- |
| `rag_study.llm` / `rag_study.embeddings` | Gemini, OpenAI ou fake. O SDK só é importado na primeira chamada. |
| `rag_study.loader` | `.txt` e `.md` |
| `rag_study.dataset` | schema do gabarito, carga do JSONL e validação offline |
| `rag_study.chunker` | `fixed` e `recursive` |
| `rag_study.vector_store` | cosseno em memória |
| `rag_study.retriever` | embedding da pergunta + `top_k` |
| `rag_study.reranker` | interface; `IdentityReranker` neste repositório |
| `rag_study.pipelines` | `no_context`, `naive_rag`, `rag_rerank` |
| `rag_study.config` | variáveis do `.env.example` |

## Roadmap

- **PR2 — dataset (feito).** Corpus sintético em `data/sample/` e gabarito em `data/eval/questions.v1.jsonl`. Validação offline de schema, citação e apoio literal. Sem resposta de modelo.
- **PR3 — avaliação.** Harness das métricas (correção, fidelidade, alucinação, custo, latência), ainda sem tabela de experimento fechada.
- **PR4 — experimentos.** Rodar as três condições, gráficos e relatório no README. Só então entram números.

## O que ainda não está feito

Não há harness de métricas, não há resposta de modelo gravada no gabarito e não há tabela experimental. Qualquer número de qualidade citado antes do PR4 está fora deste estudo.
