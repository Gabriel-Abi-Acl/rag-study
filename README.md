# rag-study

Estudo técnico que compara três formas de um LLM responder perguntas sobre um corpus fechado: sem contexto, com RAG ingênuo e com RAG mais rerank. O corpus, o gabarito e o harness de métricas estão versionados. Este repositório ainda não tem números de uma execução real. O PR4 é quem publica correção, fidelidade, alucinação, custo e latência.

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

Hipótese a testar, ainda sem evidência neste repositório: num corpus fechado, o RAG ingênuo reduz alucinação factual em relação à resposta sem contexto, e um reranker melhora a precisão dos trechos usados em relação ao RAG ingênuo. Custo e latência devem subir com a recuperação e, de novo, com o rerank. Confirmar ou refutar isso é o trabalho do PR4. O dataset e o harness são o instrumento da medida; não são a medida.

O `IdentityReranker` não reordena. Ele não serve como evidência a favor ou contra a hipótese.

## Metodologia (planejada)

1. Corpus sintético versionado em `data/sample/`. Vila Aurora não é um município real; o texto foi escrito para este estudo.
2. 50 perguntas com gabarito em `data/eval/questions.v1.jsonl`. Sem gabarito não há correção nem alucinação comparável.
3. As três pipelines acima, com o mesmo modelo, a mesma temperatura (0) e o mesmo corpus.
4. Métricas do harness, definidas na seção Avaliação. Não há biblioteca RAGAS: o match e a abstenção são código, e o juiz opcional devolve JSON.
5. Repetições, tabela preenchida e gráficos só no PR4. Nenhum gráfico entra no README antes disso.

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
- `unanswerable` não cita documento. O gabarito é sempre `Não consta no corpus.` A pergunta parece local (PIB, UTI neonatal, CEP, xadrez, caixa d'água, ciclovia), mas o fato não foi escrito. A frase marca ausência de suporte. O harness compara a saída do modelo com essa abstenção; a regra está na seção Avaliação.

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

## Avaliação

O harness responde cada pergunta de uma pipeline e grava uma linha JSONL. Ele não escreve tabela no README. Números de qualidade, custo e latência de uma execução real só entram no PR4.

RAGAS não é dependência. A biblioteca puxa LangChain, não usa o `LLM` fake da CI e não controla o ritmo do tier gratuito do Gemini. O equivalente auditável está no código: match normalizado, detector de abstenção e dois prompts de juiz que pedem JSON (`Julgamento de correção` e `Julgamento de fidelidade`).

| métrica | definição neste harness |
| --- | --- |
| acerto | Nos tipos `factual`, `numeric` e `multi_hop`, o gabarito normalizado (a mesma função do dataset: caixa e espaços, sem reescrever `12.450`) precisa ocorrer na resposta. Com juiz ligado, o campo `correct` do JSON substitui esse match nesses três tipos. Em `unanswerable`, o acerto é 1 só se a resposta se abstém. O juiz não muda essa regra. |
| fidelidade | Fração das afirmações da resposta sustentadas pelo contexto recuperado, segundo o juiz. O prompt de fidelidade não recebe o gabarito. Sem juiz, o campo fica nulo. Contexto vazio não sustenta fato; abstenção sem fato extra vale 1. |
| alucinação | Entre as `unanswerable`, a fração que não se absteve. Não usa o juiz. Conta como abstenção: texto vazio, o gabarito `Não consta no corpus.`, ou uma recusa sem dígito e sem palavra que não esteja na pergunta nem numa lista fechada de termos de recusa. `Não há menção a ciclovia.` abstém se a pergunta já diz ciclovia. `Não sei. O PIB é 9 milhões.` não abstém. |
| recall@k | Documentos de `source_doc_ids` que aparecem em algum chunk recuperado, dividido pelo tamanho do ouro. O id do chunk é `doc_id:indice`. k é a quantidade de chunks da resposta. Ouro vazio (`unanswerable`) deixa o recall nulo. `no_context` não recupera, então o recall dos outros tipos é 0. |
| custo | Tokens da API. Se a resposta não trouxer uso, a estimativa é um token a cada quatro caracteres e o campo `tokens_estimated` fica verdadeiro. USD = tokens de entrada × preço de entrada + tokens de saída × preço de saída. A saída do Gemini inclui thinking tokens quando o total da API vem preenchido. A tabela é `data/eval/prices.v1.json`. Modelo ausente deixa `cost_usd` nulo. |
| latência | Tempo de parede de `pipeline.answer`, em milissegundos, incluindo a recuperação. Linha com `cached: true` não entra na média. |

A tabela de preço é a lista do tier pago Standard, só texto, consultada em outubro de 2026 na documentação do Gemini: `gemini-2.5-flash` a US$ 0.30 / US$ 2.50 por milhão de tokens de entrada e saída, e `gemini-2.5-flash-lite` a US$ 0.10 / US$ 0.40. Não é áudio nem batch. O tier gratuito cobra zero; o campo estima o volume como se fosse pago, para comparar pipelines. Outro modelo exige uma linha no JSON. Preço zero na tabela é escolha de quem edita o arquivo, não o padrão.

O cache fica em `EVAL_CACHE_DIR` (padrão `results/cache`). A chave é o SHA-256 do modelo e do prompt. Acerto de cache não chama a rede, não espera o intervalo e não conta como latência de provedor. O diretório `results/` está no `.gitignore`: pode conter resposta de modelo e não entra no git.

Entre chamadas de rede o runner espera `REQUEST_INTERVAL_S` (padrão 6, perto de 10 requisições por minuto). Em HTTP 429, `RESOURCE_EXHAUSTED`, 500, 503 ou cabeçalho `Retry-After`, tenta de novo com espera exponencial, até `LLM_MAX_RETRIES` (padrão 6). O mesmo ritmo vale para o juiz. O provedor do juiz é `JUDGE_PROVIDER` (`gemini` por padrão, ou `openai`, `fake`, `off`). `JUDGE_MODEL` vazio segue o modelo de geração desse provedor.

O corpus indexado é o do gabarito (`data/sample`, ou `--corpus`), não um `CORPUS_PATH` diferente.

Da raiz do repositório, com `GEMINI_API_KEY` só no ambiente ou no `.env`:

```bash
uv sync --extra gemini
uv run python -m rag_study.run --pipeline no_context --out results/no_context.jsonl
uv run python -m rag_study.run --pipeline naive_rag --out results/naive_rag.jsonl
uv run python -m rag_study.run --pipeline rag_rerank --out results/rag_rerank.jsonl
uv run python -m rag_study.aggregate results/no_context.jsonl results/naive_rag.jsonl results/rag_rerank.jsonl --md results/summary.md --csv results/summary.csv
```

A agregação imprime uma tabela e, com `--md` e `--csv`, grava os dois arquivos. Há uma linha por pipeline e tipo presente (`factual`, `numeric`, `multi_hop`, `unanswerable`) e uma linha `all`. Colunas: n, acerto, fidelidade, alucinação, recall@k, latência média sem cache, soma dos tokens de resposta e de juiz, soma de custo e quantidade de linhas servidas do cache. Célula vazia significa métrica indefinida naquele recorte. Custo nulo não vira zero: a soma só aparece se alguma linha tinha preço. Pergunta repetida na mesma pipeline aborta a agregação. Um JSONL interrompido por erro de provedor pode ficar pela metade; não serve como experimento completo.

Fumaça offline, sem chave e sem medir qualidade:

```bash
LLM_PROVIDER=fake EMBEDDING_PROVIDER=fake uv run python -m rag_study.run \
  --pipeline naive_rag --out results/smoke.jsonl --judge off --interval 0 --limit 2
```

`--limit` corta o gabarito na ordem do arquivo. Não é uma amostra do estudo. `fake` não está na tabela de preço, então `cost_usd` sai nulo.

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

O comando do dataset só valida o arquivo. A execução com gabarito e métricas está na seção Avaliação.

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

O gabarito não entra no prompt. O harness lê o JSONL, chama a pipeline e grava o JSONL de saída.

| módulo | papel |
| --- | --- |
| `rag_study.llm` / `rag_study.embeddings` | Gemini, OpenAI ou fake. O SDK só é importado na primeira chamada. |
| `rag_study.loader` | `.txt` e `.md` |
| `rag_study.dataset` | schema do gabarito, carga do JSONL e validação offline |
| `rag_study.metrics` | match normalizado, abstenção, alucinação e recall@k |
| `rag_study.judge` | prompts de correção e fidelidade; a resposta do juiz é JSON |
| `rag_study.usage` | tokens do provedor, estimativa e custo pela tabela |
| `rag_study.cache` | cache em disco, intervalo e nova tentativa em 429 |
| `rag_study.run` | CLI que grava uma linha por pergunta |
| `rag_study.aggregate` | tabela markdown e CSV a partir dos JSONL |
| `rag_study.chunker` | `fixed` e `recursive` |
| `rag_study.vector_store` | cosseno em memória |
| `rag_study.retriever` | embedding da pergunta + `top_k` |
| `rag_study.reranker` | interface; `IdentityReranker` neste repositório |
| `rag_study.pipelines` | `no_context`, `naive_rag`, `rag_rerank` |
| `rag_study.config` | variáveis do `.env.example` |

## Roadmap

- **PR2 — dataset (feito).** Corpus sintético em `data/sample/` e gabarito em `data/eval/questions.v1.jsonl`. Validação offline de schema, citação e apoio literal. Sem resposta de modelo.
- **PR3 — avaliação (feito).** Harness das métricas (correção, fidelidade, alucinação, recall, custo, latência), com cache, retry e agregação. Sem execução real gravada e sem tabela preenchida.
- **PR4 — experimentos.** Rodar as três condições, gráficos e relatório no README. Só então entram números.

## O que ainda não está feito

Não há JSONL de execução no git, não há tabela preenchida e não há gráfico. O gabarito não guarda resposta de modelo. Qualquer número de qualidade citado antes do PR4 está fora deste estudo.
