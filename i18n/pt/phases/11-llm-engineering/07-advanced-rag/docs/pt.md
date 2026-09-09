# RAG avançado (Combinação, Relançamento, Pesquisa híbrida)

> RAG básico recupera os pedaços mais semelhantes. Isso funciona para perguntas simples. Desmorona para raciocínio multi-hop, consultas ambíguas e grandes corporações. RAG avançado é a diferença entre uma demonstração que funciona em 10 documentos e um sistema que funciona em 10 milhões.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 11, Lesson 06 (RAG)
**Time:** ~90 minutes
**Related:**A fase 5 · 23 (Estratégias de Chunking para RAG) abrange todos os seis algoritmos de chunking  recursivo, semântico, frase, documento-mãe, chunking tardia, recuperação contextual  com benchmarks Vectara/Antropic. Esta lição se baseia em cima: busca híbrida, re-ranqueamento, transformação de consulta.

## Objetivos de aprendizagem

- Implementar estratégias avançadas de fragmentação (semântica, recursiva, pai/filho) que preservem a estrutura e o contexto do documento
- Construir um pipeline de pesquisa híbrida combinando a combinação de palavras-chave BM25 com pesquisa semântica vetorial e um re-ranqueador de codificação cruzada
- Aplicar técnicas de transformação de consultas (HyDE, multi-query, step-back) para melhorar a recuperação em questões ambíguas ou complexas
- Diagnóstico e correção de falhas comuns do RAG: fragmento errado recuperado, resposta não contextual, desvio de raciocínio multi-hop

## O problema

Construíste um sistema básico de RAG na lição 06. Funciona para perguntas simples num pequeno corpo.

**Ambiguous query**A pesquisa semântica retorna pedaços sobre a estratégia de receita, as projeções de receita e os pensamentos do CFO sobre o crescimento da receita. Tudo semânticamente semelhante à palavra "receita". Nenhum contém o número real. A peça correta diz "$47.2M in Q3 2025" but uses the word "earnings" instead of "revenue." The embedding model thinks "revenue strategy" is closer to the query than "Q3 earnings were $47,2 M".

**Multi-hop question**A resposta é: "Qual equipe teve a maior melhoria na pontuação de satisfação do cliente?" Isso requer encontrar as pontuações de satisfação de cada equipe, compará-las e identificar o máximo. Nenhuma peça única contém a resposta.

**Large corpus problem**Você tem 2 milhões de blocos. A resposta correta é no bloco # 1,847,293. Sua busca no top 5 tira blocos # 14, # 89,201, # 1,200,000, # 44, e # 901,333. Fechado em espaço de inserção, mas nenhum contendo a resposta. Nesta escala, a pesquisa próxima aproximada introduz erro suficiente para que os resultados relevantes sejam empurrados para fora do top-k.

O RAG básico falha porque a semelhança vectorial não é o mesmo que a relevância. Um pedaço pode ser semânticamente semelhante a uma consulta sem ser útil para responder a ela. O RAG avançado aborda isso com quatro técnicas: pesquisa híbrida (aditar a correspondência de palavras-chave), re-ranqueamento (pontuar os candidatos com mais cuidado), transformação de consulta (fixar a consulta antes de pesquisar) e melhor fragmentação (recuperar na granularidade certa).

## O conceito

### Pesquisa híbrida: Semântica + Palavra-chave

A pesquisa semântica (semelhança vectorial) é boa para entender o significado. "Como cancelar minha assinatura?" coincide com "Pasos para rescindir seu plano", embora eles não compartilhem palavras. Mas não tem correspondências exatas. "Code de erro E-4021" pode não corresponder a um pedaço contendo "E-4021" se o modelo de incorporação o tratar como ruído.

Busca de palavras-chave (BM25) é o oposto. Excelente em correspondências exatas. "E-4021" combina perfeitamente. Mas "cancelar minha assinatura" retorna resultados zero se o documento diz "terminar o seu plano".

A busca híbrida executa ambas as coisas, e depois mistura os resultados.

#### BM25 Mergulho profundo

**BM25**(Best Matching 25) é o algoritmo padrão de pesquisa de palavras-chave. Tem sido a espinha dorsal dos motores de pesquisa (como Elasticsearch e Lucene) desde a década de 1990. É uma atualização evolutiva para o algoritmo TF-IDF clássico.

Se quiser uma rápida compreensão conceitual do funcionamento do BM25, pode concentrar-se em**three core pillars**(sem ficar preso pelas fórmulas matemáticas):

1. **Term Rarity (IDF: The rarer the term, the higher its weight)**- Não .
   - A ideia principal é "raridade é igual a relevância". Em uma consulta, termos comuns como "o", "como", "para" aparecem em quase todos os documentos e não ajudam a identificar o documento correto. Por outro lado, termos raros como "E-4021" (um código de erro específico) ou "alucinação" (um tópico específico) são altamente seletivos. BM25 atribui automaticamente um alto peso a termos raros e descontos ou ignora os comuns.
2. **Term Frequency Saturation (TF Saturation: Repeated hits have diminishing returns, with a ceiling)**- Não .
   - A ideia principal é evitar "enchimento de palavras-chave". Mencionar um termo 1 vez é útil, 2 vezes é ligeiramente mais útil, mas mencionar 100 vezes não torna um documento 100 vezes mais relevante. BM25 estabelece um limite máximo sobre a contribuição de frequência do termo (controlada por parâmetro $k_1$O termo aparece mais vezes num documento, e a sua contribuição incremental de pontuação saturou-se rapidamente e desabou.
3. **Document Length Normalization (L-Norm: Matching in short documents is more significant)**- Não .
   - A ideia principal é penalizar documentos inchados e "abaterados". Se um título curto de 10 palavras coincide com uma palavra-chave, e um whitepaper abrangente de 10.000 palavras coincide com a mesma palavra-chave uma vez, o título curto é altamente provável que seja totalmente dedicado a essa palavra-chave. BM25 penaliza documentos com base em seu comprimento relativo (controlado por parâmetro $b$), aumentando os documentos mais curtos e diminuindo os resultados para os documentos extremamente longos.

> Não, não.
> Para obter fórmulas matemáticas detalhadas, derivadas de suavização das Forças Armadas, controles de parâmetros e um exemplo numérico calculado à mão passo a passo, consulte a página dedicada: [BM25 Mathematical Details & Code Alignment](bm25_math_en.md)- Não .


### Fusão de grau recíproco (RRF)

Em grandes arquiteturas RAG ou de busca híbrida, **RRF (Reciprocal Rank Fusion)**É o algoritmo mais amplamente adotado e elegante para combinar listas classificadas.

Simplificando, a lógica central da RRF é: **disregard raw scores, focus only on ranks**- Não .

#### 1. Por que precisamos de RRF?
Ao construir um sistema de recuperação híbrida, usamos tipicamente:
- **Semantic Retrieval (Vector Search)**: Os resultados são geralmente de similaridade cosínea ou produto de pontos, normalmente variando em$[-1, 1]$ou $[0, 1]$- Não .
- **Keyword Retrieval (BM25)**: Os pontos são números positivos sem limites de pontos flutuantes (dependendo da frequência do termo e do comprimento do documento).

Somando diretamente essas duas pontuações é matematicamente incorreto porque suas escalas, distribuições e significados físicos são completamente diferentes. RRF contorna esse problema de alinhamento, concentrando-se apenas na "posição relativa de um documento em cada lista", padronizando as saídas de diferentes sistemas de recuperação em uma única dimensão.

#### 2. Mergulhe profundamente na fórmula

Para um documento de candidato $d$, a sua pontuação RRF é calculada da seguinte forma:

$$RRF\_score(d) = \sum_{R \in Rankings} \frac{1}{k + rank_R(d)}$$

- **$rank_R(d)$**: The rank of document $d$ in the $R$-th retrieval system (1st place is $1$, 10th place is $10$). If the document is absent from a retrieval list, the term is treated as $0$.
- **$k$ (smoothing factor, usually defaults to $60$)**: Prevents top-ranked results from dominating the final score, ensuring ranking robustness.
  - **Without $k$**: The 1st rank gets a weight of $\frac{1}{1} = 1$, while the 2nd rank drops to $\frac{1}{2} = 0.5$. This massive gap allows a "single-list winner" to dominate the combined list.
  - **With $k$**: The 1st rank gets $\frac{1}{60 + 1} \approx 0.0164$, and the 2nd rank gets $\frac{1}{60 + 2} \approx 0.0161$. The gap between top ranks is smoothed, making the fusion of multiple retrieval paths more balanced and robust.

> [!NOTE]
> **RRF Fusion Calculation & Ranking Example:**
> - **Document A**: Ranked 3rd in vector search and 2nd in BM25. Its score is:
>   $$Score(A) = \frac{1}{60 + 3} + \frac{1}{60 + 2} \approx 0.0159 + 0.0161 = 0.0320$$
> - **Document B**: Ranked 1st in vector search but not retrieved by BM25. Its score is:
>   $$Score(B) = \frac{1}{60 + 1} + 0 \approx 0.0164$$
>
> Final ranking: **Document A ($0.0320$) > Document B ($0.0164$)**.

#### 3. Core Philosophy of RRF: Consensus as Truth
RRF aligns perfectly with our retrieval intuition:
- **"Mutual Agreement" beats "Single-Index Winner"**: As shown above, Document A, which ranks highly in both systems, outperforms Document B, which won 1st place in one list but was completely missing in the other.
- **Mitigating Outliers (Noise)**: RRF suppresses false positives from individual retrieval algorithms. If a document ranks 1st in BM25 purely due to a keyword coincidence but is irrelevant to the semantic context (ranking extremely low or absent in vector search), RRF ensures it ranks below documents with solid agreement across both indices.

#### 4. Real-world Engineering Application: Multimodal Video ETL Pipelines
In modern LLM and multimodal systems, RRF is highly effective for processing heterogeneous multi-path data. For example, in a **multimodal video understanding and ETL (Extract, Transform, Load) pipeline**:
- We can leverage models like **Gemini 3.5 Flash** to extract multimodal features from videos and build three independent search indices:
  1. **ASR (Speech-to-Text) Index**: For matching spoken keywords (best suited for BM25 keyword search).
  2. **Visual Scene Semantic Index**: For retrieving visual concepts or specific frames (best suited for multimodal vector search).
  3. **Video Metadata Index**: Containing structured fields like timestamps, categories, or tags (best suited for structured metadata filtering).
- RRF combines ASR retrieval, visual vector search, and metadata query ranks with minimal computational overhead—completely bypassing the need to train or tune expensive Cross-Encoder models—delivering high-recall, high-precision video clip localization.

### Reranking

Retrieval (whether vector, keyword, or hybrid) is fast but imprecise. It uses bi-encoders: the query and each document are embedded independently, then compared. The embeddings are computed once and cached. This scales to millions of documents.

Reranking uses cross-encoders: the query and a candidate document are fed together into a model that outputs a relevance score. The model sees both texts simultaneously and can capture fine-grained interactions between them. A cross-encoder can understand that "What were Q3 earnings?" is highly relevant to a chunk containing "$47.2M in Q3" even if a bi-encoder missed the connection.

The trade-off: cross-encoders are 100-1000x slower than bi-encoders because they process the query-document pair jointly. You cannot pre-compute cross-encoder scores for a million documents. The solution: retrieve a larger candidate set (top-50 from hybrid search), then rerank with a cross-encoder to get the final top-5.

```mermaid
graph LR
    Q["Query"] --> H["Hybrid Search"]
    H --> C50["Top 50 candidates"]
    C50 --> RR["Cross-Encoder Reranker"]
    RR --> C5["Top 5 final results"]
    C5 --> P["Build prompt"]
    P --> LLM["Generate answer"]
```

Common reranking models (2026 lineup):
- Cohere Rerank 3.5: managed API, multilingual, best recall gain on mixed corpora
- Voyage rerank-2.5: managed API, lowest latency of the hosted options
- Jina-Reranker-v2 Multilingual: open-weight, 100+ languages
- bge-reranker-v2-m3: open-weight, strong baseline
- cross-encoder/ms-marco-MiniLM-L-6-v2: open-weight, runs on CPU for prototyping
- ColBERTv2 / Jina-ColBERT-v2: late-interaction multi-vector rerankers — O(tokens) not O(docs) at scoring time

### Query Transformation

Sometimes the failure in a RAG system is not due to the retrieval algorithm, but the quality of the user's query itself. A query like `"What was that thing about the new policy change?"` contains no specific entities or keywords, making its embedding representation in the semantic space extremely vague and difficult for any retrieval system to match accurately.

Query transformation leverages an LLM to reformulate, enrich, or rewrite the raw query before initiating the search. The two primary strategies for query transformation are:

#### 1. Query Rewriting: Intent Standardization
Query rewriting focuses on removing colloquial noise from the user's input or resolving context gaps in multi-turn dialogues to sharpen the search intent.

- **The Core Pain Point**: Conversational queries often contain conversational clutter (e.g., "Please tell me," "Can you check," "I want to know about..."). These words dilute the attention weights of key concepts when converted into embeddings, degrading retrieval quality.
- **LLM Solution**: By using a few-shot prompt, the LLM can strip out conversational filler and extract the raw search concepts.
- **Coreference Resolution in Multi-turn Dialogue**: In conversational RAG, users frequently use pronouns or fragments. For example, after asking: "*What is the refund policy for Acme Enterprise?*", a user might ask: "*And what about its processing time?*". Searching with the latter fragment directly would fail. An LLM rewrite resolves this coreference, combining context to output: "*The processing time for Acme Enterprise refund policy*".

```
Colloquial query: "What was that thing about the new policy change?"
LLM Rewrite: "Recent policy changes and updates"
```

#### 2. HyDE (Hypothetical Document Embeddings): Bridging the Asymmetric Gap
HyDE is a highly innovative retrieval paradigm that completely rethinks semantic search.

- **The Core Pain Point (Asymmetric Retrieval)**: In embedding spaces, **questions and answers are linguistically asymmetric**. Questions are short, interrogative, and filled with question words; answers are longer, declarative, and rich in domain-specific terminology. Consequently, a question embedding and an answer embedding can live far apart. For instance, searching for *"What is a distributed lock?"* might retrieve other question chunks containing "What is..." rather than the actual implementation details of distributed locks.
- **LLM Solution (HyDE)**:
  1. **Generate Hypothesis**: Before retrieval, call the LLM to generate a hypothetical answer to the query without any external context (even if this draft contains hallucinations or incorrect metrics, it is perfectly fine).
  2. **Embed the Hypothesis**: Convert this "fake answer" into an embedding vector.
  3. **Search Real Documents**: Use the hypothesis vector to query the database for the most similar "real documents".
- **Why it Works**: The generated "fake answer" and the "real document" share the same declarative style, vocabulary, sentence structures, and domain jargon. In the embedding space, the distance between **"Answer - Answer" is much smaller than the distance between "Question - Answer"**.

```
Query: "What is the refund policy for enterprise?"
Hypothetical Answer (Hallucinated draft): "Enterprise customers are eligible for a full refund within 60 days of purchase. Refunds are pro-rated based on the remaining subscription period and processed within 5-7 business days."
(The system embeds this hypothesis to retrieve similar real docs)
```

- **The Latency Trade-off**: HyDE introduces an extra LLM call prior to retrieval, adding 500ms to 2000ms of latency depending on the model. In production, it is typically reserved for queries that are highly abstract, short, or suffer from extreme asymmetry.

### Parent-Child Chunking

In production environments, traditional "fixed-size chunking" (e.g., splitting texts uniformly into 300 tokens) often hits a **dilemma where you cannot optimize for both retrieval and synthesis**. Parent-child chunking elegantly resolves this by **decoupling the retrieval granularity from the synthesis (inference) granularity**.

#### 1. The Dilemma of Traditional Chunking
When building indices for long documents or metadata, you must choose between:
- **Small chunks (e.g., 100 tokens)**: The vector representations are highly precise and excel at matching specific user queries (e.g., specific part numbers or contract clauses). However, because they lack surrounding context, the LLM receives isolated text fragments, often leading to out-of-context reasoning or hallucinations.
- **Large chunks (e.g., 1000 tokens)**: The context is rich enough for synthesis, but the core features are diluted by surrounding conversational noise. This dilution makes it significantly harder for vector databases to rank the correct document highly, dropping search recall.

#### 2. The Solution: Separating "Search" from "Read"
The philosophy of parent-child chunking is: **"Search with a needle, read with a net."**

In the database design, data is structured into two hierarchical layers:
1. **Child Chunks (fine-grained)**: e.g., 128 tokens. These are embedded and stored in the vector index. Because they are short and semantically pure, they act as sensitive sensors that respond to specific details in user queries.
2. **Parent Chunks (coarse-grained)**: e.g., 512 or 1024 tokens (which contain the child chunks). These do not participate in vector similarity search. Instead, they are stored in a raw text format within a document store (e.g., a Key-Value database or relational database) and mapped to child chunks via a `Parent_ID` field.

```mermaid
graph TD
    P["Parent Chunk (512 tokens)<br/>Full section on refund policy"]
    C1["Child Chunk (128 tokens)<br/>Standard plan: 30-day refund"]
    C2["Child Chunk (128 tokens)<br/>Enterprise: 60-day pro-rated refund"]
    C3["Child Chunk (128 tokens)<br/>Processing time: 5-7 days"]
    C4["Child Chunk (128 tokens)<br/>How to submit request"]

    P --> C1
    P --> C2
    P --> C3
    P --> C4

    Q["Query: enterprise refund?"] -.->|"Matches child chunk"| C2
    C2 -.->|"Returns parent chunk"| P
```

#### 3. Execution Workflow
- **Step 1 (Search)**: The user issues a query. The vector database matches and retrieves the relevant **child chunk**.
- **Step 2 (Fetch)**: The backend uses the `Parent_ID` of the matched child chunk to retrieve the corresponding full **parent chunk** from the document store.
- **Step 3 (Read)**: The full text of the parent chunk is injected into the LLM prompt as context, allowing the LLM to generate precise, grounded answers in a complete context with full reasoning capability.

### Metadata Filtering

Before running vector search, filter the corpus by metadata: date, source, category, author, language. This reduces the search space and prevents irrelevant results.

"What changed in the security policy last month?" should only search documents from the last 30 days in the security category. Without metadata filtering, you search the entire corpus and might retrieve a 2-year-old security document that happens to be semantically similar.

Production RAG systems store metadata alongside each chunk: source document, creation date, category, author, version. Vector databases support pre-filtering by metadata before similarity search, which is critical for performance at scale.

### Evaluation

You built a RAG system. How do you know if it works? Three metrics:

**Retrieval relevance (Recall@k)**: for a set of test questions with known relevant documents, what percentage of relevant documents appear in the top-k results? If the answer to a question is in chunk #47, does chunk #47 appear in the top-5?

**Faithfulness**: is the generated answer grounded in the retrieved documents? If the retrieved chunks say "60-day refund window" and the model says "90-day refund window," that is a faithfulness failure. The model hallucinated despite having the correct context.

**Answer correctness**: does the generated answer match the expected answer? This is the end-to-end metric. It combines retrieval quality and generation quality.

A simple faithfulness check: take each claim in the generated answer and verify it appears (in substance) in the retrieved chunks. If the answer contains a fact not in any retrieved chunk, it is likely hallucinated.

```mermaid
graph TD
    subgraph "Evaluation Framework"
        Q["Test questions<br/>+ expected answers<br/>+ relevant doc IDs"]
        Q --> Ret["Retrieval evaluation<br/>Recall@k: are right<br/>docs retrieved?"]
        Q --> Faith["Faithfulness evaluation<br/>Is answer grounded<br/>in retrieved docs?"]
        Q --> Correct["Correctness evaluation<br/>Does answer match<br/>expected answer?"]
    end
```

```figure
agentic-rag-loop
```

## Build It

### Step 1: BM25 Implementation

```python
import math
from collections import Counter

class BM25:
    def __init__(self, k1=1.2, b=0.75):
        self.k1 = k1
        self.b = b
        self.docs = []
        self.doc_lengths = []
        self.avg_dl = 0
        self.doc_freqs = {}
        self.n_docs = 0

    def index(self, documents):
        self.docs = documents
        self.n_docs = len(documents)
        self.doc_lengths = []
        self.doc_freqs = {}

        for doc in documents:
            words = doc.lower().split()
            self.doc_lengths.append(len(words))
            unique_words = set(words)
            for word in unique_words:
                self.doc_freqs[word] = self.doc_freqs.get(word, 0) + 1

        self.avg_dl = sum(self.doc_lengths) / self.n_docs if self.n_docs else 1

    def score(self, query, doc_idx):
        query_words = query.lower().split()
        doc_words = self.docs[doc_idx].lower().split()
        doc_len = self.doc_lengths[doc_idx]
        word_counts = Counter(doc_words)
        score = 0.0

        for term in query_words:
            if term not in word_counts:
                continue
            tf = word_counts[term]
            df = self.doc_freqs.get(term, 0)
            idf = math.log((self.n_docs - df + 0.5) / (df + 0.5) + 1)
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avg_dl)
            score += idf * numerator / denominator

        return score

    def search(self, query, top_k=10):
        scores = [(i, self.score(query, i)) for i in range(self.n_docs)]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]
```

### Step 2: Reciprocal Rank Fusion

```python
def reciprocal_rank_fusion(ranked_lists, k=60):
    scores = {}
    for ranked_list in ranked_lists:
        for rank, (doc_id, _) in enumerate(ranked_list):
            if doc_id not in scores:
                scores[doc_id] = 0.0
            scores[doc_id] += 1.0 / (k + rank + 1)
    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return fused
```

### Step 3: Hybrid Search Pipeline

```python
def hybrid_search(query, chunks, vector_embeddings, vocab, idf, bm25_index, top_k=5, fusion_k=60):
    query_emb = tfidf_embed(query, vocab, idf)
    vector_results = search(query_emb, vector_embeddings, top_k=top_k * 3)
    bm25_results = bm25_index.search(query, top_k=top_k * 3)
    fused = reciprocal_rank_fusion([vector_results, bm25_results], k=fusion_k)
    return fused[:top_k]
```

### Step 4: Simple Reranker

In production, you would use a cross-encoder model. Here we build a reranker that scores query-document relevance using word overlap, term importance, and phrase matching.

```python
def rerank(query, candidates, chunks):
    query_words = set(query.lower().split())
    stop_words = {"the", "a", "an", "is", "are", "was", "were", "what", "how",
                  "why", "when", "where", "do", "does", "for", "of", "in", "to",
                  "and", "or", "on", "at", "by", "it", "its", "this", "that",
                  "with", "from", "be", "has", "have", "had", "not", "but"}
    query_terms = query_words - stop_words

    scored = []
    for doc_id, initial_score in candidates:
        chunk = chunks[doc_id].lower()
        chunk_words = set(chunk.split())

        term_overlap = len(query_terms & chunk_words)

        query_bigrams = set()
        q_list = [w for w in query.lower().split() if w not in stop_words]
        for i in range(len(q_list) - 1):
            query_bigrams.add(q_list[i] + " " + q_list[i + 1])
        bigram_matches = sum(1 for bg in query_bigrams if bg in chunk)

        position_boost = 0
        for term in query_terms:
            pos = chunk.find(term)
            if pos != -1 and pos < len(chunk) // 3:
                position_boost += 0.5

        rerank_score = (
            term_overlap * 1.0
            + bigram_matches * 2.0
            + position_boost
            + initial_score * 5.0
        )
        scored.append((doc_id, rerank_score))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored
```

### Step 5: HyDE (Hypothetical Document Embeddings)

```python
def hyde_generate_hypothesis(query):
    templates = {
        "what": "The answer to '{query}' is as follows: Based on our documentation, {topic} involves specific policies and procedures that define how the process works.",
        "how": "To address '{query}': The process involves several steps. First, you need to initiate the request. Then, the system processes it according to the defined rules.",
        "default": "Regarding '{query}': Our records indicate specific details and policies related to this topic that provide a comprehensive answer."
    }
    query_lower = query.lower()
    if query_lower.startswith("what"):
        template = templates["what"]
    elif query_lower.startswith("how"):
        template = templates["how"]
    else:
        template = templates["default"]

    topic_words = [w for w in query.lower().split()
                   if w not in {"what", "is", "the", "how", "do", "does", "a", "an",
                                "for", "of", "to", "in", "on", "at", "by", "and", "or"}]
    topic = " ".join(topic_words) if topic_words else "this topic"

    return template.format(query=query, topic=topic)


def hyde_search(query, chunks, vector_embeddings, vocab, idf, top_k=5):
    hypothesis = hyde_generate_hypothesis(query)
    hypothesis_emb = tfidf_embed(hypothesis, vocab, idf)
    results = search(hypothesis_emb, vector_embeddings, top_k)
    return results, hypothesis
```

### Step 6: Parent-Child Chunking

```python
def create_parent_child_chunks(text, parent_size=200, child_size=50):
    words = text.split()
    parents = []
    children = []
    child_to_parent = {}

    parent_idx = 0
    start = 0
    while start < len(words):
        parent_end = min(start + parent_size, len(words))
        parent_text = " ".join(words[start:parent_end])
        parents.append(parent_text)

        child_start = start
        while child_start < parent_end:
            child_end = min(child_start + child_size, parent_end)
            child_text = " ".join(words[child_start:child_end])
            child_idx = len(children)
            children.append(child_text)
            child_to_parent[child_idx] = parent_idx
            child_start += child_size

        parent_idx += 1
        start += parent_size

    return parents, children, child_to_parent
```

### Step 7: Faithfulness Evaluation

```python
def evaluate_faithfulness(answer, retrieved_chunks):
    answer_sentences = [s.strip() for s in answer.split(".") if len(s.strip()) > 10]
    if not answer_sentences:
        return 1.0, []

    grounded = 0
    ungrounded = []
    context = " ".join(retrieved_chunks).lower()

    for sentence in answer_sentences:
        words = set(sentence.lower().split())
        stop_words = {"the", "a", "an", "is", "are", "was", "were", "and", "or",
                      "to", "of", "in", "for", "on", "at", "by", "it", "this", "that"}
        content_words = words - stop_words
        if not content_words:
            grounded += 1
            continue

        matched = sum(1 for w in content_words if w in context)
        ratio = matched / len(content_words) if content_words else 0

        if ratio >= 0.5:
            grounded += 1
        else:
            ungrounded.append(sentence)

    score = grounded / len(answer_sentences) if answer_sentences else 1.0
    return score, ungrounded


def evaluate_retrieval_recall(queries_with_relevant, retrieval_fn, k=5):
    total_recall = 0.0
    results = []

    for query, relevant_indices in queries_with_relevant:
        retrieved = retrieval_fn(query, k)
        retrieved_indices = set(idx for idx, _ in retrieved)
        relevant_set = set(relevant_indices)
        hits = len(retrieved_indices & relevant_set)
        recall = hits / len(relevant_set) if relevant_set else 1.0
        total_recall += recall
        results.append({
            "query": query,
            "recall": recall,
            "hits": hits,
            "total_relevant": len(relevant_set)
        })

    avg_recall = total_recall / len(queries_with_relevant) if queries_with_relevant else 0
    return avg_recall, results
```

## Use It

With a real cross-encoder for reranking:

```python
from sentence_transformers import CrossEncoder

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def rerank_with_cross_encoder(query, candidates, chunks, top_k=5):
    pairs = [(query, chunks[doc_id]) for doc_id, _ in candidates]
    scores = reranker.predict(pairs)
    scored = list(zip([doc_id for doc_id, _ in candidates], scores))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]
```

With Cohere's managed reranker:

```python
import cohere

co = cohere.Client()

def rerank_with_cohere(query, candidates, chunks, top_k=5):
    docs = [chunks[doc_id] for doc_id, _ in candidates]
    response = co.rerank(
        model="rerank-english-v3.0",
        query=query,
        documents=docs,
        top_n=top_k
    )
    return [(candidates[r.index][0], r.relevance_score) for r in response.results]
```

For HyDE with a real LLM:

```python
import anthropic

client = anthropic.Anthropic()

def hyde_with_llm(query):
    response = client.messages.create(
        model="claude-sonnet-5",
        max_tokens=256,
        messages=[{
            "role": "user",
            "content": f"Write a short paragraph that would be a good answer to this question. Do not say you don't know. Just write what the answer would look like.\n\nQuestion: {query}"
        }]
    )
    return response.content[0].text
```

For production hybrid search with Weaviate:

```python
import weaviate

client = weaviate.connect_to_local()

collection = client.collections.get("Documents")
response = collection.query.hybrid(
    query="enterprise refund policy",
    alpha=0.5,
    limit=10
)
```

The alpha parameter controls the balance: 0.0 = pure keyword (BM25), 1.0 = pure vector, 0.5 = equal weight. Most production systems use alpha between 0.3 and 0.7.

## Ship It

This lesson produces:
- `outputs/prompt-advanced-rag-debugger.md` -- a prompt for diagnosing and fixing RAG quality issues
- `outputs/skill-advanced-rag.md` -- a skill for building production-grade RAG with hybrid search and reranking

## Exercises

1. Compare BM25 vs vector search vs hybrid search on the sample documents. For each of the 5 test queries, record which approach returns the most relevant chunk in position #1. Hybrid search should win on at least 3 out of 5.

2. Implement a metadata filter. Add a "category" field to each document (security, billing, api, product). Before running vector search, filter chunks to only the relevant category. Test with "What encryption is used?" and verify it only searches security-category chunks.

3. Build a full HyDE pipeline using the simple generate function from Lesson 06. Compare retrieval quality (top-3 relevance) between direct query search and HyDE search on all 5 test queries. HyDE should improve results for vague queries.

4. Implement the parent-child chunking strategy on the sample documents. Use child_size=30 and parent_size=100. Search with child chunks but return parent chunks in the prompt. Compare the generated answers to standard chunking with chunk_size=50.

5. Create an evaluation dataset: 10 questions with known answer chunks. Measure Recall@3, Recall@5, and Recall@10 for (a) vector search only, (b) BM25 only, (c) hybrid search, (d) hybrid + reranking. Plot the results and identify where reranking helps most.

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| BM25 | "Keyword search" | A probabilistic ranking algorithm that scores documents by term frequency, inverse document frequency, and document length normalization |
| Hybrid search | "Best of both worlds" | Running semantic (vector) and keyword (BM25) search in parallel, then merging results with rank fusion |
| Reciprocal Rank Fusion | "Merge ranked lists" | Combining multiple ranked lists by summing 1/(k + rank) for each document across all lists |
| Reranking | "Second pass scoring" | Using a more expensive cross-encoder model to re-score a candidate set from initial retrieval |
| Cross-encoder | "Joint query-document model" | A model that takes a query and document as a single input, producing a relevance score; more accurate than bi-encoders but too slow for full corpus search |
| Bi-encoder | "Independent embedding model" | A model that embeds queries and documents independently; fast because embeddings are precomputed, but less accurate than cross-encoders |
| HyDE | "Search with a fake answer" | Generate a hypothetical answer to the query, embed it, and search for real documents similar to it |
| Parent-child chunking | "Small search, big context" | Index small chunks for precise retrieval but return the larger parent chunk to provide sufficient context |
| Metadata filtering | "Narrow before searching" | Filtering documents by attributes (date, source, category) before running vector search to reduce the search space |
| Faithfulness | "Did it stay grounded" | Whether the generated answer is supported by the retrieved documents, as opposed to hallucinated from the model's training data |

## Further Reading

- Robertson & Zaragoza, "The Probabilistic Relevance Framework: BM25 and Beyond" (2009) -- the definitive reference for BM25, explaining the probabilistic foundations behind the formula
- Cormack et al., "Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank Learning Methods" (2009) -- the original RRF paper showing it beats more complex fusion methods
- Gao et al., "Precise Zero-Shot Dense Retrieval without Relevance Labels" (2022) -- the HyDE paper demonstrating that hypothetical document embeddings improve retrieval without any training data
- Nogueira & Cho, "Passage Re-ranking with BERT" (2019) -- showed cross-encoder reranking on top of BM25 significantly improves retrieval quality
- [Khattab et al., "DSPy: Compiling Declarative Language Model Calls into Self-Improving Pipelines" (2023)](https://arxiv.org/abs/2310.03714) -- treats prompt construction and weight selection as an optimization problem over retrieval pipelines; read this for "program LLMs" instead of "prompt LLMs."
- [Edge et al., "From Local to Global: A Graph RAG Approach to Query-Focused Summarization" (Microsoft Research 2024)](https://arxiv.org/abs/2404.16130) -- GraphRAG paper: entity-relation extraction + Leiden community detection for query-focused summarization; the global vs local retrieval distinction.
- [Asai et al., "Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection" (ICLR 2024)](https://arxiv.org/abs/2310.11511) -- self-evaluating RAG with reflection tokens; the agentic frontier past static retrieve-then-generate.
- [LangChain Query Construction blog](https://blog.langchain.dev/query-construction/) -- how to translate natural-language queries into structured database queries (Text-to-SQL, Cypher) as a pre-retrieval step.
