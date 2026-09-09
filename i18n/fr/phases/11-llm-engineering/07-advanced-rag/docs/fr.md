# RAG avancé (déchets, réaffectation, recherche hybride)

> RAG de base récupère les plus similaires de la partie supérieure. Cela fonctionne pour les questions simples. Il se décompose pour le raisonnement multi-hop, les requêtes ambiguës et les grandes corporations. RAG avancé est la différence entre une démo qui fonctionne sur 10 documents et un système qui fonctionne sur 10 millions.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 11, Lesson 06 (RAG)
**Time:** ~90 minutes
**Related:**La phase 5 · 23 (Strategie de déchiquetage pour RAG) couvre les six algorithmes de déchiquetage  récursif, sémantique, phrase, parent-document, déchiquetage tardif, récupération contextuelle  avec des repères Vectara/Anthropic. Cette leçon s'appuie sur le dessus: recherche hybride, réévaluation, transformation de requête.

## Objectifs d'apprentissage

- Mettre en œuvre des stratégies de déchiquetage avancées (sémantique, récursive, parent-enfant) qui préservent la structure et le contexte du document
- Construire un pipeline de recherche hybride combinant le partage de mots clés BM25 avec la recherche vectorielle sémantique et un réencodeur croisé
- Appliquer des techniques de transformation des requêtes (HyDE, multi requêtes, step-back) pour améliorer la récupération sur des questions ambiguës ou complexes
- Diagnostication et réparation des défaillances courantes du RAG: mauvais morceau récupéré, réponse non contextuelle, décomposition du raisonnement multi-hop

## Le problème

Vous avez construit un pipeline RAG de base dans la leçon 06. Il fonctionne pour des questions simples sur un petit corpus.

**Ambiguous query**"Quel était le chiffre d'affaires au dernier trimestre?" La recherche sémantique renvoie des morceaux sur la stratégie de revenus, les projections de revenus et les pensées du directeur financier sur la croissance des revenus. Tout cela est semanticement similaire au mot "revenus". Aucun ne contient le nombre réel.$47.2M in Q3 2025" but uses the word "earnings" instead of "revenue." The embedding model thinks "revenue strategy" is closer to the query than "Q3 earnings were $47,2 M. "

**Multi-hop question**Le rapport de satisfaction de chaque équipe exige de trouver les scores de satisfaction de chaque équipe, de les comparer et d'identifier le maximum.

**Large corpus problem**Vous avez 2 millions de blocs. La réponse correcte est dans la partie #1,847,293. Votre recherche de top-5 tire les blocs #14, #89,201, #1,200,000, #44, et #901,333. Fermé dans l'espace d'intégration, mais aucun ne contient la réponse. À cette échelle, la recherche de voisin le plus proche approximatif introduit suffisamment d'erreur pour que les résultats pertinents soient repoussés hors de la partie supérieure.

Le RAG de base échoue parce que la similitude vectorielle n'est pas la même que la pertinence. Une pièce peut être semantiquement similaire à une requête sans être utile pour la répondre. Le RAG avancé aborde ce problème avec quatre techniques: recherche hybride (ajout de correspondance de mots clés), réévaluation (téléchargement des candidats plus attentivement), transformation de requête (corrigation de la requête avant de la recherche) et meilleure fragmentation (obtention de la bonne granularité).

## Le concept

### Recherche hybride: sémantique + mot clé

La recherche sémantique (semblance vectorielle) est bonne pour comprendre la signification. " Comment annuler mon abonnement ? " correspond à " Pas pour mettre fin à votre plan " même s'ils ne partagent pas de mots. Mais il manque de correspondances exactes. " Code d'erreur E-4021 " peut ne pas correspondre à une pièce contenant " E-4021 " si le modèle d'intégration le traite comme du bruit.

La recherche de mots clés (BM25) est l'inverse. Elle excelle à des correspondances exactes. "E-4021" correspond parfaitement. Mais "annuler mon abonnement" renvoie zéro résultats si le document dit "terminer votre plan".

La recherche hybride fait les deux, puis fusionne les résultats.

#### BM25 plongée profonde

**BM25**(Best Matching 25) est l'algorithme de recherche de mots clés standard. Il est la colonne vertébrale des moteurs de recherche (comme Elasticsearch et Lucene) depuis les années 1990. Il s'agit d'une mise à niveau évolutionnaire de l'algorithme TF-IDF classique.

Si vous voulez une compréhension conceptuelle rapide du fonctionnement de BM25, vous pouvez vous concentrer sur son**three core pillars**(sans se laisser embourber par les formules mathématiques):

1. **Term Rarity (IDF: The rarer the term, the higher its weight)**- Le numéro de la liste:
   - L'idée de base est "rarité équivaut à la pertinence". Dans une requête, des termes communs comme "le", "comment", "à" apparaissent dans presque tous les documents et ne permettent pas d'identifier le document correct. D'autre part, des termes rares comme "E-4021" (un code d'erreur spécifique) ou "hallucination" (un sujet spécifique) sont très sélectifs. BM25 attribue automatiquement un poids élevé à des termes rares et des réductions ou ignore les termes communs.
2. **Term Frequency Saturation (TF Saturation: Repeated hits have diminishing returns, with a ceiling)**- Le numéro de la liste:
   - L'idée principale est d'éviter le " remplissage de mots clés ".Mentionner un terme 1 fois est utile, 2 fois est légèrement plus utile, mais le mentionner 100 fois ne rend pas un document 100 fois plus pertinent. BM25 fixe un plafond sur la contribution de fréquence des termes (contrôlé par paramètre $k_1$Comme un terme apparaît plus de fois dans un document, sa contribution progressive à la note saturera rapidement et se réduira à des plateaux.
3. **Document Length Normalization (L-Norm: Matching in short documents is more significant)**- Le numéro de la liste:
   - L'idée principale est de pénaliser les documents gonflés et "d'eau vers le bas". Si un titre court de 10 mots correspond à un mot clé et un livre blanc complet de 10 000 mots correspond à ce même mot clé une fois, le titre court est très probablement entièrement dédié à ce mot clé. BM25 pénalise les documents en fonction de leur longueur relative (contrôlé par paramètre $b$), en augmentant les documents plus courts et en abaissant les scores pour les documents extrêmement longs.

> Je suis là.
> Pour des formules mathématiques détaillées, des dérivées de l'armée israélienne de l'air, des contrôles de paramètres et un exemple numérique calculé à la main étape par étape, consultez la page dédiée: [BM25 Mathematical Details & Code Alignment](bm25_math_en.md)- Je suis désolé .


### Fusion de rang réciproque (RRF)

Dans les architectures RAG ou Hybrid Search à grande échelle, **RRF (Reciprocal Rank Fusion)**est l'algorithme le plus largement adopté et élégant pour combiner les listes classées.

En termes simples, la logique de base de RRF est: **disregard raw scores, focus only on ranks**- Je suis désolé .

#### 1. Pourquoi avons-nous besoin de RRF ?
Lors de la construction d'un système de récupération hybride, nous utilisons généralement:
- **Semantic Retrieval (Vector Search)**: Les scores sont généralement de similitude cosine ou produit de point, généralement allant de $[-1, 1]$ou $[0, 1]$- Je suis désolé .
- **Keyword Retrieval (BM25)**: Les scores sont des nombres positifs flotifs sans limite (selon la fréquence des termes et la longueur du document).

La somme directe de ces deux scores est mathématiquement incorrecte car leurs échelles, leurs distributions et leurs significations physiques sont complètement différentes. RRF contourne ce problème d'alignement en se concentrant uniquement sur la "position relative d'un document dans chaque liste", en normalisant les sorties de différents systèmes de récupération dans une seule dimension.

#### 2. Une plongée dans la formule

Pour un document de candidature $d$, son score RRF est calculé comme suit:

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
