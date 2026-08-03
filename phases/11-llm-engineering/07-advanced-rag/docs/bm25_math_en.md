# BM25 Mathematical Details & Code Alignment (For Reference in Hand-written Implementations)

To implement the BM25 algorithm from scratch, you need to understand its mathematical representation and parameter control logic:

## 1. Mathematical Formula

For query $q$ and document $d$, the BM25 relevance score is calculated as follows:

$$BM25(q, d) = \sum_{t \in q} IDF(t) \cdot \frac{tf(t, d) \cdot (k_1 + 1)}{tf(t, d) + k_1 \cdot \left(1 - b + b \cdot \frac{|d|}{avgdl}\right)}$$

## 2. Core Formula Breakdown & Code Alignment

*   **Inverse Document Frequency $IDF(t)$**: Calculated in the code as `math.log((self.n_docs - df + 0.5) / (df + 0.5) + 1)`, which is mathematically equivalent to $\ln \left( \frac{N - n(t) + 0.5}{n(t) + 0.5} + 1 \right) = \ln \left( \frac{N + 1}{n(t) + 0.5} \right)$. Here, $N$ is the total number of documents, and $n(t)$ is the number of documents containing the term $t$. Using $0.5$ as a smoothing value avoids division by zero or negative scores for highly frequent terms.
*   **Term Frequency Saturation Parameter $k_1$ (default 1.2)**: Controls the speed at which term frequency contribution saturates. As term frequency $tf \to \infty$, the score contribution from term frequency is capped at $k_1 + 1$.
*   **Document Length Penalty Parameter $b$ (default 0.75)**: Controls the intensity of the document length penalty. When $b=1$, the length penalty is fully applied based on the ratio of document length to average document length $\frac{|d|}{avgdl}$; when $b=0$, length penalty is completely disabled.

## 3. A Simple Step-by-Step Numerical Example

To intuitively understand the inner workings of BM25, let's assume a minimal corpus of 3 documents (total documents $N = 3$) and use hyperparameters $k_1 = 1.2, b = 0.75$:

*   **Document 1 ($D_1$)**: "Artificial intelligence is the future of AI technology" (length $|D_1| = 8$)
*   **Document 2 ($D_2$)**: "AI technology shapes advanced robotics and artificial neural networks" (length $|D_2| = 9$)
*   **Document 3 ($D_3$)**: "Robotics and automation are growing rapidly" (length $|D_3| = 6$)

The average document length is $avgdl = \frac{8 + 9 + 6}{3} \approx 7.67$.

We issue the query **$q$: "AI robotics"** (containing terms "ai" and "robotics").

### Step 1: Calculate IDF for each term
*   The term **"ai"** appears in $D_1$ and $D_2$, so $n(\text{"ai"}) = 2$:
    $$IDF(\text{"ai"}) = \ln \left( \frac{3 - 2 + 0.5}{2 + 0.5} + 1 \right) = \ln(0.6 + 1) = \ln(1.6) \approx 0.470$$
*   The term **"robotics"** appears in $D_2$ and $D_3$, so $n(\text{"robotics"}) = 2$:
    $$IDF(\text{"robotics"}) = \ln(1.6) \approx 0.470$$

### Step 2: Calculate the BM25 score for each document

*   **Document $D_1$ Score**:
    - Contains "ai" (term frequency $tf = 1$), does not contain "robotics" ($tf = 0$).
    - Calculate the length normalization penalty term $K$ for $D_1$:
        $$K_{1} = 1.2 \cdot \left(1 - 0.75 + 0.75 \cdot \frac{8}{7.67}\right) \approx 1.2 \cdot (0.25 + 0.782) \approx 1.238$$
    - Contribution of "ai" to the score:
        $$Score(\text{"ai"}, D_1) = 0.470 \cdot \frac{1 \cdot (1.2 + 1)}{1 + 1.238} \approx 0.470 \cdot \frac{2.2}{2.238} \approx 0.462$$
    - **$Score(q, D_1) \approx 0.462$**.

*   **Document $D_2$ Score**:
    - Contains both "ai" ($tf = 1$) and "robotics" ($tf = 1$).
    - Calculate the length normalization penalty term $K$ for $D_2$:
        $$K_{2} = 1.2 \cdot \left(1 - 0.75 + 0.75 \cdot \frac{9}{7.67}\right) \approx 1.2 \cdot (0.25 + 0.880) \approx 1.356$$
    - Contribution of "ai" to the score:
        $$Score(\text{"ai"}, D_2) = 0.470 \cdot \frac{1 \cdot 2.2}{1 + 1.356} \approx 0.470 \cdot 0.934 \approx 0.439$$
    - Contribution of "robotics" to the score is also $0.439$.
    - **$Score(q, D_2) \approx 0.439 + 0.439 = 0.878$**.

*   **Document $D_3$ Score**:
    - Does not contain "ai" ($tf = 0$), contains "robotics" ($tf = 1$).
    - Calculate the length normalization penalty term $K$ for $D_3$:
        $$K_{3} = 1.2 \cdot \left(1 - 0.75 + 0.75 \cdot \frac{6}{7.67}\right) \approx 1.2 \cdot (0.25 + 0.587) \approx 1.004$$
    - Contribution of "robotics" to the score:
        $$Score(\text{"robotics"}, D_3) = 0.470 \cdot \frac{1 \cdot 2.2}{1 + 1.004} \approx 0.470 \cdot \frac{2.2}{2.004} \approx 0.516$$
    - **$Score(q, D_3) \approx 0.516$**.

### Ranking Results & Length Penalty Observation
The final retrieval ranking is: **$D_2$ ($0.878$) > $D_3$ ($0.516$) > $D_1$ ($0.462$)**.

> [NOTE]
> **Comparing Length Penalty Efficacy:**
> Compare $D_1$ and $D_3$. Each document contains exactly one search term ($D_1$ has "ai", $D_3$ has "robotics"), and both terms have the exact same global IDF ($0.470$).
> However, $D_3$ is shorter ($|D_3|=6 < avgdl$), whereas $D_1$ is longer ($|D_1|=8 > avgdl$). Due to length normalization, the shorter document $D_3$ receives a significantly higher score ($0.516$) than the longer document $D_1$ ($0.462$). This demonstrates how the length penalty parameter $b$ successfully boosts shorter, more concise matches.
