---
title: Self-Consistency (自洽性)
type: Concept
last_updated: 2026-07-04
sources:
  - "few_shot_cot_cn.md"
---
# Self-Consistency (自洽性)

### Core Mechanism
Introduced by Wang et al. (2023), Self-Consistency samples multiple ($N$) independent reasoning paths from the model at a temperature $> 0$ and performs a majority vote on the final answer. This mitigates the risk of a single reasoning path containing a random calculation error.

### Implementation Example
```python
def self_consistency_solve(question, examples, client, model, n_samples=5):
    system, user = build_cot_prompt(question, examples)
    answers = []
    reasonings = []
    for _ in range(n_samples):
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user}
            ],
            temperature=0.7 # Crucial for diverse reasoning paths
        )
        text = response.choices[0].message.content
        reasonings.append(text)
        answer = extract_answer(text)
        if answer is not None:
            answers.append(answer)
    vote_counts = Counter(answers)
    best_answer = vote_counts.most_common(1)[0][0] if vote_counts else None
    confidence = vote_counts[best_answer] / len(answers) if best_answer else 0
    return best_answer, confidence, reasonings, vote_counts
```

### Parameters & Caveats
* **Temperature**: Must be set to $> 0$ (typically $0.7$) to ensure path diversity. At $0.0$, all paths will be identical.
* **Sample Size ($N$)**: $N=5$ captures most benefits. $N=3$ is the minimum for a meaningful vote. $N > 10$ yields diminishing returns.
* **Cost**: $N$ times the API cost and latency of a single run.

## Outgoing Links
- [[chain_of_thought_cot_思维链|Chain of Thought (CoT) (思维链)]]

## References
- [[index|Wiki Index]]
