---
title: Tree of Thoughts (ToT) (思维树)
type: Concept
last_updated: 2026-07-04
sources:
  - "few_shot_cot_cn.md"
---
# Tree of Thoughts (ToT) (思维树)

### Core Mechanism
Introduced by Yao et al. (2023), Tree of Thoughts (ToT) generalizes Chain of Thought by allowing the exploration of multiple reasoning branches. It evaluates the progress of each branch and uses search algorithms (like BFS or DFS) to decide which branches to expand or prune.

### Key Components
1. **Thought Generation**: Proposing multiple candidate next steps.
2. **State Evaluation**: Scoring each candidate (often using the LLM itself as an evaluator).
3. **Search Algorithm**: Navigating the tree and pruning low-scoring branches.

### Implementation Example
```python
def tree_of_thought_solve(question, client, model, breadth=3, depth=3):
    thoughts = generate_initial_thoughts(question, client, model, breadth)
    scored = [(t, evaluate_thought(t, question, client, model)) for t in thoughts]
    scored.sort(key=lambda x: x[1], reverse=True)
    
    for current_depth in range(1, depth):
        next_thoughts = []
        for thought, score in scored[:2]: # Expand top 2 branches
            extensions = extend_thought(thought, question, client, model, breadth)
            for ext in extensions:
                ext_score = evaluate_thought(ext, question, client, model)
                next_thoughts.append((ext, ext_score))
        scored = sorted(next_thoughts, key=lambda x: x[1], reverse=True)
        
    best_thought = scored[0][0] if scored else ""
    return extract_answer(best_thought), best_thought
```

### Parameters & Caveats
* **Cost**: Extremely expensive. A tree with branch factor 3 and depth 3 can require up to 39 LLM calls.
* **Best For**: Complex planning, puzzles (e.g., Game of 24, where ToT solved 74% vs CoT's 4%), and highly constrained creative problem-solving.

## Outgoing Links
- [[chain_of_thought_cot_思维链|Chain of Thought (CoT) (思维链)]]

## References
- [[index|Wiki Index]]
