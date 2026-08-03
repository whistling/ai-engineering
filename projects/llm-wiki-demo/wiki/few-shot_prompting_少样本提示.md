---
title: Few-Shot Prompting (少样本提示)
type: Concept
last_updated: 2026-07-04
sources:
  - "few_shot_cot_cn.md"
---
# Few-Shot Prompting (少样本提示)

### Core Mechanism
Few-Shot Prompting provides the model with concrete input-output examples before presenting the actual task. This acts as compressed instructions, allowing the model to match patterns and output formats more reliably than abstract instructions. For complex tasks like multi-step arithmetic and symbolic reasoning, few-shot prompting can improve accuracy by 10-25% compared to zero-shot prompting.

### Selection Principles
1. **Semantic Similarity**: Select examples closest to the input in the embedding space.
2. **Label Diversity**: Cover all output categories in the examples.
3. **Difficulty Matching**: Match the complexity of the target problem.

### Parameters & Caveats
* **Optimal Count**: 3-5 examples. Fewer than 3 provides insufficient signal; more than 5 yields diminishing returns and wastes context window tokens.
* **Best For**: Format-sensitive tasks, classification, structured extraction, domain-specific terminology.
* **Avoid For**: Simple factual questions, creative tasks where examples limit creativity.

## Outgoing Links
- [[chain_of_thought_cot_思维链|Chain of Thought (CoT) (思维链)]]
- [[langchain|LangChain]]
- [[dspy|DSPy]]

## References
- [[index|Wiki Index]]
