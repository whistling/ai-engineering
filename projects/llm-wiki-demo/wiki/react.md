---
title: ReAct
type: Concept
last_updated: 2026-07-04
sources:
  - "few_shot_cot_cn.md"
---
# ReAct

### Core Mechanism
Introduced by Yao et al. (2022), ReAct (Reasoning and Acting) combines reasoning traces with action execution. The model alternates between generating reasoning thoughts and executing actions (such as calling APIs, searching databases, or running calculations), then observing the results to update its plan.

### Example Flow
1. **Thought**: I need to find the country where the Eiffel Tower is located.
2. **Action**: Search["Eiffel Tower location"]
3. **Observation**: Paris, France
4. **Thought**: Now I need the population of France.
5. **Action**: Search["France population 2024"]
6. **Observation**: 68.4 million
7. **Thought**: I have the final answer.
8. **Answer**: 68.4 million

### Parameters & Caveats
* **Performance**: Achieved 35.1% exact match on HotpotQA (multi-hop QA) compared to 29.4% for standard CoT.
* **Error Correction**: Highly robust because the model can correct reasoning errors dynamically based on real-world observations.

## Outgoing Links
- [[chain_of_thought_cot_思维链|Chain of Thought (CoT) (思维链)]]

## References
- [[index|Wiki Index]]
