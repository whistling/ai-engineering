---
title: Prompt Patterns
type: Concept
last_updated: 2026-07-04
sources:
  - "prompt_engineering_cn.md"
---
# Prompt Patterns

### Core Mechanism
Prompt patterns are reusable structural configurations designed to solve specific types of LLM interaction challenges. Rather than simple copy-paste text templates, they act as architectural blueprints that adapt to different domains and models.

### Examples/Templates
The following Python dictionary defines 10 core prompt patterns with their templates, variables, recommended temperatures, and descriptions:

```python
PROMPT_PATTERNS = {
    "persona": {
        "name": "角色模式",
        "template": (
            "You are {role} with {experience}.\n"
            "Your communication style is {style}.\n"
            "You prioritize {priority}.\n\n"
            "{task}"
        ),
        "variables": ["role", "experience", "style", "priority", "task"],
        "temperature": 0.7,
        "description": "Activates a specific expert distribution in the model's training data",
    },
    "few_shot": {
        "name": "少样本模式",
        "template": (
            "Here are examples of the expected input/output format:\n\n"
            "{examples}\n\n"
            "Now process this input:\n{input}"
        ),
        "variables": ["examples", "input"],
        "temperature": 0.0,
        "description": "Provides concrete examples to anchor the output format and style",
    },
    "chain_of_thought": {
        "name": "思维链模式",
        "template": (
            "Think through this step by step.\n\n"
            "Problem: {problem}\n\n"
            "Steps:\n"
            "1. Identify the key components\n"
            "2. Analyze each component\n"
            "3. Synthesize your findings\n"
            "4. State your conclusion\n\n"
            "Show your reasoning before giving the final answer."
        ),
        "variables": ["problem"],
        "temperature": 0.3,
        "description": "Forces explicit reasoning steps before the final answer",
    },
    "template_fill": {
        "name": "模板填充模式",
        "template": (
            "Extract information from the following text and fill in the template.\n\n"
            "Text: {text}\n\n"
            "Template:\n{template_structure}\n\n"
            "Fill in every field. If information is not available, write 'N/A'."
        ),
        "variables": ["text", "template_structure"],
        "temperature": 0.0,
        "description": "Constrains output to a specific structure with named fields",
    },
    "critique": {
        "name": "批判模式",
        "template": (
            "Task: {task}\n\n"
            "Step 1: Generate an initial response.\n"
            "Step 2: Critique your response for accuracy, completeness, and clarity.\n"
            "Step 3: Produce an improved final version.\n\n"
            "Label each step clearly."
        ),
        "variables": ["task"],
        "temperature": 0.5,
        "description": "Self-refinement through explicit critique before final output",
    },
    "guardrail": {
        "name": "护栏模式",
        "template": (
            "You are a {role}.\n\n"
            "Rules:\n"
            "- ONLY answer questions about {domain}\n"
            "- If the question is outside {domain}, say: 'This is outside my scope.'\n"
            "- NEVER make up information. If unsure, say 'I don't know.'\n"
            "- {additional_rules}\n\n"
            "User question: {question}"
        ),
        "variables": ["role", "domain", "additional_rules", "question"],
        "temperature": 0.3,
        "description": "Constrains the model to a specific domain with explicit boundaries",
    },
    "meta_prompt": {
        "name": "元提示词模式",
        "template": (
            "Write a prompt for an LLM that will {objective}.\n\n"
            "The prompt should include:\n"
            "- A specific role/persona\n"
            "- Clear constraints and output format\n"
            "- 2-3 few-shot examples\n"
            "- Edge case handling\n\n"
            "Optimize the prompt for {metric}.\n"
            "Target model: {model}."
        ),
        "variables": ["objective", "metric", "model"],
        "temperature": 0.7,
        "description": "Uses the LLM to generate optimized prompts for other tasks",
    },
    "decomposition": {
        "name": "分解模式",
        "template": (
            "Problem: {problem}\n\n"
            "Break this into sub-problems:\n"
            "1. List each sub-problem\n"
            "2. Solve each independently\n"
            "3. Combine sub-solutions into a final answer\n"
            "4. Verify the final answer against the original problem"
        ),
        "variables": ["problem"],
        "temperature": 0.3,
        "description": "Breaks complex problems into manageable pieces",
    },
    "audience_adapt": {
        "name": "受众适应模式",
        "template": (
            "Explain {concept} for the following audience: {audience}.\n\n"
            "Constraints:\n"
            "- Use vocabulary appropriate for {audience}\n"
            "- Length: {length}\n"
            "- Include {include}\n"
            "- Exclude {exclude}"
        ),
        "variables": ["concept", "audience", "length", "include", "exclude"],
        "temperature": 0.5,
        "description": "Adapts explanation complexity to the target audience",
    },
    "boundary": {
        "name": "边界模式",
        "template": (
            "You are an assistant that ONLY handles {scope}.\n\n"
            "If the user's request is within scope, help them fully.\n"
            "If the user's request is outside scope, respond exactly with:\n"
            "'{refusal_message}'\n\n"
            "Do not attempt to answer out-of-scope questions.\n\n"
            "User: {user_input}"
        ),
        "variables": ["scope", "refusal_message", "user_input"],
        "temperature": 0.0,
        "description": "Hard boundary on what the model will and will not respond to",
    }
}
```

### Parameters/Caveats
* **Over-constraining**: Providing too many rules (e.g., system prompts exceeding 2,000 words) consumes the model's cognitive capacity, leaving less room for actual task execution. Keep system prompts under 500 tokens for most tasks.
* **Contradictory Instructions**: Avoid conflicting rules like "Be concise" and "Be thorough and cover all edge cases". When instructions conflict, the model will choose arbitrarily.

## Outgoing Links
- [[prompt_anatomy|Prompt Anatomy]]
- [[temperature_and_sampling|Temperature and Sampling]]
- [[prompt_testing_and_scoring_framework|Prompt Testing and Scoring Framework]]

## References
- [[index|Wiki Index]]
