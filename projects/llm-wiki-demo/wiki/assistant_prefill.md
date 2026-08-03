---
title: Assistant Prefill
type: Technology
last_updated: 2026-07-04
sources:
  - "prompt_engineering_cn.md"
---
# Assistant Prefill

### Core Mechanism
Assistant Prefill is an API-level technique that allows developers to inject a partial response into the assistant's message role. The LLM is forced to complete the generation starting from the provided string, effectively bypassing conversational filler (e.g., "Sure, here is the JSON you requested:") and guaranteeing structural compliance.

### Examples/Templates
```python
import anthropic

client = anthropic.Anthropic()

response = client.messages.create(
    model="claude-opus-4-7",
    max_tokens=1024,
    temperature=0.0,
    system="You are a data extraction engine. Output valid JSON only.",
    messages=[
        {
            "role": "user",
            "content": "Extract: John Smith, age 34, works at Google as a senior engineer since 2019.",
        },
        {
            "role": "assistant",
            "content": "{",
        },
    ],
)

result = "{" + response.content[0].text
print(result)
```

### Parameters/Caveats
* **Provider Support**: This is a native feature of Anthropic's API. OpenAI's API does not natively support assistant prefill; developers must use OpenAI's Structured Outputs or strict system instructions instead.

## Outgoing Links
- [[prompt_anatomy|Prompt Anatomy]]
- [[prompt_patterns|Prompt Patterns]]

## References
- [[index|Wiki Index]]
