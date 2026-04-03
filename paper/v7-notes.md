# Paper v7 Notes

## Framing shift: "knowledge" → "capabilities"

The paper currently frames everything as "domain knowledge injection."
But code adapters demonstrate something broader: they add *capabilities*
(recursion, patterns, scope understanding), not just *knowledge* (API names, syntax rules).

This distinction matters:
- Knowledge = facts you can look up (metformin treats diabetes)
- Capabilities = skills you can apply (how to implement binary search in Ruby)

The grove doesn't just store knowledge — it stores *ways of doing things*.
A Ruby adapter doesn't just know Ruby syntax; it can *write* Ruby.

### What to change in v7:
- Title: consider "Long Tail of Language Model **Capabilities**" or keep Knowledge but address the broader scope in the abstract
- Section 1: introduce the knowledge/capability distinction
- Section 2: learntropy measures compression progress on capabilities, not just facts
- Section 3: adapters add capabilities (FFN = knowledge storage, attention = relational patterns, early layers = syntactic capabilities)
- E8: code generation explicitly demonstrates capability injection beyond knowledge

### Key evidence for this framing:
- PPL improvement (knowledge) does NOT predict generation quality (capability)
- Layer 1 start matters for code (syntactic capability lives in early layers)
- Attention gates are function-specific (relational capability)
- Code generation improves with adapter (the model can DO something new)
