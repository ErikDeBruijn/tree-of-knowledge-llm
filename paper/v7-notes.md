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
- Ruby: 20% → 70% correct with adapter (the model learns to WRITE Ruby, not just recognize it)

## Grove vs RAG: different systems for different purposes

RAG adds facts to the context — the model reads more but doesn't change.
Grove adds capabilities to the model — it can DO more (write Ruby, reason medically).

This is not a competition but a complement:
- **Grove**: capability injection (how to write Ruby, how to reason about cardiology)
- **RAG**: fact injection (latest drug interactions, current guidelines)

A pytorch specialist doesn't need to memorize the API docs (RAG can provide those).
It needs to understand tensor operations, autograd patterns, module composition.
That's a capability, not a fact.

## Hierarchical capability trees

The grove naturally forms a tree:
- Code expert → Ruby specialist → Rails specialist
- Code expert → Python specialist → pandas specialist, pytorch specialist
- Medical expert → Radiology specialist

Each level learns only the DELTA from its parent. A pytorch specialist
doesn't learn "what is a function" (code expert knows) or "what is Python"
(Python specialist knows). It learns tensor patterns, autograd idioms.

This means: adding a new language (Rust, Go) requires minimal data,
because the shared programming foundation is already in the code expert.
