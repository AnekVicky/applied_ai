🚨 RAG doesn’t always need retrieval.

While building an enterprise RAG pipeline, I ran into an interesting problem:

The user asks a question →
🔍 Vector search runs →
📄 Top-K documents are retrieved →
🤖 LLM finally says: “I don’t know.”

The problem?

We already spent retrieval cost before knowing whether retrieval was necessary.

For example:

“Any issues with my login?”

If the system cannot confidently determine the relevant intent/category, blindly querying the vector store can introduce:

❌ Unnecessary vector search
❌ Irrelevant documents
❌ Extra latency
❌ More tokens/context
❌ Potentially noisy LLM responses

This led me to explore Intent-Based Routing:

User Query
↓
🎯 Intent / Category Detection
↓
Is retrieval required?
↙️ ↘️
No Yes
↓ ↓
Direct response Targeted retrieval
↓
RAG → LLM

The key architectural shift is:

Don’t retrieve first and decide later. Decide the route first.

I'm currently experimenting with different routing strategies, including keyword-based detection, metadata filtering, and semantic/LLM-based intent routing, and comparing their trade-offs.

This is where RAG becomes more than just:

Embedding → Vector DB → LLM

It becomes a retrieval architecture problem.

