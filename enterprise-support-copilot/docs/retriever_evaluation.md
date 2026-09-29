                 RAG RETRIEVER EVALUATION
                 ────────────────────────

                     Test Dataset
                         │
                         ▼
                ┌─────────────────┐
                │   User Question  │
                │                 │
                │ "What happens   │
                │ if API limit    │
                │ is exceeded?"   │
                └────────┬────────┘
                         │
                         ▼
                ┌─────────────────┐
                │    Retriever    │
                │   Top-K = 3     │
                └────────┬────────┘
                         │
                         ▼
              ┌──────────────────────┐
              │ Retrieved Documents  │
              │                      │
              │ TCK-003              │
              │ TCK-028              │
              │ TCK-014              │
              └──────────┬───────────┘
                         │
                         ▼
              ┌──────────────────────┐
              │ Compare with         │
              │ Expected Sources     │
              │                      │
              │ Expected:            │
              │ TCK-003, TCK-028     │
              └──────────┬───────────┘
                         │
                         ▼
              ┌──────────────────────┐
              │  Retrieval Metrics   │
              │                      │
              │ Precision@K          │
              │ Recall@K             │
              │ F1 Score             │
              └──────────┬───────────┘
                         │
                         ▼
                ┌─────────────────┐
                │ Retrieval       │
                │ Quality Report  │
                └─────────────────┘

