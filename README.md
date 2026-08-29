# AI Evaluation Platform

Unified evaluation harness for AI systems — RAG quality metrics, embedding retrieval scoring, and LLM cost estimation — composable via a single pipeline interface.

## Components

| Module | Evaluates | Metrics |
|--------|-----------|---------|
| `rag` | Retrieval-Augmented Generation | Context Precision, Context Recall, Answer Faithfulness, Answer Relevance |
| `embedding` | Dense embedding retrieval | Recall@1/5/10, MRR, MAP |
| `cost` | LLM API cost | Token counts (tiktoken for GPT, approx for Claude/Gemini), USD cost |

## Usage

### Full pipeline in one call

```python
import numpy as np
from ai_eval import EvalPipeline, RagSample

pipeline = EvalPipeline()

report = pipeline.run_all(
    # RAG quality
    rag_samples=[
        RagSample(
            query="What is RAG?",
            retrieved_ids=["chunk_1", "chunk_2"],
            relevant_ids=["chunk_1"],
            answer="RAG is retrieval-augmented generation.",
            context="RAG stands for retrieval augmented generation combining retrieval and generation.",
        ),
    ],
    # Embedding retrieval
    query_embeddings=query_emb_matrix,    # np.ndarray (n_queries, dim)
    corpus_embeddings=corpus_emb_matrix,  # np.ndarray (n_corpus, dim)
    relevant_indices=[{0}, {1}, {2}],     # ground truth

    # Cost estimate
    cost_model="gpt-4o",
    cost_input=system_prompt + user_message,
    cost_output_tokens=400,
)

print(report.summary())
```

Output:
```
=======================================================
AI Evaluation Platform — Report
=======================================================

RAG Metrics
  context_precision  : 0.5000
  context_recall     : 1.0000
  answer_faithfulness: 0.8571
  answer_relevance   : 0.7234
  mean_score         : 0.7701

Embedding Retrieval
  recall@1 : 0.8000
  recall@5 : 1.0000
  recall@10: 1.0000
  mrr      : 0.8667
  map      : 0.9000

Cost Estimate
  model           : gpt-4o
  input_tokens    : 387
  output_tokens   : 400
  total_cost_usd  : $0.007935
```

### Individual modules

```python
from ai_eval import rag_evaluate, embedding_evaluate, cost_estimate

# RAG
metrics = rag_evaluate(rag_samples)
print(f"Mean RAG score: {metrics.mean_score():.4f}")

# Embedding
metrics = embedding_evaluate(query_embeddings, corpus_embeddings, relevant_indices)
print(f"Recall@1: {metrics.recall_at_1:.4f}")
print(f"MRR:      {metrics.mrr:.4f}")

# Cost
estimate = cost_estimate("gpt-4o-mini", prompt_text, output_tokens=200)
print(f"Cost: ${estimate.total_cost_usd:.6f}")
```

## Running Tests

```bash
pip install -e . -r requirements.txt
pytest --tb=short -v
```

## Tech Stack

- Python 3.11+
- NumPy 1.26 — vectorized embedding operations
- scikit-learn 1.5 — TF-IDF for answer relevance
- tiktoken 0.7.0 — exact token counts for GPT models
- pytest — test runner
