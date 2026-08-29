from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Set

import numpy as np

from .cost import CostEstimate, cost_estimate
from .embedding import EmbeddingMetrics, embedding_evaluate
from .rag import RagMetrics, RagSample, rag_evaluate


@dataclass
class EvalReport:
    rag: Optional[RagMetrics] = None
    embedding: Optional[EmbeddingMetrics] = None
    cost: Optional[CostEstimate] = None

    def summary(self) -> str:
        lines = ["=" * 55, "AI Evaluation Platform — Report", "=" * 55]

        if self.rag:
            lines += [
                "",
                "RAG Metrics",
                f"  context_precision  : {self.rag.context_precision:.4f}",
                f"  context_recall     : {self.rag.context_recall:.4f}",
                f"  answer_faithfulness: {self.rag.answer_faithfulness:.4f}",
                f"  answer_relevance   : {self.rag.answer_relevance:.4f}",
                f"  mean_score         : {self.rag.mean_score():.4f}",
            ]

        if self.embedding:
            lines += [
                "",
                "Embedding Retrieval",
                f"  recall@1 : {self.embedding.recall_at_1:.4f}",
                f"  recall@5 : {self.embedding.recall_at_5:.4f}",
                f"  recall@10: {self.embedding.recall_at_10:.4f}",
                f"  mrr      : {self.embedding.mrr:.4f}",
                f"  map      : {self.embedding.map:.4f}",
            ]

        if self.cost:
            lines += [
                "",
                "Cost Estimate",
                f"  model           : {self.cost.model_id}",
                f"  input_tokens    : {self.cost.input_tokens:,}",
                f"  output_tokens   : {self.cost.output_tokens:,}",
                f"  total_cost_usd  : ${self.cost.total_cost_usd:.6f}",
            ]

        return "\n".join(lines)


class EvalPipeline:
    """
    Unified evaluation pipeline.

    Run any combination of RAG quality evaluation, embedding retrieval
    scoring, and LLM cost estimation in a single pass.
    """

    def run_rag(self, samples: List[RagSample]) -> RagMetrics:
        return rag_evaluate(samples)

    def run_embedding(
        self,
        query_embeddings: np.ndarray,
        corpus_embeddings: np.ndarray,
        relevant_indices: List[Set[int]],
    ) -> EmbeddingMetrics:
        return embedding_evaluate(query_embeddings, corpus_embeddings, relevant_indices)

    def run_cost(
        self,
        model_id: str,
        input_text: str,
        output_text: str = "",
        *,
        output_tokens: Optional[int] = None,
    ) -> CostEstimate:
        return cost_estimate(model_id, input_text, output_text, output_tokens=output_tokens)

    def run_all(
        self,
        *,
        rag_samples: Optional[List[RagSample]] = None,
        query_embeddings: Optional[np.ndarray] = None,
        corpus_embeddings: Optional[np.ndarray] = None,
        relevant_indices: Optional[List[Set[int]]] = None,
        cost_model: Optional[str] = None,
        cost_input: str = "",
        cost_output_tokens: int = 0,
    ) -> EvalReport:
        report = EvalReport()

        if rag_samples is not None:
            report.rag = self.run_rag(rag_samples)

        if query_embeddings is not None and corpus_embeddings is not None and relevant_indices is not None:
            report.embedding = self.run_embedding(query_embeddings, corpus_embeddings, relevant_indices)

        if cost_model is not None:
            report.cost = self.run_cost(cost_model, cost_input, output_tokens=cost_output_tokens)

        return report
