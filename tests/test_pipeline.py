import numpy as np
import pytest
from ai_eval import EvalPipeline, RagSample


def _make_perfect_embeddings(n=3, dim=8):
    corpus = np.eye(n + 2, dim)
    queries = corpus[:n]
    relevant = [{i} for i in range(n)]
    return queries, corpus, relevant


class TestEvalPipeline:

    def setup_method(self):
        self.pipeline = EvalPipeline()

    def test_run_rag_returns_rag_metrics(self):
        samples = [
            RagSample("q", ["d1"], ["d1"], "answer text", "answer text context")
        ]
        metrics = self.pipeline.run_rag(samples)
        assert 0.0 <= metrics.context_precision <= 1.0
        assert 0.0 <= metrics.context_recall <= 1.0

    def test_run_embedding_returns_embedding_metrics(self):
        q, c, rel = _make_perfect_embeddings()
        metrics = self.pipeline.run_embedding(q, c, rel)
        assert metrics.recall_at_1 == pytest.approx(1.0)
        assert metrics.mrr == pytest.approx(1.0)

    def test_run_cost_returns_positive_cost(self):
        estimate = self.pipeline.run_cost("gpt-4o", "Hello world", output_tokens=100)
        assert estimate.total_cost_usd > 0.0
        assert estimate.model_id == "gpt-4o"

    def test_run_all_with_rag_only_populates_rag(self):
        samples = [RagSample("q", ["d1"], ["d1"], "a", "a ctx")]
        report = self.pipeline.run_all(rag_samples=samples)
        assert report.rag is not None
        assert report.embedding is None
        assert report.cost is None

    def test_run_all_with_all_components(self):
        q, c, rel = _make_perfect_embeddings()
        samples = [RagSample("q", ["d1"], ["d1"], "answer", "answer ctx")]
        report = self.pipeline.run_all(
            rag_samples=samples,
            query_embeddings=q,
            corpus_embeddings=c,
            relevant_indices=rel,
            cost_model="gpt-4o-mini",
            cost_input="Hello world",
            cost_output_tokens=50,
        )
        assert report.rag is not None
        assert report.embedding is not None
        assert report.cost is not None

    def test_summary_contains_all_sections(self):
        q, c, rel = _make_perfect_embeddings()
        samples = [RagSample("q", ["d1"], ["d1"], "answer", "answer ctx")]
        report = self.pipeline.run_all(
            rag_samples=samples,
            query_embeddings=q,
            corpus_embeddings=c,
            relevant_indices=rel,
            cost_model="gpt-4o-mini",
            cost_input="prompt text",
            cost_output_tokens=100,
        )
        summary = report.summary()
        assert "RAG Metrics" in summary
        assert "Embedding Retrieval" in summary
        assert "Cost Estimate" in summary
        assert "gpt-4o-mini" in summary

    def test_summary_empty_report_has_no_sections(self):
        report = self.pipeline.run_all()
        summary = report.summary()
        assert "RAG Metrics" not in summary
        assert "Embedding Retrieval" not in summary
