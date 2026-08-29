import pytest
from ai_eval import RagSample, rag_evaluate


def _sample(retrieved=None, relevant=None, answer="answer", context="answer context", query="q"):
    return RagSample(
        query=query,
        retrieved_ids=retrieved or ["d1"],
        relevant_ids=relevant or ["d1"],
        answer=answer,
        context=context,
    )


class TestRagEvaluate:

    def test_perfect_sample_all_metrics_near_one(self):
        samples = [_sample(
            retrieved=["d1", "d2"],
            relevant=["d1", "d2"],
            answer="Paris is the capital of France",
            context="Paris is the capital of France and a great city",
        )]
        m = rag_evaluate(samples)
        assert m.context_precision == pytest.approx(1.0)
        assert m.context_recall == pytest.approx(1.0)
        assert m.answer_faithfulness == pytest.approx(1.0)
        assert m.answer_relevance > 0.0

    def test_empty_samples_returns_zero_metrics(self):
        m = rag_evaluate([])
        assert m.context_precision == 0.0
        assert m.context_recall == 0.0
        assert m.answer_faithfulness == 0.0
        assert m.answer_relevance == 0.0

    def test_mean_score_is_average_of_four(self):
        samples = [_sample(retrieved=["d1"], relevant=["d1"])]
        m = rag_evaluate(samples)
        expected = (
            m.context_precision + m.context_recall + m.answer_faithfulness + m.answer_relevance
        ) / 4
        assert m.mean_score() == pytest.approx(expected)

    def test_partial_retrieval_lowers_precision(self):
        samples = [_sample(retrieved=["d1", "noise1"], relevant=["d1"])]
        m = rag_evaluate(samples)
        assert m.context_precision == pytest.approx(0.5)

    def test_missing_relevant_lowers_recall(self):
        samples = [_sample(retrieved=["d1"], relevant=["d1", "d2"])]
        m = rag_evaluate(samples)
        assert m.context_recall == pytest.approx(0.5)

    def test_hallucinated_answer_lowers_faithfulness(self):
        samples = [_sample(
            answer="Quantum entanglement is a physics phenomenon",
            context="France is a country in Europe",
        )]
        m = rag_evaluate(samples)
        assert m.answer_faithfulness < 0.2
