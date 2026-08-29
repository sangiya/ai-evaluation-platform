from __future__ import annotations

from dataclasses import dataclass
from typing import List, Set

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sk_cosine


@dataclass
class RagSample:
    query: str
    retrieved_ids: List[str]
    relevant_ids: List[str]
    answer: str
    context: str


@dataclass
class RagMetrics:
    context_precision: float
    context_recall: float
    answer_faithfulness: float
    answer_relevance: float

    def mean_score(self) -> float:
        return (
            self.context_precision
            + self.context_recall
            + self.answer_faithfulness
            + self.answer_relevance
        ) / 4


def _context_precision(retrieved: List[str], relevant: Set[str]) -> float:
    if not retrieved:
        return 0.0
    hits = sum(1 for d in retrieved if d in relevant)
    return hits / len(retrieved)


def _context_recall(retrieved: List[str], relevant: Set[str]) -> float:
    if not relevant:
        return 1.0
    hits = sum(1 for d in retrieved if d in relevant)
    return hits / len(relevant)


def _answer_faithfulness(answer: str, context: str) -> float:
    answer_tokens = set(answer.lower().split())
    context_tokens = set(context.lower().split())
    if not answer_tokens:
        return 0.0
    return len(answer_tokens & context_tokens) / len(answer_tokens)


def _answer_relevance(query: str, answer: str) -> float:
    try:
        tfidf = TfidfVectorizer().fit_transform([query, answer])
        return float(sk_cosine(tfidf[0:1], tfidf[1:2])[0][0])
    except ValueError:
        return 0.0


def rag_evaluate(samples: List[RagSample]) -> RagMetrics:
    """Compute mean RAG metrics across all samples."""
    if not samples:
        return RagMetrics(0.0, 0.0, 0.0, 0.0)

    precisions, recalls, faithfulness, relevances = [], [], [], []

    for s in samples:
        rel_set = set(s.relevant_ids)
        precisions.append(_context_precision(s.retrieved_ids, rel_set))
        recalls.append(_context_recall(s.retrieved_ids, rel_set))
        faithfulness.append(_answer_faithfulness(s.answer, s.context))
        relevances.append(_answer_relevance(s.query, s.answer))

    n = len(samples)
    return RagMetrics(
        context_precision=sum(precisions) / n,
        context_recall=sum(recalls) / n,
        answer_faithfulness=sum(faithfulness) / n,
        answer_relevance=sum(relevances) / n,
    )
