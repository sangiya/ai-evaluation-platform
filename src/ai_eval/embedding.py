from __future__ import annotations

from dataclasses import dataclass
from typing import List, Set

import numpy as np


@dataclass
class EmbeddingMetrics:
    recall_at_1: float
    recall_at_5: float
    recall_at_10: float
    mrr: float
    map: float


def _pairwise_cosine(queries: np.ndarray, corpus: np.ndarray) -> np.ndarray:
    q_norms = np.linalg.norm(queries, axis=1, keepdims=True)
    c_norms = np.linalg.norm(corpus, axis=1, keepdims=True)
    q_norms = np.where(q_norms == 0, 1.0, q_norms)
    c_norms = np.where(c_norms == 0, 1.0, c_norms)
    return np.clip((queries / q_norms) @ (corpus / c_norms).T, -1.0, 1.0)


def _recall_at_k(sim: np.ndarray, relevant: List[Set[int]], k: int) -> float:
    if not relevant:
        return 0.0
    scores = []
    for i, rel in enumerate(relevant):
        if not rel:
            scores.append(1.0)
            continue
        top_k = set(np.argsort(sim[i])[::-1][:k].tolist())
        scores.append(len(top_k & rel) / len(rel))
    return float(np.mean(scores))


def _mrr(sim: np.ndarray, relevant: List[Set[int]]) -> float:
    if not relevant:
        return 0.0
    rrs = []
    for i, rel in enumerate(relevant):
        if not rel:
            rrs.append(1.0)
            continue
        ranked = np.argsort(sim[i])[::-1].tolist()
        rr = next((1.0 / (rank + 1) for rank, d in enumerate(ranked) if d in rel), 0.0)
        rrs.append(rr)
    return float(np.mean(rrs))


def _map(sim: np.ndarray, relevant: List[Set[int]]) -> float:
    if not relevant:
        return 0.0
    aps = []
    for i, rel in enumerate(relevant):
        if not rel:
            aps.append(1.0)
            continue
        ranked = np.argsort(sim[i])[::-1].tolist()
        hits, psum = 0, 0.0
        for rank, d in enumerate(ranked, start=1):
            if d in rel:
                hits += 1
                psum += hits / rank
        aps.append(psum / len(rel))
    return float(np.mean(aps))


def embedding_evaluate(
    query_embeddings: np.ndarray,
    corpus_embeddings: np.ndarray,
    relevant_indices: List[Set[int]],
) -> EmbeddingMetrics:
    sim = _pairwise_cosine(query_embeddings, corpus_embeddings)
    return EmbeddingMetrics(
        recall_at_1=_recall_at_k(sim, relevant_indices, 1),
        recall_at_5=_recall_at_k(sim, relevant_indices, 5),
        recall_at_10=_recall_at_k(sim, relevant_indices, 10),
        mrr=_mrr(sim, relevant_indices),
        map=_map(sim, relevant_indices),
    )
