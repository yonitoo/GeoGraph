from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np

from geograph.retrieval.embedder import Embedder

logger = logging.getLogger(__name__)


def cosine_scores(embedder: Embedder, query: str, candidates: List[str]) -> List[float]:
    """Cosine similarity of each candidate's embedding against the query embedding."""
    if not candidates:
        return []
    candidate_vectors = np.array(embedder.embed_batch(candidates))
    candidate_vectors = candidate_vectors / (np.linalg.norm(candidate_vectors, axis=1, keepdims=True) + 1e-9)
    query_vector = np.array(embedder.embed(query))
    query_vector = query_vector / (np.linalg.norm(query_vector) + 1e-9)
    return (candidate_vectors @ query_vector).tolist()


class EmbeddingCache:
    def __init__(self, embedder: Embedder, cached: Dict[str, np.ndarray]):
        self._embedder = embedder
        self._cache = cached
        self.hits = 0
        self.misses = 0

    @classmethod
    def from_documents(
        cls, embedder: Embedder, documents: List[str], embeddings: List[List[float]],
    ) -> "EmbeddingCache":
        cached = {doc: np.array(vec) for doc, vec in zip(documents, embeddings)}
        return cls(embedder, cached)

    def embed_many(self, texts: List[str]) -> np.ndarray:
        """Return an (n, dim) array of embeddings, reusing cached vectors where possible."""
        vectors: List[Optional[np.ndarray]] = [self._cache.get(t) for t in texts]
        miss_indices = [i for i, v in enumerate(vectors) if v is None]

        self.hits += len(texts) - len(miss_indices)
        self.misses += len(miss_indices)

        if miss_indices:
            fresh = self._embedder.embed_batch([texts[i] for i in miss_indices])
            for idx, vec in zip(miss_indices, fresh):
                vectors[idx] = np.array(vec)

        return np.array(vectors)

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0
