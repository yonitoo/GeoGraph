import logging
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)

_STOP_WORDS = {
    "в", "на", "са", "кои", "е", "с", "и", "или", "до", "от", "за", "под", "над", "през",
}


class LexicalReranker:
    """Ranks candidates by deduplicated keyword-overlap score against the query."""

    def rerank(self, query: str, candidates: List[str], top_k: int) -> List[int]:
        """Score all candidates by keyword overlap and return top_k indices."""
        if not candidates:
            return []
        top_k = min(top_k, len(candidates))
        keywords = {
            kw for kw in query.lower().split()
            if kw not in _STOP_WORDS and len(kw) > 2
        }
        scores = [
            sum(len(kw) * 2 for kw in keywords if kw in c.lower())
            for c in candidates
        ]
        ranked = np.argsort(scores)[::-1][:top_k]
        return [int(i) for i in ranked]


_STEM_MIN = 4
_STEM_TRIM = 3


def _stem(word: str) -> str:
    return word[: max(_STEM_MIN, len(word) - _STEM_TRIM)]


class StemLexicalReranker:
    def rerank(self, query: str, candidates: List[str], top_k: int) -> List[int]:
        if not candidates:
            return []
        top_k = min(top_k, len(candidates))
        keywords = {
            kw for kw in query.lower().split()
            if kw not in _STOP_WORDS and len(kw) > 2
        }
        stems = {_stem(kw) for kw in keywords}
        scores = []
        for c in candidates:
            c_lower = c.lower()
            c_words = c_lower.split()
            c_stems = {_stem(w) for w in c_words if len(w) > 2}
            exact_score = sum(len(kw) * 2 for kw in keywords if kw in c_lower)
            stem_score = sum(len(s) for s in stems if s in c_stems)
            scores.append(exact_score + stem_score)
        ranked = np.argsort(scores)[::-1][:top_k]
        return [int(i) for i in ranked]


class CrossEncoderReranker:
    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        device: Optional[str] = None,
    ) -> None:
        self.model_name = model_name
        self._device = device
        self._model = None  # lazy

    def _load(self) -> None:
        if self._model is not None:
            return
        logger.info("Loading cross-encoder reranker: %s", self.model_name)
        from sentence_transformers import CrossEncoder
        self._model = CrossEncoder(self.model_name, device=self._device)
        logger.info("Reranker ready.")

    def rerank(self, query: str, candidates: List[str], top_k: int) -> List[int]:
        """Score all (query, candidate) pairs and return top_k indices by score."""
        if not candidates:
            return []
        self._load()
        top_k = min(top_k, len(candidates))
        pairs = [(query, c) for c in candidates]
        scores = self._model.predict(pairs, show_progress_bar=False)
        ranked = sorted(range(len(scores)), key=lambda i: float(scores[i]), reverse=True)
        return ranked[:top_k]
