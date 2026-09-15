from __future__ import annotations

import logging
from typing import Dict, List, Set

from rdflib import Graph, RDFS, URIRef

logger = logging.getLogger(__name__)

_MIN_LABEL_LENGTH = 4


class EntityLinker:
    def __init__(self, label_index: Dict[str, URIRef]):
        self._label_index = label_index
        self._labels_by_length = sorted(label_index, key=len, reverse=True)

    @property
    def label_index(self) -> Dict[str, URIRef]:
        return self._label_index

    @classmethod
    def from_graph(cls, graph: Graph) -> "EntityLinker":
        index: Dict[str, URIRef] = {}
        for subj, label in graph.subject_objects(predicate=RDFS.label):
            if not isinstance(subj, URIRef):
                continue
            normalized = str(label).strip().lower()
            if len(normalized) < _MIN_LABEL_LENGTH:
                continue
            index[normalized] = subj
        logger.info("Entity linker index built: %d labels.", len(index))
        return cls(index)

    def link(self, text: str, max_entities: int = 12) -> List[URIRef]:
        text_lower = text.lower()
        matched: List[URIRef] = []
        seen: Set[URIRef] = set()

        for label in self._labels_by_length:
            if label in text_lower:
                uri = self._label_index[label]
                if uri not in seen:
                    matched.append(uri)
                    seen.add(uri)
                    if len(matched) >= max_entities:
                        break
        return matched
