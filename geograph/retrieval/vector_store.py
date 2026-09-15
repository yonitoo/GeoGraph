from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import chromadb
import torch
from rdflib import URIRef
from tqdm import tqdm

from geograph.config import OntologyConfig, RetrievalConfig
from geograph.retrieval.embedder import Embedder
from geograph.retrieval.embedding_cache import EmbeddingCache
from geograph.ontology.rdf_processor import RDFProcessor

logger = logging.getLogger(__name__)


def create_chroma_client(config: RetrievalConfig) -> chromadb.ClientAPI:
    """Create a ChromaDB client — persistent if db_path is set, in-memory otherwise."""
    if config.chroma_db_path:
        import os
        os.makedirs(config.chroma_db_path, exist_ok=True)
        client = chromadb.PersistentClient(path=config.chroma_db_path)
        logger.info("ChromaDB persistent client at %s", config.chroma_db_path)
    else:
        client = chromadb.Client()
        logger.info("ChromaDB in-memory client")
    return client


class VectorStore:
    """Manages ChromaDB ingestion and retrieval of embedded semantic triples."""

    def __init__(self, client: chromadb.ClientAPI, config: RetrievalConfig):
        self.client = client
        self.config = config
        self.collection: Optional[chromadb.Collection] = None

    def ingest(
        self,
        semantic_strings: List[str],
        metadata_list: List[Dict[str, Any]],
        embedder: Embedder,
        embed_batch_size: int = 32,
    ) -> None:
        """Embed and ingest semantic strings into ChromaDB."""
        if not semantic_strings:
            logger.error("No semantic strings to ingest.")
            self.collection = self.client.get_or_create_collection(
                name=self.config.chroma_collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            return

        logger.info(
            "Embedding %d semantic strings in batches of %d...",
            len(semantic_strings), embed_batch_size,
        )
        embeddings: List[List[float]] = []
        is_mps = embedder.device.type == "mps"
        for start in tqdm(range(0, len(semantic_strings), embed_batch_size), desc="Embedding"):
            batch = semantic_strings[start : start + embed_batch_size]
            embeddings.extend(embedder.embed_batch(batch, batch_size=embed_batch_size))
            if is_mps:
                torch.mps.empty_cache()

        ids = [str(i) for i in range(len(semantic_strings))]

        try:
            existing = self.client.get_collection(name=self.config.chroma_collection_name)
            self.client.delete_collection(name=existing.name)
        except Exception:
            pass
        self.collection = self.client.create_collection(
            name=self.config.chroma_collection_name,
            metadata={"hnsw:space": "cosine"}
        )

        batch_size = self.config.chroma_batch_size
        for i in tqdm(range(0, len(ids), batch_size), desc="Ingesting to ChromaDB"):
            end = i + batch_size
            self.collection.add(
                ids=ids[i:end],
                documents=semantic_strings[i:end],
                embeddings=embeddings[i:end],
                metadatas=metadata_list[i:end],
            )
        logger.info("Ingestion complete: %d items.", len(ids))

    def load_existing(self) -> None:
        """Load an existing ChromaDB collection."""
        self.collection = self.client.get_collection(name=self.config.chroma_collection_name)
        logger.info("Loaded collection '%s' with %d items.", self.config.chroma_collection_name, self.collection.count())

    def load_embedding_cache(self, embedder: Embedder) -> EmbeddingCache:
        """Fetch all ingested documents+embeddings once for reuse as a lookup cache
        (avoids re-embedding candidates that were already embedded during ingestion)."""
        if self.collection is None:
            raise RuntimeError("Collection not initialized. Call ingest() or load_existing() first.")
        data = self.collection.get(include=["documents", "embeddings"])
        logger.info("Loaded embedding cache: %d documents.", len(data["documents"]))
        return EmbeddingCache.from_documents(embedder, data["documents"], data["embeddings"])

    def change_distance_metric(self, new_metric: str = "cosine") -> None:
        """Change the distance metric of an existing collection without re-embedding."""
        if new_metric not in ("cosine", "l2", "ip"):
            raise ValueError(f"Invalid metric: {new_metric}. Use 'cosine', 'l2', or 'ip'.")

        old_collection = self.client.get_collection(name=self.config.chroma_collection_name)
        count = old_collection.count()

        logger.info(
            "Changing metric for collection '%s' (%d items) from %s to %s...",
            self.config.chroma_collection_name,
            count,
            old_collection.metadata.get("hnsw:space", "l2") if old_collection.metadata else "l2",
            new_metric,
        )

        logger.info("Extracting embeddings and metadata...")
        all_data = old_collection.get(include=["embeddings", "documents", "metadatas"])

        ids = all_data["ids"]
        embeddings = all_data["embeddings"]
        documents = all_data["documents"]
        metadatas = all_data["metadatas"]

        self.client.delete_collection(name=self.config.chroma_collection_name)
        logger.info("Deleted old collection.")

        self.collection = self.client.create_collection(
            name=self.config.chroma_collection_name,
            metadata={"hnsw:space": new_metric}
        )
        logger.info("Created new collection with metric '%s'.", new_metric)

        batch_size = self.config.chroma_batch_size
        for i in tqdm(range(0, len(ids), batch_size), desc="Re-ingesting with new metric"):
            end = i + batch_size
            self.collection.add(
                ids=ids[i:end],
                documents=documents[i:end],
                embeddings=embeddings[i:end],
                metadatas=metadatas[i:end],
            )

        logger.info("Metric change complete: %d items re-added.", len(ids))

    def query_raw(self, query_embedding: List[float], top_k: int) -> QueryResult:
        """Raw vector similarity search."""
        if self.collection is None:
            raise RuntimeError("Collection not initialized. Call ingest() or load_existing() first.")
        return self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["metadatas", "documents", "distances"],
        )

    def retrieve_and_rerank(
        self,
        user_query: str,
        embedder: Embedder,
        ont_config: OntologyConfig,
        rdf_processor: Optional[RDFProcessor] = None,
        top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieve triples with vector search and apply domain-aware re-ranking."""
        final_top_k = top_k or self.config.top_k
        initial_top_k = final_top_k * self.config.top_k_initial_multiplier

        query_for_embedding = _strip_country_from_query(user_query)
        query_embedding = embedder.embed(query_for_embedding)

        try:
            results = self.query_raw(query_embedding, initial_top_k)
        except Exception as e:
            logger.error("Error querying ChromaDB: %s", e, exc_info=True)
            return []

        all_metadatas = results.get("metadatas", [[]])[0]
        all_documents = results.get("documents", [[]])[0]
        all_distances = results.get("distances", [[]])[0]

        if not all_metadatas:
            return []

        combined = [
            {"metadata": all_metadatas[i], "document": all_documents[i], "distance": all_distances[i]}
            for i in range(len(all_metadatas))
        ]

        ranker = _ReRanker(user_query, ont_config, rdf_processor)
        combined.sort(key=ranker.score)

        reranked = [{**res["metadata"], "document": res["document"]} for res in combined[:final_top_k]]

        logger.info("Retrieved and re-ranked: %d initial → %d final.", len(all_metadatas), len(reranked))

        if rdf_processor:
            for i, res in enumerate(combined[:final_top_k + 5]):
                s_uri = res["metadata"].get("subject_uri")
                s_label = rdf_processor.get_readable_label(URIRef(s_uri)) if s_uri else "N/A"
                score = ranker.score(res)
                logger.info("  Top %d: \"%s\" (Score: %.2f) S: %s", i + 1, res["document"], score, s_label)

        return reranked


class _ReRanker:
    """Encapsulates the re-ranking scoring logic."""

    _STOP_WORDS = {
        "в", "на", "са", "кои", "е", "с", "и", "или", "до", "от", "за", "под", "над", "през",
    }
    _QUANTITY_KEYWORDS = [
        "колко", "висок", "дълг", "дълж", "площ", "дълбок", "обем", "населен",
        "дебит", "температур", "капацитет", "най-",
    ]
    _QUANTITY_AFFINITY = {
        "населен": ["население"],
        "висок": ["височина"],
        "височин": ["височина"],
        "дълг": ["дължина"],
        "дълж": ["дължина"],
        "площ": ["площ"],
        "дълбок": ["дълбочина"],
        "дълбочин": ["дълбочина"],
        "обем": ["обем"],
        "дебит": ["дебит"],
        "температур": ["температура"],
        "капацитет": ["капацитет"],
    }

    def __init__(self, user_query: str, ont_config: OntologyConfig, rdf_processor: Optional[RDFProcessor]):
        self.query_lower = user_query.lower()
        self.ont = ont_config
        self.rdf_processor = rdf_processor
        self._quantity_uri_strs = {str(u) for u in ont_config.quantity_property_uris}
        self._is_quantity_query = any(kw in self.query_lower for kw in self._QUANTITY_KEYWORDS)
        self._query_keywords = {
            kw for kw in self.query_lower.split()
            if kw not in self._STOP_WORDS and len(kw) > 2
        }

    def score(self, item: Dict[str, Any]) -> float:
        distance = item["distance"]
        meta = item["metadata"]
        doc_lower = item["document"].lower()

        pred_importance = meta.get("predicate_importance", 99) * 1.5
        penalty = self._country_penalty(meta)
        bonus = self._quantity_bonus(meta, doc_lower) + self._keyword_bonus(doc_lower)

        return distance * 0.8 + pred_importance + penalty + bonus

    def _country_penalty(self, meta: Dict[str, Any]) -> float:
        pred_uri = meta.get("predicate_uri", "")
        obj_uri = meta.get("object_uri", "")

        is_country_bg = (
            pred_uri == str(self.ont.namespace.държава)
            and obj_uri == str(self.ont.namespace.България)
        )
        if not is_country_bg:
            return 0.0

        if any(kw in self.query_lower for kw in ("държава", "българия", "къде се намира")):
            return 0.0

        if any(kw in self.query_lower for kw in ("височина", "дължина", "площ", "колко е", "най-", "тип")):
            return 60.0
        return 30.0

    def _quantity_bonus(self, meta: Dict[str, Any], doc_lower: str) -> float:
        if not self._is_quantity_query:
            return 0.0
        if meta.get("predicate_uri") not in self._quantity_uri_strs:
            return 0.0

        pred_local = meta.get("predicate_uri", "").split("#")[-1].lower()

        has_affinity = False
        for query_kw, pred_fragments in self._QUANTITY_AFFINITY.items():
            if query_kw in self.query_lower:
                if any(frag in pred_local for frag in pred_fragments):
                    has_affinity = True
                    break

        bonus = -60.0 if has_affinity else -15.0

        subject_types = meta.get("subject_types", "").lower()
        for word in self.query_lower.split():
            if len(word) > 3 and word in subject_types and word not in ("река", "планина", "езеро", "връх"):
                if word in doc_lower:
                    bonus -= 5.0
                    break
        return bonus

    _MAX_KEYWORD_BONUS_MAGNITUDE = 20.0  # keeps keyword overlap a tiebreaker, not the dominant signal

    def _keyword_bonus(self, doc_lower: str) -> float:
        bonus = 0.0
        for kw in self._query_keywords:
            if kw in doc_lower:
                bonus -= len(kw) * 2
        return max(bonus, -self._MAX_KEYWORD_BONUS_MAGNITUDE)


def _strip_country_from_query(query: str) -> str:
    """Remove 'България' from query to improve embedding relevance."""
    lower = query.lower()
    if "българия" not in lower:
        return query
    stripped = lower.replace("в българия", "").replace("българия", "").strip()
    if len(stripped.split()) > 1 or (len(stripped.split()) == 1 and len(stripped) > 3):
        logger.info("Modified query for embedding: '%s' → '%s'", query, stripped)
        return stripped
    return query
