import logging
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from geograph.config import PipelineConfig
from geograph.retrieval.embedder import Embedder
from geograph.retrieval.embedding_cache import EmbeddingCache
from geograph.ontology.entity_linker import EntityLinker
from geograph.llm.llm_client import BgGPTChatClient
from geograph.llm.prompt_builder import build_grounded_prompt
from geograph.ontology.rdf_processor import RDFProcessor
from geograph.ontology.reasoning_filter import ReasoningFilter
from geograph.retrieval.reranker import CrossEncoderReranker
from geograph.ontology.sparql_expander import SPARQLExpander
from geograph.retrieval.vector_store import VectorStore, create_chroma_client

logger = logging.getLogger(__name__)

_EMPTY_CONTEXT_POLICIES = {"fallback", "abstain"}


@dataclass
class QueryTimings:
    retrieval_s: float = 0.0
    expansion_s: float = 0.0
    rerank_s: float = 0.0
    filtering_s: float = 0.0
    generation_s: float = 0.0

    @property
    def total_s(self) -> float:
        return self.retrieval_s + self.expansion_s + self.rerank_s + self.filtering_s + self.generation_s

    def summary(self) -> str:
        return (
            f"retrieval={self.retrieval_s:.2f}s, expansion={self.expansion_s:.2f}s, "
            f"rerank={self.rerank_s:.2f}s, filtering={self.filtering_s:.2f}s, "
            f"generation={self.generation_s:.2f}s, total={self.total_s:.2f}s"
        )


@dataclass
class QueryDiagnostics:
    n_seeds: int = 0
    n_linked_entities: int = 0
    n_expanded: int = 0
    n_protected: int = 0
    n_reranked: int = 0
    n_prefiltered: int = 0
    n_filtered: int = 0
    filter_llm_used: bool = False
    used_fallback_context: bool = False
    seed_docs: List[str] = field(default_factory=list)
    expanded_triples: List[str] = field(default_factory=list)
    context_triples: List[str] = field(default_factory=list)


class KGRAGPipeline:
    """Knowledge Graph-based Retrieval-Augmented Generation pipeline.

    Flow: Vector Search + Entity Linking → Per-Seed Relevance-Guided SPARQL
    Expansion → Rerank → Reasoning Filter → Grounded Generation
    """

    def __init__(
        self,
        config: PipelineConfig,
        embed_batch_size: int = 32,
        empty_context_policy: str = "fallback",
        fallback_top_k: int = 30,
    ):
        if empty_context_policy not in _EMPTY_CONTEXT_POLICIES:
            raise ValueError(
                f"Invalid empty_context_policy '{empty_context_policy}'. "
                f"Use one of {sorted(_EMPTY_CONTEXT_POLICIES)}."
            )

        self.config = config
        self.embed_batch_size = embed_batch_size
        self.empty_context_policy = empty_context_policy
        self.fallback_top_k = fallback_top_k

        self.embedder = Embedder(config.embedding)
        self.rdf_processor: Optional[RDFProcessor] = None
        self.embedding_cache: Optional[EmbeddingCache] = None
        self.entity_linker: Optional[EntityLinker] = None
        self.vector_store = VectorStore(create_chroma_client(config.retrieval), config.retrieval)
        self.sparql_expander = SPARQLExpander(config.sparql, config.ontology)
        self.expansion_reranker = CrossEncoderReranker()
        self.reasoning_filter = ReasoningFilter(config.reasoning, config.ontology)
        self.llm_client = BgGPTChatClient(config.llm)

    def ingest(self) -> None:
        """Load RDF graph, process triples, embed and store in vector DB."""
        self.rdf_processor = RDFProcessor.load(self.config.ttl_path, self.config.ontology)
        semantic_strings, metadata_list = self.rdf_processor.process_triples()
        self.vector_store.ingest(
            semantic_strings, metadata_list, self.embedder, embed_batch_size=self.embed_batch_size,
        )
        self._finalize_setup()
        logger.info("KG-RAG ingestion complete.")

    def load_graph_only(self) -> None:
        """Load RDF graph and existing vector store (skip ingestion)."""
        self.rdf_processor = RDFProcessor.load(self.config.ttl_path, self.config.ontology)
        self.vector_store.load_existing()
        self._finalize_setup()

    def _finalize_setup(self) -> None:
        """Build the embedding cache and entity linker once the graph and vector
        store are both ready — shared by ingest() and load_graph_only()."""
        self.embedding_cache = self.vector_store.load_embedding_cache(self.embedder)
        self.entity_linker = EntityLinker.from_graph(self.rdf_processor.graph)

    def query(self, user_query: str) -> Tuple[str, QueryTimings, QueryDiagnostics]:
        """Execute the full KG-RAG pipeline for a user query."""
        if self.rdf_processor is None:
            raise RuntimeError("Pipeline not initialized. Call ingest() or load_graph_only() first.")

        timings = QueryTimings()
        diagnostics = QueryDiagnostics()

        # vec retrieval with re-ranking
        t0 = time.time()
        retrieved_metadata = self.vector_store.retrieve_and_rerank(
            user_query, self.embedder, self.config.ontology, self.rdf_processor,
        )
        timings.retrieval_s = time.time() - t0
        diagnostics.seed_docs = [item.get("document", "") for item in retrieved_metadata]
        diagnostics.n_seeds = len(retrieved_metadata)
        logger.info("Retrieval: %d results in %.2fs", len(retrieved_metadata), timings.retrieval_s)

        if not retrieved_metadata:
            logger.warning("No metadata retrieved. Context will be empty.")

        # Step 1b: link KG entities named in the question
        linked_entities = self.entity_linker.link(user_query)
        diagnostics.n_linked_entities = len(linked_entities)
        logger.info("Entity linking: %d entities found in query text.", len(linked_entities))

        # Per-seed, relevance-guided SPARQL expansion
        t0 = time.time()
        expansion = self.sparql_expander.expand(
            self.rdf_processor, retrieved_metadata, user_query, self._score_relevance,
            extra_seed_entities=set(linked_entities),
        )
        expanded_triples, protected_triples = expansion.semantic_strings, expansion.protected_strings
        timings.expansion_s = time.time() - t0
        diagnostics.expanded_triples = expanded_triples
        diagnostics.n_expanded = len(expanded_triples)
        diagnostics.n_protected = len(protected_triples)
        logger.info(
            "Expansion: %d triples (%d protected) in %.2fs",
            len(expanded_triples), len(protected_triples), timings.expansion_s,
        )

        # two-stage funnel before the reasoning filter
        t0 = time.time()
        protected_kept = [t for t in expanded_triples if t in protected_triples]
        prunable = [t for t in expanded_triples if t not in protected_triples]

        coarse_budget = max(self.config.sparql.expansion_coarse_top_k - len(protected_kept), 0)
        coarse_survivors = self._coarse_filter_by_similarity(user_query, prunable, coarse_budget)

        rerank_budget = max(self.config.sparql.expansion_rerank_top_k - len(protected_kept), 0)
        ranked_indices = self.expansion_reranker.rerank(user_query, coarse_survivors, rerank_budget)
        reranked_triples = protected_kept + [coarse_survivors[i] for i in ranked_indices]

        timings.rerank_s = time.time() - t0
        diagnostics.n_reranked = len(reranked_triples)
        logger.info(
            "Rerank: %d → %d (coarse) → %d (cross-encoder, %d protected) in %.2fs",
            len(expanded_triples), len(protected_kept) + len(coarse_survivors), len(reranked_triples),
            len(protected_kept), timings.rerank_s,
        )

        # Reasoning-based filtering
        t0 = time.time()
        filter_result = self.reasoning_filter.filter(reranked_triples, user_query)
        timings.filtering_s = time.time() - t0
        diagnostics.n_prefiltered = filter_result.n_prefiltered
        diagnostics.n_filtered = len(filter_result.triples)
        diagnostics.filter_llm_used = filter_result.llm_used
        logger.info("Filtering: %d triples in %.2fs", len(filter_result.triples), timings.filtering_s)

        context_triples = filter_result.triples
        if not context_triples and self.empty_context_policy == "fallback":
            context_triples = reranked_triples[: self.fallback_top_k]
            diagnostics.used_fallback_context = bool(context_triples)
            logger.info(
                "Reasoning filter returned nothing — falling back to top %d reranked triples.",
                len(context_triples),
            )
        diagnostics.context_triples = context_triples

        if not context_triples and self.empty_context_policy == "abstain":
            logger.info("No context available and policy is 'abstain' — skipping generation.")
            return "-", timings, diagnostics

        # Prompt construction and gen
        prompt = build_grounded_prompt(context_triples, user_query)
        t0 = time.time()
        response = self.llm_client.generate(prompt)
        timings.generation_s = time.time() - t0
        logger.info("Generation in %.2fs", timings.generation_s)

        return response, timings, diagnostics

    def _score_relevance(self, query: str, candidates: List[str]) -> List[float]:
        if not candidates:
            return []
        candidate_vectors = self.embedding_cache.embed_many(candidates)
        candidate_vectors = candidate_vectors / (np.linalg.norm(candidate_vectors, axis=1, keepdims=True) + 1e-9)

        query_vector = np.array(self.embedder.embed(query))
        query_vector = query_vector / (np.linalg.norm(query_vector) + 1e-9)

        return (candidate_vectors @ query_vector).tolist()

    def _coarse_filter_by_similarity(
        self, query: str, candidates: List[str], top_k: int,
    ) -> List[str]:
        if len(candidates) <= top_k:
            return candidates

        scores = self._score_relevance(query, candidates)
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [candidates[i] for i in top_indices]
