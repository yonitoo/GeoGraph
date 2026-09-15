import logging
from typing import Optional

from geograph.config import PipelineConfig
from geograph.retrieval.embedder import Embedder
from geograph.ontology.rdf_processor import RDFProcessor
from geograph.retrieval.vector_store import VectorStore, create_chroma_client
from geograph.ontology.sparql_expander import SPARQLExpander
from geograph.ontology.reasoning_filter import ReasoningFilter
from geograph.llm.llm_client import BgGPTClient

logger = logging.getLogger(__name__)


class ComponentRegistry:
    def __init__(self) -> None:
        self.config: Optional[PipelineConfig] = None
        self._embedder: Optional[Embedder] = None
        self._rdf_processor: Optional[RDFProcessor] = None
        self._vector_store: Optional[VectorStore] = None
        self._sparql_expander: Optional[SPARQLExpander] = None
        self._reasoning_filter: Optional[ReasoningFilter] = None
        self._llm_client: Optional[BgGPTClient] = None
        self._semantic_strings: list = []
        self._metadata_list: list = []

    def _require_config(self) -> PipelineConfig:
        if self.config is None:
            raise RuntimeError("Config not loaded. Server did not start correctly.")
        return self.config


    @property
    def embedder(self) -> Embedder:
        if self._embedder is None:
            cfg = self._require_config()
            logger.info("Initializing embedder (model: %s)...", cfg.embedding.model_name)
            self._embedder = Embedder(cfg.embedding)
            logger.info("Embedder ready on %s", self._embedder.device)
        return self._embedder

    @property
    def embedder_ready(self) -> bool:
        return self._embedder is not None


    @property
    def rdf_processor(self) -> RDFProcessor:
        if self._rdf_processor is None:
            raise RuntimeError("RDF not loaded. Call POST /api/rdf/load first.")
        return self._rdf_processor

    def load_rdf(self, ttl_path: str) -> RDFProcessor:
        cfg = self._require_config()
        self._rdf_processor = RDFProcessor.load(ttl_path, cfg.ontology)
        return self._rdf_processor

    def process_triples(self):
        rdf = self.rdf_processor
        self._semantic_strings, self._metadata_list = rdf.process_triples()
        return self._semantic_strings, self._metadata_list

    @property
    def rdf_ready(self) -> bool:
        return self._rdf_processor is not None

    @property
    def triples_processed(self) -> bool:
        return len(self._semantic_strings) > 0


    @property
    def vector_store(self) -> VectorStore:
        if self._vector_store is None:
            cfg = self._require_config()
            client = create_chroma_client(cfg.retrieval)
            self._vector_store = VectorStore(client, cfg.retrieval)
        return self._vector_store

    @property
    def vector_store_ready(self) -> bool:
        return self._vector_store is not None and self._vector_store.collection is not None


    @property
    def sparql_expander(self) -> SPARQLExpander:
        if self._sparql_expander is None:
            cfg = self._require_config()
            if not cfg.sparql.endpoint_url:
                raise RuntimeError("SPARQL endpoint not configured. Set GRAPHDB_REPO_ENDPOINT in constants.")
            self._sparql_expander = SPARQLExpander(cfg.sparql, cfg.ontology)
        return self._sparql_expander

    @property
    def sparql_ready(self) -> bool:
        return self._sparql_expander is not None


    @property
    def reasoning_filter(self) -> ReasoningFilter:
        if self._reasoning_filter is None:
            cfg = self._require_config()
            if not cfg.reasoning.api_key:
                raise RuntimeError("OpenAI API key not set. Set OPENAI_API_KEY in constants.")
            self._reasoning_filter = ReasoningFilter(cfg.reasoning, cfg.ontology)
        return self._reasoning_filter

    @property
    def reasoning_ready(self) -> bool:
        return self._reasoning_filter is not None


    @property
    def llm_client(self) -> BgGPTClient:
        if self._llm_client is None:
            cfg = self._require_config()
            if not cfg.llm.api_key:
                raise RuntimeError("BgGPT API key not set. Set BGGPT_API_KEY in constants.")
            self._llm_client = BgGPTClient(cfg.llm)
        return self._llm_client

    @property
    def llm_ready(self) -> bool:
        return self._llm_client is not None

    def status(self) -> dict:
        cfg = self._require_config()
        return {
            "config": True,
            "embedder": self.embedder_ready,
            "rdf_processor": self.rdf_ready,
            "triples_processed": self.triples_processed,
            "vector_store": self.vector_store_ready,
            "sparql_expander": self.sparql_ready,
            "reasoning_filter": self.reasoning_ready,
            "llm_client": self.llm_ready,
            "has_openai_key": bool(cfg.reasoning.api_key),
            "has_bggpt_key": bool(cfg.llm.api_key),
            "has_sparql_endpoint": bool(cfg.sparql.endpoint_url),
        }


registry = ComponentRegistry()
