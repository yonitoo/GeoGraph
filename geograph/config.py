from __future__ import annotations

import logging
from dataclasses import dataclass, field
from rdflib import Namespace, URIRef
from typing import List, Set, Optional

from geograph.constants import (
    TOP_K, CHROMA_COLLECTION_NAME, TTL_FILE_PATH, GEOGRAPHBG,
    EMBEDDING_MODEL_NAME, TOKENIZER_NAME, BGGPT_MODEL, BGGPT_V2_API_URL,
    ONT_QUANTITY_PROPERTIES, ONT_VALUE_PROPERTY, ONT_UNIT_PROPERTY,
    REASONING_MODEL_PROMPT, OPENAI_API_KEY, GRAPHDB_REPO_ENDPOINT,
    OPENAI_REASONING_MODEL_SYSTEM_PROMPT, MISSING_VALUE_STRING,
    UNWANTED_TRIPLE_ENDINGS, BGGPT_API_KEY, BGGPT_V3_MODEL, BGGPT_V3_API_URL,
    CHROMA_DB_PATH, EXPANSION_COARSE_TOP_K, EXPANSION_RERANK_TOP_K,
    EXPANSION_PER_SEED_LIMIT, EXPANSION_BEAM_WIDTH, EXPANSION_FRONTIER_PER_SEED,
    REASONING_MIN_KEEP,
)

logger = logging.getLogger(__name__)


@dataclass
class OntologyConfig:
    namespace: Namespace
    quantity_properties: List[str] = field(default_factory=list)
    value_property: str = ""
    unit_property: str = ""
    missing_value_string: str = "[ЛИПСВАЩА СТОЙНОСТ]"
    unwanted_triple_endings: List[str] = field(default_factory=list)

    @property
    def quantity_property_uris(self) -> Set[URIRef]:
        return {self.namespace[p] for p in self.quantity_properties}

    @property
    def value_property_uri(self) -> Optional[URIRef]:
        return self.namespace[self.value_property] if self.value_property else None

    @property
    def unit_property_uri(self) -> Optional[URIRef]:
        return self.namespace[self.unit_property] if self.unit_property else None


@dataclass
class RetrievalConfig:
    top_k: int = 10
    top_k_initial_multiplier: int = 30
    chroma_collection_name: str = "geographbg_triples"
    chroma_batch_size: int = 5000
    chroma_db_path: str = "./data/persisted_db_data"


@dataclass
class SPARQLConfig:
    endpoint_url: str = ""
    max_hops: int = 2
    max_related_by_type: int = 30
    timeout: int = 60
    expansion_coarse_top_k: int = 300
    expansion_rerank_top_k: int = 100
    expansion_per_seed_limit: int = 100
    expansion_beam_width: int = 50
    expansion_frontier_per_seed: int = 20


@dataclass
class ReasoningConfig:
    api_key: str = ""
    model: str = "o4-mini"
    effort: str = "medium"
    system_prompt: str = ""
    filter_prompt_template: str = ""
    min_keep: int = 10
    enabled: bool = True


@dataclass
class LLMConfig:
    api_key: str = ""
    model: str = ""
    api_url: str = "https://api.bggpt.ai/completions"
    max_tokens: int = 1000
    temperature: float = 0.0
    top_k: int = 20
    repetition_penalty: float = 1.1


@dataclass
class EmbeddingConfig:
    model_name: str = ""
    tokenizer_name: str = ""
    max_length: int = 512


@dataclass
class PipelineConfig:
    ttl_path: str = ""
    ontology: OntologyConfig = field(default_factory=lambda: OntologyConfig(namespace=Namespace("")))
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    sparql: SPARQLConfig = field(default_factory=SPARQLConfig)
    reasoning: ReasoningConfig = field(default_factory=ReasoningConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)

    @staticmethod
    def from_constants() -> PipelineConfig:
        return PipelineConfig(
            ttl_path=TTL_FILE_PATH,
            ontology=OntologyConfig(
                namespace=GEOGRAPHBG,
                quantity_properties=list(ONT_QUANTITY_PROPERTIES),
                value_property=ONT_VALUE_PROPERTY,
                unit_property=ONT_UNIT_PROPERTY,
                missing_value_string=MISSING_VALUE_STRING,
                unwanted_triple_endings=list(UNWANTED_TRIPLE_ENDINGS),
            ),
            embedding=EmbeddingConfig(
                model_name=EMBEDDING_MODEL_NAME,
                tokenizer_name=TOKENIZER_NAME,
            ),
            retrieval=RetrievalConfig(
                top_k=TOP_K,
                chroma_collection_name=CHROMA_COLLECTION_NAME,
                chroma_db_path=CHROMA_DB_PATH,
            ),
            sparql=SPARQLConfig(
                endpoint_url=GRAPHDB_REPO_ENDPOINT,
                expansion_coarse_top_k=EXPANSION_COARSE_TOP_K,
                expansion_rerank_top_k=EXPANSION_RERANK_TOP_K,
                expansion_per_seed_limit=EXPANSION_PER_SEED_LIMIT,
                expansion_beam_width=EXPANSION_BEAM_WIDTH,
                expansion_frontier_per_seed=EXPANSION_FRONTIER_PER_SEED,
            ),
            reasoning=ReasoningConfig(
                api_key=OPENAI_API_KEY,
                system_prompt=OPENAI_REASONING_MODEL_SYSTEM_PROMPT,
                filter_prompt_template=REASONING_MODEL_PROMPT,
                min_keep=REASONING_MIN_KEEP,
            ),
            llm=LLMConfig(
                api_key=BGGPT_API_KEY,
                model=BGGPT_V3_MODEL,
                api_url=BGGPT_V3_API_URL,
            ),
        )


def llm_configs_for_baselines() -> dict[str, "LLMConfig"]:
    return {
        "bggpt-gemma2-27b": LLMConfig(
            api_key=BGGPT_API_KEY,
            model=BGGPT_MODEL,
            api_url=BGGPT_V2_API_URL,
        ),
        "bggpt-gemma3-27b": LLMConfig(
            api_key=BGGPT_API_KEY,
            model=BGGPT_V3_MODEL,
            api_url=BGGPT_V3_API_URL,
        ),
    }


@dataclass
class ExperimentConfig:
    pipeline_type: str = "kg_rag"  # "kg_rag" | "vector_rag" | "both"
    input_file: str = "testset/demo.jsonl"
    output_file: str = "testset/output.jsonl"
    run_ingestion: bool = True
    log_dir: str = "logs"
