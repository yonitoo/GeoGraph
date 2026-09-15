from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class EmbedRequest(BaseModel):
    text: Optional[str] = None
    texts: Optional[List[str]] = None


class LoadRDFRequest(BaseModel):
    ttl_path: str = "ontologies/huge_ontology_v3.ttl"


class IngestRequest(BaseModel):
    ttl_path: str = "ontologies/huge_ontology_v3.ttl"


class IngestWikiRequest(BaseModel):
    data_dir: str = "data/cleaned_data"
    chunk_size: int = 500
    chunk_overlap: int = 50
    excluded_files: List[str] = [
        "fetched-wiki-bg-geo-in-bg.json",
        "fetched-wiki-bg-geo-okolna-sreda.json",
        "fetched-wiki-bg-geo-admin-split.json",
    ]


class QueryRequest(BaseModel):
    query: str
    top_k: int = 5


class SPARQLExpandRequest(BaseModel):
    query: str
    top_k: int = 5


class FilterRequest(BaseModel):
    triples: List[str]
    query: str


class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 1000


class PipelineQueryRequest(BaseModel):
    query: str


class OntologySnapshotRequest(BaseModel):
    ttl_path: str = "ontologies/huge_ontology_v3.ttl"


class OntologyProcessBatchRequest(BaseModel):
    pages: List[Dict[str, str]] = Field(..., description="List of {title, content} dicts")


class EvaluateRequest(BaseModel):
    input_file: str
    answer_key: str
    label: str = ""


class StatusResponse(BaseModel):
    status: str
    components: Dict[str, bool]


class EmbedResponse(BaseModel):
    embeddings: List[List[float]]
    dimensions: int
    device: str


class LoadRDFResponse(BaseModel):
    triples_count: int
    message: str


class ProcessTriplesResponse(BaseModel):
    count: int
    sample: List[str]


class IngestResponse(BaseModel):
    count: int
    collection: str
    persisted_at: str


class LoadExistingResponse(BaseModel):
    collection: str
    count: int


class ChangeMetricRequest(BaseModel):
    collection_name: str = "wiki_bg_geo_chunks"
    new_metric: str = "cosine"  # cosine, l2, ip


class TripleResult(BaseModel):
    subject: str
    predicate: str
    object: str
    document: Optional[str] = None


class VectorQueryResponse(BaseModel):
    results: List[TripleResult]
    count: int


class SPARQLExpandResponse(BaseModel):
    expanded_triples: List[str]
    count: int


class FilterResponse(BaseModel):
    filtered: List[str]
    original_count: int
    filtered_count: int


class GenerateResponse(BaseModel):
    response: str


class TimingsDict(BaseModel):
    retrieval_s: float = 0.0
    expansion_s: float = 0.0
    filtering_s: float = 0.0
    generation_s: float = 0.0
    total_s: float = 0.0


class PipelineResponse(BaseModel):
    response: str
    timings: TimingsDict


class OntologySnapshotResponse(BaseModel):
    classes: int
    properties: int
    instances: int
    schema_turtle_preview: str
    sample_instances: List[str]


class ExtractionResultDTO(BaseModel):
    page_title: str
    triples_added: int
    new_classes: List[str]
    new_properties: List[str]
    new_instances: List[str]
    validation_errors: List[str]
    success: bool


class OntologyProcessBatchResponse(BaseModel):
    results: List[ExtractionResultDTO]


class EvaluateResponse(BaseModel):
    label: str
    accuracy: float
    total: int
    correct: int
    errors: List[Dict[str, str]]


class ErrorResponse(BaseModel):
    detail: str
