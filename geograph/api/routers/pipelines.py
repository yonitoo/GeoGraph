import time

from fastapi import APIRouter, HTTPException

from geograph.api.dependencies import registry
from geograph.api.schemas import PipelineQueryRequest, PipelineResponse, TimingsDict
from geograph.retrieval.embedding_cache import cosine_scores
from geograph.llm.prompt_builder import build_grounded_prompt

router = APIRouter(prefix="/api/pipelines", tags=["pipelines"])


@router.post("/kg-rag", response_model=PipelineResponse)
def kg_rag_query(req: PipelineQueryRequest):
    """Execute the full KG-RAG pipeline: retrieve -> expand -> filter -> generate."""
    if not registry.vector_store_ready:
        raise HTTPException(400, "Vector store not initialized. Call /api/vector-store/ingest first.")
    if not registry.rdf_ready:
        raise HTTPException(400, "RDF not loaded.")

    cfg = registry.config

    # retrieval
    t0 = time.time()
    retrieved = registry.vector_store.retrieve_and_rerank(
        req.query, registry.embedder, cfg.ontology, registry.rdf_processor,
    )
    retrieval_s = time.time() - t0

    # SPARQL expansion
    try:
        expander = registry.sparql_expander
    except RuntimeError as e:
        raise HTTPException(400, f"SPARQL not available: {e}")

    def relevance_scorer(query: str, candidates: list) -> list:
        return cosine_scores(registry.embedder, query, candidates)

    t0 = time.time()
    expansion = expander.expand(registry.rdf_processor, retrieved, req.query, relevance_scorer)
    expanded = expansion.semantic_strings
    expansion_s = time.time() - t0

    # Reasoning filter
    try:
        rf = registry.reasoning_filter
    except RuntimeError as e:
        raise HTTPException(400, f"Reasoning filter not available: {e}")

    t0 = time.time()
    filter_result = rf.filter(expanded, req.query)
    filtering_s = time.time() - t0

    # gen
    try:
        llm = registry.llm_client
    except RuntimeError as e:
        raise HTTPException(400, f"LLM not available: {e}")

    prompt = build_grounded_prompt(filter_result.triples, req.query)
    t0 = time.time()
    response = llm.generate(prompt)
    generation_s = time.time() - t0

    return PipelineResponse(
        response=response,
        timings=TimingsDict(
            retrieval_s=round(retrieval_s, 3),
            expansion_s=round(expansion_s, 3),
            filtering_s=round(filtering_s, 3),
            generation_s=round(generation_s, 3),
            total_s=round(retrieval_s + expansion_s + filtering_s + generation_s, 3),
        ),
    )


@router.post("/vector-rag", response_model=PipelineResponse)
def vector_rag_query(req: PipelineQueryRequest):
    """Execute Vector RAG pipeline using wiki chunks: retrieve -> generate (no SPARQL, no reasoning)."""
    from geograph.pipelines.vector_rag_pipeline import VectorRAGPipeline

    cfg = registry.config

    try:
        pipeline = VectorRAGPipeline(cfg)
        pipeline.load_existing()
    except Exception as e:
        raise HTTPException(400, f"Failed to load wiki chunks: {e}. Did you run /api/vector-store/ingest-wiki?")

    try:
        llm = registry.llm_client
    except RuntimeError as e:
        raise HTTPException(400, f"LLM not available: {e}")

    try:
        response, timings = pipeline.query(req.query, top_k=10)
    except Exception as e:
        raise HTTPException(500, f"Vector RAG query failed: {e}")

    return PipelineResponse(
        response=response,
        timings=TimingsDict(
            retrieval_s=round(timings.retrieval_s, 3),
            generation_s=round(timings.generation_s, 3),
            total_s=round(timings.total_s, 3),
        ),
    )
