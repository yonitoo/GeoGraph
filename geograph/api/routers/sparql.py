from fastapi import APIRouter, HTTPException

from geograph.api.dependencies import registry
from geograph.api.schemas import SPARQLExpandRequest, SPARQLExpandResponse

router = APIRouter(prefix="/api/sparql", tags=["sparql"])


@router.post("/expand", response_model=SPARQLExpandResponse)
def expand(req: SPARQLExpandRequest):
    """Retrieve from vector store then expand via SPARQL n-hop traversal."""
    if not registry.vector_store_ready:
        raise HTTPException(400, "Vector store not initialized. Call /api/vector-store/ingest first.")
    if not registry.rdf_ready:
        raise HTTPException(400, "RDF not loaded. Call /api/rdf/load first.")

    try:
        expander = registry.sparql_expander
    except RuntimeError as e:
        raise HTTPException(400, str(e))

    retrieved = registry.vector_store.retrieve_and_rerank(
        req.query,
        registry.embedder,
        registry.config.ontology,
        registry.rdf_processor,
        top_k=req.top_k,
    )
    expanded = expander.expand(registry.rdf_processor, retrieved)

    return SPARQLExpandResponse(expanded_triples=expanded, count=len(expanded))
