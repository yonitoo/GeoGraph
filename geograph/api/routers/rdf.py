from fastapi import APIRouter, HTTPException

from geograph.api.dependencies import registry
from geograph.api.schemas import (
    LoadRDFRequest, LoadRDFResponse,
    ProcessTriplesResponse,
)

router = APIRouter(prefix="/api/rdf", tags=["rdf"])


@router.post("/load", response_model=LoadRDFResponse)
def load_rdf(req: LoadRDFRequest):
    try:
        rdf = registry.load_rdf(req.ttl_path)
    except Exception as e:
        raise HTTPException(400, f"Failed to load RDF: {e}")

    return LoadRDFResponse(
        triples_count=len(rdf.graph),
        message=f"Loaded {req.ttl_path}",
    )


@router.post("/process-triples", response_model=ProcessTriplesResponse)
def process_triples():
    if not registry.rdf_ready:
        raise HTTPException(400, "RDF not loaded. Call POST /api/rdf/load first.")

    semantic_strings, _ = registry.process_triples()

    return ProcessTriplesResponse(
        count=len(semantic_strings),
        sample=semantic_strings[:10],
    )
