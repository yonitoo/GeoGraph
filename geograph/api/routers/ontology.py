from fastapi import APIRouter, HTTPException

from geograph.api.dependencies import registry
from geograph.api.schemas import (
    OntologySnapshotRequest, OntologySnapshotResponse,
    OntologyProcessBatchRequest, OntologyProcessBatchResponse, ExtractionResultDTO,
)
from geograph.ontology.ontology_builder import OntologyBuilder

router = APIRouter(prefix="/api/ontology", tags=["ontology"])


@router.post("/snapshot", response_model=OntologySnapshotResponse)
def snapshot(req: OntologySnapshotRequest):
    """Get a snapshot of the current ontology schema and instance counts."""
    cfg = registry._require_config()
    try:
        builder = OntologyBuilder(
            ontology_path=req.ttl_path,
            api_key=cfg.reasoning.api_key or "not-set",
            namespace=str(cfg.ontology.namespace),
        )
        snap = builder.snapshot()
    except Exception as e:
        raise HTTPException(500, f"Snapshot failed: {e}")

    return OntologySnapshotResponse(
        classes=snap.class_count,
        properties=snap.property_count,
        instances=snap.instance_count,
        schema_turtle_preview=snap.schema_turtle[:2000],
        sample_instances=snap.instance_names[:20],
    )


@router.post("/process-batch", response_model=OntologyProcessBatchResponse)
def process_batch(req: OntologyProcessBatchRequest):
    """Process a batch of wiki pages to extract and merge ontology triples."""
    cfg = registry._require_config()
    if not cfg.reasoning.api_key:
        raise HTTPException(400, "OpenAI API key not set. Needed for LLM-based extraction.")

    try:
        builder = OntologyBuilder(
            ontology_path=cfg.ttl_path,
            api_key=cfg.reasoning.api_key,
            namespace=str(cfg.ontology.namespace),
        )
        results = builder.process_batch(req.pages)
    except Exception as e:
        raise HTTPException(500, f"Batch processing failed: {e}")

    return OntologyProcessBatchResponse(
        results=[
            ExtractionResultDTO(
                page_title=r.page_title,
                triples_added=r.triples_added,
                new_classes=r.new_classes,
                new_properties=r.new_properties,
                new_instances=r.new_instances,
                validation_errors=r.validation_errors,
                success=r.success,
            )
            for r in results
        ]
    )
