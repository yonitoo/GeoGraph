from fastapi import APIRouter, HTTPException

from geograph.api.dependencies import registry
from geograph.api.schemas import FilterRequest, FilterResponse

router = APIRouter(prefix="/api/reasoning", tags=["reasoning"])


@router.post("/filter", response_model=FilterResponse)
def filter_triples(req: FilterRequest):
    """Filter semantic triples for relevance using the reasoning model."""
    try:
        rf = registry.reasoning_filter
    except RuntimeError as e:
        raise HTTPException(400, str(e))

    result = rf.filter(req.triples, req.query)

    return FilterResponse(
        filtered=result.triples,
        original_count=len(req.triples),
        filtered_count=len(result.triples),
    )
