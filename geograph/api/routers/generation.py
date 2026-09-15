from fastapi import APIRouter, HTTPException

from geograph.api.dependencies import registry
from geograph.api.schemas import GenerateRequest, GenerateResponse

router = APIRouter(prefix="/api/generation", tags=["generation"])


@router.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    """Generate text using BgGPT."""
    try:
        client = registry.llm_client
    except RuntimeError as e:
        raise HTTPException(400, str(e))

    response = client.generate(req.prompt, max_tokens=req.max_tokens)
    return GenerateResponse(response=response)
