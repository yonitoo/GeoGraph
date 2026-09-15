from fastapi import APIRouter, HTTPException

from geograph.api.dependencies import registry
from geograph.api.schemas import EmbedRequest, EmbedResponse

router = APIRouter(prefix="/api/embeddings", tags=["embeddings"])


@router.post("/embed", response_model=EmbedResponse)
def embed(req: EmbedRequest):
    if not req.text and not req.texts:
        raise HTTPException(400, "Provide 'text' or 'texts'.")

    embedder = registry.embedder
    texts = req.texts if req.texts else [req.text]
    embeddings = embedder.embed_batch(texts)

    return EmbedResponse(
        embeddings=embeddings,
        dimensions=len(embeddings[0]) if embeddings else 0,
        device=str(embedder.device),
    )
