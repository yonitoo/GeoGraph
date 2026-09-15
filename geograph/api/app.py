import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from geograph.api.dependencies import registry
from geograph.api.schemas import StatusResponse
from geograph.config import PipelineConfig

from geograph.api.routers import (
    embeddings,
    rdf,
    vector_store,
    sparql,
    reasoning,
    generation,
    pipelines,
    ontology,
    evaluation,
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    )
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("chromadb").setLevel(logging.WARNING)

    logger.info("Loading PipelineConfig from constants...")
    registry.config = PipelineConfig.from_constants()
    logger.info("Config loaded. API ready at /docs")
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="KG-RAG API",
    description="Component testing backend for Knowledge Graph-based Retrieval-Augmented Generation",
    version="0.1.0",
    lifespan=lifespan,
)

# Mount all routers
app.include_router(embeddings.router)
app.include_router(rdf.router)
app.include_router(vector_store.router)
app.include_router(sparql.router)
app.include_router(reasoning.router)
app.include_router(generation.router)
app.include_router(pipelines.router)
app.include_router(ontology.router)
app.include_router(evaluation.router)


@app.get("/api/health", response_model=StatusResponse, tags=["health"])
def health():
    """Check which components are initialized."""
    return StatusResponse(
        status="ok",
        components=registry.status(),
    )
