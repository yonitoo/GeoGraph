from fastapi import APIRouter, HTTPException

from geograph.constants import CHROMA_VECTOR_RAG_COLLECTION_NAME
from geograph.api.dependencies import registry
from geograph.api.schemas import (
    IngestRequest, IngestResponse,
    IngestWikiRequest,
    LoadExistingResponse,
    ChangeMetricRequest,
    QueryRequest, VectorQueryResponse, TripleResult,
)
from geograph.retrieval.wiki_chunker import WikiChunker
from geograph.retrieval.vector_store import VectorStore, create_chroma_client

router = APIRouter(prefix="/api/vector-store", tags=["vector-store"])


@router.post("/ingest", response_model=IngestResponse)
def ingest(req: IngestRequest):
    """Load RDF, process triples, embed, and ingest into persistent ChromaDB."""
    try:
        registry.load_rdf(req.ttl_path)
        semantic_strings, metadata_list = registry.process_triples()
        registry.vector_store.ingest(semantic_strings, metadata_list, registry.embedder)
    except Exception as e:
        raise HTTPException(500, f"Ingestion failed: {e}")

    cfg = registry.config
    return IngestResponse(
        count=len(semantic_strings),
        collection=cfg.retrieval.chroma_collection_name,
        persisted_at=cfg.retrieval.chroma_db_path or "(in-memory)",
    )


@router.post("/ingest-wiki", response_model=IngestResponse)
def ingest_wiki(req: IngestWikiRequest):
    """Load wiki pages, chunk them, embed, and ingest into separate ChromaDB collection."""
    try:
        chunker = WikiChunker(
            data_dir=req.data_dir,
            chunk_size=req.chunk_size,
            chunk_overlap=req.chunk_overlap,
            excluded_files=req.excluded_files,
        )
        chunks = chunker.load_and_chunk()

        texts = [chunk.text for chunk in chunks]
        metadata_list = [chunk.to_metadata() for chunk in chunks]

        cfg = registry.config
        wiki_retrieval_config = registry.config.retrieval
        wiki_retrieval_config.chroma_collection_name = CHROMA_VECTOR_RAG_COLLECTION_NAME

        client = create_chroma_client(wiki_retrieval_config)
        wiki_vs = VectorStore(client, wiki_retrieval_config)
        wiki_vs.ingest(texts, metadata_list, registry.embedder)

        return IngestResponse(
            count=len(chunks),
            collection=CHROMA_VECTOR_RAG_COLLECTION_NAME,
            persisted_at=cfg.retrieval.chroma_db_path or "(in-memory)",
        )
    except Exception as e:
        raise HTTPException(500, f"Wiki ingestion failed: {e}")


@router.post("/load-existing", response_model=LoadExistingResponse)
def load_existing():
    """Load an existing persisted ChromaDB collection."""
    try:
        registry.vector_store.load_existing()
    except Exception as e:
        raise HTTPException(400, f"Failed to load collection: {e}")

    count = registry.vector_store.collection.count() if registry.vector_store.collection else 0
    return LoadExistingResponse(
        collection=registry.config.retrieval.chroma_collection_name,
        count=count,
    )


@router.post("/change-metric", response_model=IngestResponse)
def change_metric(req: ChangeMetricRequest):
    """Change distance metric of a collection without re-embedding (fast)."""
    try:
        cfg = registry.config

        # Create vector store for target collection
        target_config = cfg.retrieval
        target_config.chroma_collection_name = req.collection_name

        client = create_chroma_client(target_config)
        vs = VectorStore(client, target_config)

        # Change metric
        vs.change_distance_metric(req.new_metric)

        return IngestResponse(
            count=vs.collection.count(),
            collection=req.collection_name,
            persisted_at=cfg.retrieval.chroma_db_path or "(in-memory)",
        )
    except Exception as e:
        raise HTTPException(500, f"Metric change failed: {e}")


@router.post("/query", response_model=VectorQueryResponse)
def query(req: QueryRequest):
    """Query the RDF triples vector store with re-ranking."""
    if not registry.vector_store_ready:
        raise HTTPException(400, "Vector store not initialized. Call /ingest or /load-existing first.")

    rdf = registry._rdf_processor  # may be None if only load-existing was called
    results = registry.vector_store.retrieve_and_rerank(
        req.query,
        registry.embedder,
        registry.config.ontology,
        rdf,
        top_k=req.top_k,
    )

    triple_results = []
    for r in results:
        triple_results.append(TripleResult(
            subject=r.get("subject_uri", "").split("#")[-1],
            predicate=r.get("predicate_uri", "").split("#")[-1],
            object=r.get("object_value", r.get("object_uri", "").split("#")[-1]),
            document=r.get("document"),
        ))

    return VectorQueryResponse(results=triple_results, count=len(triple_results))


@router.post("/query-wiki", response_model=VectorQueryResponse)
def query_wiki(req: QueryRequest):
    """Query the wiki chunks vector store."""
    from geograph.constants import CHROMA_VECTOR_RAG_COLLECTION_NAME

    try:
        cfg = registry.config
        wiki_retrieval_config = registry.config.retrieval
        wiki_retrieval_config.chroma_collection_name = CHROMA_VECTOR_RAG_COLLECTION_NAME

        client = create_chroma_client(wiki_retrieval_config)
        wiki_vs = VectorStore(client, wiki_retrieval_config)
        wiki_vs.load_existing()

        query_embedding = registry.embedder.embed(req.query)
        raw_results = wiki_vs.query_raw(query_embedding, req.top_k)

        results = []
        metadatas = raw_results.get("metadatas", [[]])[0]
        documents = raw_results.get("documents", [[]])[0]
        distances = raw_results.get("distances", [[]])[0]

        for i in range(len(documents)):
            meta = metadatas[i]
            results.append(TripleResult(
                subject=meta.get("title", ""),
                predicate=f"chunk_{meta.get('chunk_index', 0)}",
                object=f"{distances[i]:.4f}",
                document=documents[i][:200] + "..." if len(documents[i]) > 200 else documents[i],
            ))

        return VectorQueryResponse(results=results, count=len(results))
    except Exception as e:
        raise HTTPException(500, f"Wiki query failed: {e}")
