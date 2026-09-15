import logging
import time
from dataclasses import dataclass
from typing import List, Tuple

from geograph.config import PipelineConfig
from geograph.retrieval.embedder import Embedder
from geograph.llm.llm_client import BgGPTClient
from geograph.llm.prompt_builder import build_grounded_prompt
from geograph.retrieval.vector_store import VectorStore, create_chroma_client
from geograph.retrieval.wiki_chunker import WikiChunker

logger = logging.getLogger(__name__)


@dataclass
class VectorQueryTimings:
    retrieval_s: float = 0.0
    generation_s: float = 0.0

    @property
    def total_s(self) -> float:
        return self.retrieval_s + self.generation_s

    def summary(self) -> str:
        return (
            f"retrieval={self.retrieval_s:.2f}s, generation={self.generation_s:.2f}s, "
            f"total={self.total_s:.2f}s"
        )


class VectorRAGPipeline:
    """Vector-only Retrieval-Augmented Generation using Wikipedia chunks.

    Flow: Vector Search (embedding similarity) → Grounded Generation
    """

    def __init__(
        self,
        config: PipelineConfig,
        wiki_data_dir: str = "data/deduplicated_data",
        chunk_size: int = 500,
        chunk_overlap: int = 50,
    ):
        self.config = config
        self.wiki_data_dir = wiki_data_dir
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.embedder = Embedder(config.embedding)
        self.vector_store: VectorStore = None
        self.llm_client = BgGPTClient(config.llm)

    def ingest(self) -> None:
        """Load wiki pages, chunk them, embed, and store in vector DB."""
        from geograph.constants import CHROMA_VECTOR_RAG_COLLECTION_NAME

        # Load and chunk wiki pages
        chunker = WikiChunker(
            data_dir=self.wiki_data_dir,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        chunks = chunker.load_and_chunk()

        # Extract texts and metadata
        texts = [chunk.text for chunk in chunks]
        metadata_list = [chunk.to_metadata() for chunk in chunks]

        # Create vector store with wiki-specific collection
        wiki_config = self.config.retrieval
        wiki_config.chroma_collection_name = CHROMA_VECTOR_RAG_COLLECTION_NAME

        client = create_chroma_client(wiki_config)
        self.vector_store = VectorStore(client, wiki_config)
        self.vector_store.ingest(texts, metadata_list, self.embedder)
        logger.info("Vector RAG wiki chunk ingestion complete: %d chunks.", len(chunks))

    def load_existing(self) -> None:
        """Load existing wiki chunks vector store."""
        from geograph.constants import CHROMA_VECTOR_RAG_COLLECTION_NAME

        wiki_config = self.config.retrieval
        wiki_config.chroma_collection_name = CHROMA_VECTOR_RAG_COLLECTION_NAME

        client = create_chroma_client(wiki_config)
        self.vector_store = VectorStore(client, wiki_config)
        self.vector_store.load_existing()
        logger.info("Loaded existing wiki chunks vector store.")

    def query(self, user_query: str, top_k: int = 10) -> Tuple[str, VectorQueryTimings]:
        """Execute the Vector RAG pipeline for a user query."""
        if self.vector_store is None:
            raise RuntimeError("Pipeline not initialized. Call ingest() or load_existing() first.")

        timings = VectorQueryTimings()

        # vec similarity search
        t0 = time.time()
        query_embedding = self.embedder.embed(user_query)
        raw_results = self.vector_store.query_raw(query_embedding, top_k)
        timings.retrieval_s = time.time() - t0

        documents = raw_results.get("documents", [[]])[0]
        logger.info("Retrieval: %d chunks in %.2fs", len(documents), timings.retrieval_s)

        if not documents:
            logger.warning("No documents retrieved.")
            return "Няма информация за този въпрос.", timings

        # build context from retrieved chunks
        context_chunks = documents

        # prompt construction and gen
        prompt = build_grounded_prompt(context_chunks, user_query)
        t0 = time.time()
        response = self.llm_client.generate(prompt)
        timings.generation_s = time.time() - t0
        logger.info("Generation in %.2fs", timings.generation_s)

        return response, timings
