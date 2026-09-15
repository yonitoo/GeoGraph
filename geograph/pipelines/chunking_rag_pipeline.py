import logging
import pathlib
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Type

import chromadb
import torch
from tqdm import tqdm

from geograph.retrieval.chunkers import (
    BaseChunker,
    Chunk,
    FixedSizeChunker,
    ParentChildChunker,
    ParentStore,
    StructureAwareChunker,
)
from geograph.config import PipelineConfig, RetrievalConfig
from geograph.retrieval.embedder import Embedder
from geograph.llm.llm_client import BgGPTChatClient
from geograph.llm.prompt_builder import build_grounded_prompt, build_mcq_prompt
from geograph.retrieval.reranker import CrossEncoderReranker
from geograph.retrieval.vector_store import VectorStore, create_chroma_client

logger = logging.getLogger(__name__)

_WORDS_PER_TOKEN = 1.3

@dataclass
class StrategyConfig:
    collection_name: str
    chunker_class: Type[BaseChunker]
    chunker_kwargs: Dict[str, Any]
    use_parent_child: bool = False
    parent_store_path: Optional[str] = None


def _build_registry() -> Dict[str, StrategyConfig]:
    from geograph.constants import (
        CHROMA_FIXED256_COLLECTION,
        CHROMA_PARENTCHILD_COLLECTION,
        CHROMA_STRUCTURE_COLLECTION,
        CHROMA_VECTOR_RAG_COLLECTION_NAME,
        PARENT_STORE_PATH,
    )
    return {
        "fixed256": StrategyConfig(
            collection_name=CHROMA_FIXED256_COLLECTION,
            chunker_class=FixedSizeChunker,
            chunker_kwargs={"chunk_size": 256, "chunk_overlap": 32},
        ),
        "structure": StrategyConfig(
            collection_name=CHROMA_STRUCTURE_COLLECTION,
            chunker_class=StructureAwareChunker,
            chunker_kwargs={"max_chunk_size": 256, "overlap": 32},
        ),
        "parentchild": StrategyConfig(
            collection_name=CHROMA_PARENTCHILD_COLLECTION,
            chunker_class=ParentChildChunker,
            chunker_kwargs={"child_size": 128, "parent_size": 512, "child_overlap": 16},
            use_parent_child=True,
            parent_store_path=PARENT_STORE_PATH,
        ),
        # Reference point — reuses the existing collection ingested by VectorRAGPipeline
        "fixed500": StrategyConfig(
            collection_name=CHROMA_VECTOR_RAG_COLLECTION_NAME,
            chunker_class=FixedSizeChunker,
            chunker_kwargs={"chunk_size": 500, "chunk_overlap": 50},
        ),
    }


STRATEGY_REGISTRY: Dict[str, StrategyConfig] = {}  # populated lazily on first access


def get_strategy(name: str) -> StrategyConfig:
    if not STRATEGY_REGISTRY:
        STRATEGY_REGISTRY.update(_build_registry())  # mutate in place so imported refs see changes
    if name not in STRATEGY_REGISTRY:
        raise ValueError(
            f"Unknown strategy '{name}'. Available: {list(STRATEGY_REGISTRY.keys())}"
        )
    return STRATEGY_REGISTRY[name]


@dataclass
class ChunkingTimings:
    embedding_s: float = 0.0
    retrieval_s: float = 0.0
    rerank_s: float = 0.0
    generation_s: float = 0.0

    @property
    def total_s(self) -> float:
        return self.embedding_s + self.retrieval_s + self.rerank_s + self.generation_s

    def summary(self) -> str:
        return (
            f"embed={self.embedding_s:.2f}s "
            f"retrieve={self.retrieval_s:.2f}s "
            f"rerank={self.rerank_s:.2f}s "
            f"generate={self.generation_s:.2f}s "
            f"total={self.total_s:.2f}s"
        )


class ChunkingRAGPipeline:
    """Vector RAG pipeline parameterised by chunking strategy.

    The strategy controls which chunker is used, which Chroma collection is
    targeted, and whether parent expansion is applied before reranking.
    All other components (embedder, reranker, LLM) are held constant across
    strategies so that chunking is the only experimental variable.
    """

    def __init__(
        self,
        config: PipelineConfig,
        strategy_name: str = "structure",
        data_dir: str = "data/deduplicated_data",
        use_reranker: bool = True,
        embed_batch_size: int = 32,
    ) -> None:
        self.config = config
        self.strategy_name = strategy_name
        self.strategy = get_strategy(strategy_name)
        self.data_dir = data_dir
        self.use_reranker = use_reranker
        self.embed_batch_size = embed_batch_size

        self.embedder = Embedder(config.embedding)
        self.llm_client = BgGPTChatClient(config.llm)

        # Lazy-load reranker (avoids the ~2.3 GB download on startup)
        self._reranker: Optional[CrossEncoderReranker] = None

        self._vector_store: Optional[VectorStore] = None
        self._parent_store: Optional[ParentStore] = None


    def _retrieval_config(self) -> RetrievalConfig:
        cfg = self.config.retrieval
        cfg.chroma_collection_name = self.strategy.collection_name
        return cfg

    def _chroma_client(self) -> chromadb.ClientAPI:
        return create_chroma_client(self._retrieval_config())

    def ingest(self) -> None:
        """Chunk all Wikipedia pages, embed them, and write to Chroma."""
        chunker_kwargs = {**self.strategy.chunker_kwargs, "data_dir": self.data_dir}
        chunker: BaseChunker = self.strategy.chunker_class(**chunker_kwargs)

        logger.info("Chunking with strategy '%s'...", self.strategy_name)
        pages = chunker.load_all_pages()
        chunks: List[Chunk] = chunker.chunk_pages(pages)
        logger.info("Produced %d chunks.", len(chunks))

        if self.strategy.use_parent_child and isinstance(chunker, ParentChildChunker):
            self._parent_store = chunker.parent_store
            assert self.strategy.parent_store_path, "parent_store_path must be set"
            self._parent_store.save(self.strategy.parent_store_path)

        texts = [c.text for c in chunks]
        metadata_list = [c.to_metadata() for c in chunks]

        logger.info("Embedding %d chunks in batches of %d...", len(texts), self.embed_batch_size)
        all_embeddings: List[List[float]] = []
        is_mps = self.embedder.device.type == "mps"
        for start in tqdm(range(0, len(texts), self.embed_batch_size), desc="Embedding"):
            batch = texts[start : start + self.embed_batch_size]
            all_embeddings.extend(self.embedder.embed_batch(batch, batch_size=self.embed_batch_size))
            if is_mps:
                torch.mps.empty_cache()

        client = self._chroma_client()
        collection_name = self.strategy.collection_name
        try:
            client.delete_collection(collection_name)
        except Exception:
            pass
        collection = client.create_collection(
            name=collection_name, metadata={"hnsw:space": "cosine"}
        )

        chroma_batch = self.config.retrieval.chroma_batch_size
        ids = [str(i) for i in range(len(texts))]
        for i in tqdm(range(0, len(ids), chroma_batch), desc="Storing in Chroma"):
            end = i + chroma_batch
            collection.add(
                ids=ids[i:end],
                documents=texts[i:end],
                embeddings=all_embeddings[i:end],
                metadatas=metadata_list[i:end],
            )

        self._vector_store = VectorStore(client, self._retrieval_config())
        self._vector_store.collection = collection
        logger.info("Ingestion complete: %d items in '%s'.", len(ids), collection_name)

    def load_existing(self) -> None:
        """Load an already-ingested Chroma collection (and parent store for parent-child)."""
        client = self._chroma_client()
        self._vector_store = VectorStore(client, self._retrieval_config())
        self._vector_store.load_existing()

        if self.strategy.use_parent_child:
            path = self.strategy.parent_store_path
            if not path or not pathlib.Path(path).exists():
                raise FileNotFoundError(
                    f"Parent store not found at '{path}'. Run --ingest first."
                )
            self._parent_store = ParentStore.load(path)


    def query(
        self,
        question: str,
        options: List[Dict[str, str]],
        top_n: Optional[int] = None,
        top_k: Optional[int] = None,
        budget_tokens: Optional[int] = None,
    ) -> Tuple[str, List[str], ChunkingTimings]:
        """Run the full chunking-RAG query pipeline."""
        from geograph.constants import (
            CONTEXT_TOKEN_BUDGET,
            TOP_K_RERANKED,
            TOP_N_CANDIDATES,
        )

        if self._vector_store is None:
            raise RuntimeError("Pipeline not initialised. Call ingest() or load_existing() first.")

        top_n = top_n or TOP_N_CANDIDATES
        top_k = top_k or TOP_K_RERANKED
        budget_tokens = budget_tokens or CONTEXT_TOKEN_BUDGET

        timings = ChunkingTimings()
        mcq_query = build_mcq_prompt(question, options)

        # Embed Q
        t0 = time.time()
        query_embedding = self.embedder.embed(mcq_query)
        timings.embedding_s = time.time() - t0

        # retrieval
        t0 = time.time()
        raw = self._vector_store.query_raw(query_embedding, top_n)
        timings.retrieval_s = time.time() - t0

        candidate_docs: List[str] = raw.get("documents", [[]])[0]
        candidate_metas: List[Dict] = raw.get("metadatas", [[]])[0]

        if not candidate_docs:
            logger.warning("No candidates retrieved.")
            return "-", [], timings

        # Parent expansion (parent-child strategy only)
        if self.strategy.use_parent_child and self._parent_store:
            expanded: List[str] = []
            seen_parents: set = set()
            for doc, meta in zip(candidate_docs, candidate_metas):
                pid = meta.get("parent_id", "")
                if pid and pid not in seen_parents:
                    parent = self._parent_store.get(pid)
                    expanded.append(parent["text"] if parent else doc)
                    seen_parents.add(pid)
                elif not pid:
                    expanded.append(doc)
            candidate_docs = expanded

        # rerank
        t0 = time.time()
        if self.use_reranker:
            ranked_indices = self._get_reranker().rerank(mcq_query, candidate_docs, top_k)
        else:
            ranked_indices = list(range(min(top_k, len(candidate_docs))))
        timings.rerank_s = time.time() - t0

        context_chunks: List[str] = []
        used_words = 0
        budget_words = int(budget_tokens / _WORDS_PER_TOKEN)
        seen_texts: set = set()
        for idx in ranked_indices:
            text = candidate_docs[idx]
            if text in seen_texts:
                continue
            seen_texts.add(text)
            word_count = len(text.split())
            if used_words + word_count > budget_words:
                break
            context_chunks.append(text)
            used_words += word_count

        if not context_chunks:
            return "-", [], timings

        # gen
        prompt = build_grounded_prompt(context_chunks, mcq_query)
        t0 = time.time()
        response = self.llm_client.generate(prompt)
        timings.generation_s = time.time() - t0

        logger.debug(
            "Query done | strategy=%s | retrieved=%d | ctx_chunks=%d | %s",
            self.strategy_name, len(raw.get("documents", [[]])[0]),
            len(context_chunks), timings.summary(),
        )
        return response, context_chunks, timings


    def _get_reranker(self) -> CrossEncoderReranker:
        if self._reranker is None:
            from geograph.constants import RERANKER_MODEL
            self._reranker = CrossEncoderReranker(model_name=RERANKER_MODEL)
        return self._reranker
