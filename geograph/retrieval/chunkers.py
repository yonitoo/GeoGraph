import glob
import json
import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_WORDS_PER_TOKEN = 1.3
_SECTION_RE = re.compile(r"==\s*(.+?)\s*==")


def _tokens_to_words(token_count: int) -> int:
    return max(1, int(token_count / _WORDS_PER_TOKEN))


@dataclass
class Chunk:
    text: str
    title: str
    section: str
    chunk_index: int
    source_file: str
    parent_id: str = ""
    is_child: bool = False

    def to_metadata(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "section": self.section,
            "chunk_index": self.chunk_index,
            "source_file": self.source_file,
            "parent_id": self.parent_id,
            "is_child": self.is_child,
        }


class BaseChunker(ABC):
    def __init__(
        self,
        data_dir: str = "data/deduplicated_data",
        excluded_files: Optional[List[str]] = None,
    ) -> None:
        self.data_dir = data_dir
        self.excluded_files = set(excluded_files or [])

    def load_all_pages(self) -> List[Dict[str, str]]:
        pattern = f"{self.data_dir}/fetched-*.json"
        files = glob.glob(pattern)
        logger.info("Found %d wiki data files", len(files))

        pages: List[Dict[str, str]] = []
        for path in files:
            filename = path.split("/")[-1]
            if filename in self.excluded_files:
                logger.info("Skipping excluded file: %s", filename)
                continue
            try:
                with open(path, encoding="utf-8") as fh:
                    for page in json.load(fh):
                        page["source_file"] = filename
                        pages.append(page)
            except Exception as exc:
                logger.warning("Failed to load %s: %s", path, exc)

        logger.info("Loaded %d pages total", len(pages))
        return pages

    @abstractmethod
    def chunk_pages(self, pages: List[Dict[str, str]]) -> List[Chunk]:
        """Convert loaded pages into chunks according to the strategy."""

    def load_and_chunk(self) -> List[Chunk]:
        return self.chunk_pages(self.load_all_pages())


    @staticmethod
    def _window_chunks(
        text: str,
        title: str,
        section: str,
        source_file: str,
        chunk_size_tokens: int,
        overlap_tokens: int,
        start_index: int = 0,
        parent_id: str = "",
        is_child: bool = False,
        prefix: str = "",
    ) -> List[Chunk]:
        """Sliding word-window chunking. `prefix` is prepended to every chunk text."""
        words = text.split()
        w_size = _tokens_to_words(chunk_size_tokens)
        w_overlap = _tokens_to_words(overlap_tokens)
        stride = max(1, w_size - w_overlap)

        chunks: List[Chunk] = []
        idx = start_index
        pos = 0
        while pos < len(words):
            end = pos + w_size
            fragment = " ".join(words[pos:end])
            if fragment.strip():
                chunk_text = f"{prefix}{fragment}" if prefix else fragment
                chunks.append(Chunk(
                    text=chunk_text,
                    title=title,
                    section=section,
                    chunk_index=idx,
                    source_file=source_file,
                    parent_id=parent_id,
                    is_child=is_child,
                ))
                idx += 1
            if end >= len(words):
                break
            pos += stride
        return chunks



class FixedSizeChunker(BaseChunker):
    """Fixed-size overlapping word-window chunks (256 token / 32 overlap).

    Factoid-tuned replacement for the incumbent 500/50 config.
    No structural signals are used — pure token-count splitting.
    """

    def __init__(
        self,
        chunk_size: int = 256,
        chunk_overlap: int = 32,
        data_dir: str = "data/deduplicated_data",
        excluded_files: Optional[List[str]] = None,
    ) -> None:
        super().__init__(data_dir, excluded_files)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_pages(self, pages: List[Dict[str, str]]) -> List[Chunk]:
        all_chunks: List[Chunk] = []
        for page in pages:
            title = page.get("title", "Unknown")
            content = page.get("content", "")
            source_file = page.get("source_file", "unknown")
            if not content.strip():
                continue
            chunks = self._window_chunks(
                text=content,
                title=title,
                section="",
                source_file=source_file,
                chunk_size_tokens=self.chunk_size,
                overlap_tokens=self.chunk_overlap,
                start_index=0,
            )
            all_chunks.extend(chunks)

        logger.info("FixedSizeChunker: %d chunks from %d pages", len(all_chunks), len(pages))
        return all_chunks


class StructureAwareChunker(BaseChunker):
    """Splits Wikipedia content at == section headers, then sub-splits long sections.

    Each chunk text is prefixed with the article title and section name so that
    entity-grounding survives chunk boundaries (e.g. 'Мусала: Геология\\n...').
    Sections that fit within `max_chunk_size` tokens become a single chunk.
    """

    def __init__(
        self,
        max_chunk_size: int = 256,
        overlap: int = 32,
        data_dir: str = "data/deduplicated_data",
        excluded_files: Optional[List[str]] = None,
    ) -> None:
        super().__init__(data_dir, excluded_files)
        self.max_chunk_size = max_chunk_size
        self.overlap = overlap

    def chunk_pages(self, pages: List[Dict[str, str]]) -> List[Chunk]:
        all_chunks: List[Chunk] = []
        for page in pages:
            title = page.get("title", "Unknown")
            content = page.get("content", "")
            source_file = page.get("source_file", "unknown")
            if not content.strip():
                continue
            all_chunks.extend(self._chunk_article(title, content, source_file))

        logger.info("StructureAwareChunker: %d chunks from %d pages", len(all_chunks), len(pages))
        return all_chunks

    def _chunk_article(self, title: str, content: str, source_file: str) -> List[Chunk]:
        sections = self._split_sections(content)
        chunks: List[Chunk] = []
        global_idx = 0
        for section_title, section_body in sections:
            if not section_body.strip():
                continue
            prefix = f"{title}: {section_title}\n" if section_title else f"{title}\n"
            w_max = _tokens_to_words(self.max_chunk_size)
            words = section_body.split()
            if len(words) <= w_max:
                # Entire section fits in one chunk
                chunks.append(Chunk(
                    text=f"{prefix}{section_body.strip()}",
                    title=title,
                    section=section_title,
                    chunk_index=global_idx,
                    source_file=source_file,
                ))
                global_idx += 1
            else:
                sub = self._window_chunks(
                    text=section_body,
                    title=title,
                    section=section_title,
                    source_file=source_file,
                    chunk_size_tokens=self.max_chunk_size,
                    overlap_tokens=self.overlap,
                    start_index=global_idx,
                    prefix=prefix,
                )
                chunks.extend(sub)
                global_idx += len(sub)
        return chunks

    @staticmethod
    def _split_sections(content: str) -> List[Tuple[str, str]]:
        parts = _SECTION_RE.split(content)
        sections: List[Tuple[str, str]] = []
        if parts[0].strip():
            sections.append(("", parts[0].strip()))
        i = 1
        while i < len(parts) - 1:
            section_title = parts[i].strip()
            section_body = parts[i + 1].strip() if i + 1 < len(parts) else ""
            sections.append((section_title, section_body))
            i += 2
        return sections


@dataclass
class ParentStore:
    """In-memory mapping from parent_id → parent document dict.

    Persisted to / loaded from a JSONL file so re-ingestion is not required
    at query time.
    """
    _store: Dict[str, Dict[str, str]] = field(default_factory=dict)

    def add(self, parent_id: str, text: str, title: str, section: str) -> None:
        self._store[parent_id] = {"text": text, "title": title, "section": section}

    def get(self, parent_id: str) -> Optional[Dict[str, str]]:
        return self._store.get(parent_id)

    def save(self, path: str) -> None:
        import pathlib
        pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            for pid, doc in self._store.items():
                fh.write(json.dumps({"parent_id": pid, **doc}, ensure_ascii=False) + "\n")
        logger.info("Parent store saved: %d parents → %s", len(self._store), path)

    @classmethod
    def load(cls, path: str) -> "ParentStore":
        store = cls()
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                pid = rec.pop("parent_id")
                store._store[pid] = rec
        logger.info("Parent store loaded: %d parents from %s", len(store._store), path)
        return store

    def __len__(self) -> int:
        return len(self._store)


class ParentChildChunker(BaseChunker):
    """Two-level chunking: small children (128 tokens) for precise retrieval,
    full section parents (≤512 tokens) for rich generation context.

    Children carry a `parent_id` that the pipeline uses to expand to the parent
    at query time. After `chunk_pages()`, the populated `parent_store` attribute
    holds all parents and can be persisted via `parent_store.save(path)`.
    """

    def __init__(
        self,
        child_size: int = 128,
        parent_size: int = 512,
        child_overlap: int = 16,
        data_dir: str = "data/deduplicated_data",
        excluded_files: Optional[List[str]] = None,
    ) -> None:
        super().__init__(data_dir, excluded_files)
        self.child_size = child_size
        self.parent_size = parent_size
        self.child_overlap = child_overlap
        self.parent_store = ParentStore()

    def chunk_pages(self, pages: List[Dict[str, str]]) -> List[Chunk]:
        self.parent_store = ParentStore()
        all_children: List[Chunk] = []
        for page in pages:
            title = page.get("title", "Unknown")
            content = page.get("content", "")
            source_file = page.get("source_file", "unknown")
            if not content.strip():
                continue
            children = self._chunk_article(title, content, source_file)
            all_children.extend(children)

        logger.info(
            "ParentChildChunker: %d children, %d parents from %d pages",
            len(all_children), len(self.parent_store), len(pages),
        )
        return all_children

    def _chunk_article(self, title: str, content: str, source_file: str) -> List[Chunk]:
        sections = StructureAwareChunker._split_sections(content)
        all_children: List[Chunk] = []
        child_idx = 0

        for s_idx, (section_title, section_body) in enumerate(sections):
            if not section_body.strip():
                continue

            prefix = f"{title}: {section_title}\n" if section_title else f"{title}\n"
            parent_words = section_body.split()
            w_parent = _tokens_to_words(self.parent_size)

            # Split section into parent-sized windows
            parent_pos = 0
            local_parent_idx = 0
            while parent_pos < len(parent_words):
                parent_end = parent_pos + w_parent
                parent_fragment = " ".join(parent_words[parent_pos:parent_end])

                parent_id = f"{source_file}::{title}::s{s_idx}::p{local_parent_idx}"
                parent_text = f"{prefix}{parent_fragment}"
                self.parent_store.add(parent_id, parent_text, title, section_title)

                # Produce children within this parent
                children = self._window_chunks(
                    text=parent_fragment,
                    title=title,
                    section=section_title,
                    source_file=source_file,
                    chunk_size_tokens=self.child_size,
                    overlap_tokens=self.child_overlap,
                    start_index=child_idx,
                    parent_id=parent_id,
                    is_child=True,
                    prefix=prefix,
                )
                all_children.extend(children)
                child_idx += len(children)

                if parent_end >= len(parent_words):
                    break
                parent_pos += w_parent  # parents do not overlap
                local_parent_idx += 1

        return all_children
