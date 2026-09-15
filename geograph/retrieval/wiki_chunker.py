import glob
import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


@dataclass
class WikiChunk:
    text: str
    title: str
    chunk_index: int
    source_file: str

    def to_metadata(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "chunk_index": self.chunk_index,
            "source_file": self.source_file,
        }


class WikiChunker:
    def __init__(
        self,
        data_dir: str = "data/deduplicated_data",
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        excluded_files: List[str] = None,
    ):
        self.data_dir = data_dir
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.excluded_files = set(excluded_files or [])

    def load_all_pages(self) -> List[Dict[str, str]]:
        """Load all wiki pages from JSON files, excluding specified files."""
        pattern = f"{self.data_dir}/fetched-*.json"
        files = glob.glob(pattern)
        logger.info(f"Found {len(files)} wiki data files")

        all_pages = []
        for file_path in files:
            filename = file_path.split('/')[-1]
            if filename in self.excluded_files:
                logger.info(f"Skipping excluded file: {filename}")
                continue

            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    pages = json.load(f)
                    for page in pages:
                        page['source_file'] = filename
                    all_pages.extend(pages)
            except Exception as e:
                logger.warning(f"Failed to load {file_path}: {e}")

        logger.info(f"Loaded {len(all_pages)} total pages")
        return all_pages

    def chunk_pages(self, pages: List[Dict[str, str]]) -> List[WikiChunk]:
        """Chunk pages with token-based splitting and overlap."""
        chunks = []

        for page in pages:
            title = page.get('title', 'Unknown')
            content = page.get('content', '')
            source_file = page.get('source_file', 'unknown')

            if not content.strip():
                continue

            page_chunks = self._chunk_text(content, title, source_file)
            chunks.extend(page_chunks)

        logger.info(f"Created {len(chunks)} chunks from {len(pages)} pages")
        return chunks

    def _chunk_text(self, text: str, title: str, source_file: str) -> List[WikiChunk]:
        """Split text into overlapping chunks."""
        words = text.split()
        word_chunk_size = int(self.chunk_size / 1.3)
        word_overlap = int(self.chunk_overlap / 1.3)

        chunks = []
        start = 0
        chunk_idx = 0

        while start < len(words):
            end = start + word_chunk_size
            chunk_words = words[start:end]
            chunk_text = ' '.join(chunk_words)

            if chunk_text.strip():
                chunks.append(WikiChunk(
                    text=chunk_text,
                    title=title,
                    chunk_index=chunk_idx,
                    source_file=source_file,
                ))
                chunk_idx += 1

            start += word_chunk_size - word_overlap

            if end >= len(words):
                break

        return chunks

    def load_and_chunk(self) -> List[WikiChunk]:
        """Convenience method: load all pages and chunk them."""
        pages = self.load_all_pages()
        return self.chunk_pages(pages)
