from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

from rdflib import Graph, RDFS

from geograph.constants import UNWANTED_SECTIONS
from geograph.data_prep.fetch_recursively_data_wikipedia import fetch_article_content, search_article_title
from geograph.ontology.ontology_builder import OntologyBuilder
from geograph.data_prep.process_wiki_pages import clean_wiki_content

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEDUP_DATA_DIR = REPO_ROOT / "data/deduplicated_data"
TARGETS_PATH = REPO_ROOT / "testset/kg_rag/curation_targets.json"
BASE_TTL_PATH = REPO_ROOT / "ontologies/huge_ontology.ttl"
OUTPUT_TTL_PATH = REPO_ROOT / "ontologies/huge_ontology_v2.ttl"

MAX_PAGE_CHARS = 6000
DEFAULT_MODEL = "gpt-4o"
DEFAULT_MAX_TOKENS = 1500


class WikiPageStore:
    """Lazily loads and caches deduplicated_data JSON files by filename."""

    def __init__(self, dedup_dir: Path):
        self._dedup_dir = dedup_dir
        self._files: Dict[str, Dict[str, str]] = {}

    def get(self, filename: str, title: str) -> Optional[str]:
        pages_by_title = self._files.get(filename)
        if pages_by_title is None:
            with open(self._dedup_dir / filename, encoding="utf-8") as f:
                pages = json.load(f)
            pages_by_title = {p["title"]: p.get("content", "") for p in pages}
            self._files[filename] = pages_by_title
        return pages_by_title.get(title)


def _clean(content: str) -> str:
    cleaned = clean_wiki_content(content, UNWANTED_SECTIONS)
    return cleaned[:MAX_PAGE_CHARS]


def collect_pages(targets: List[dict], store: WikiPageStore, fetch_missing: bool) -> List[Dict[str, str]]:
    """Resolve curation targets to deduplicated {title, content} pages."""
    pages_by_title: Dict[str, str] = {}

    for target in targets:
        if not target.get("fetchable", True):
            continue

        source_pages = target.get("source_pages") or []
        if source_pages:
            first = source_pages[0]
            if first["title"] in pages_by_title:
                continue
            content = store.get(first["file"], first["title"])
            if content:
                pages_by_title[first["title"]] = content
            continue

        if not fetch_missing:
            continue
        term = target["term"]
        lookup_title = term[:1].upper() + term[1:]
        if lookup_title in pages_by_title:
            continue
        try:
            content = fetch_article_content(lookup_title, lang="bg")
            if not content:
                matched_title = search_article_title(term, lang="bg")
                if matched_title:
                    lookup_title = matched_title
                    content = fetch_article_content(matched_title, lang="bg")
        except Exception as e:
            logger.warning("Fetch failed for %r: %s", term, e)
            continue
        if content:
            pages_by_title[lookup_title] = content
        else:
            logger.info("No Wikipedia page found for %r", term)

    return [{"title": title, "content": _clean(content)} for title, content in pages_by_title.items() if content.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None, help="Process at most N pages (for smoke-testing).")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--skip-fetch", action="store_true", help="Track A only; skip live Wikipedia fetches (Track B).")
    parser.add_argument("--output", default=str(OUTPUT_TTL_PATH))
    parser.add_argument("--targets", default=str(TARGETS_PATH), help="Curation targets JSON.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")

    with open(args.targets, encoding="utf-8") as f:
        data = json.load(f)
    targets = data["targets"]
    logger.info("Loaded %d curation targets from %s", len(targets), args.targets)

    store = WikiPageStore(DEDUP_DATA_DIR)
    pages = collect_pages(targets, store, fetch_missing=not args.skip_fetch)
    logger.info("Resolved %d unique source pages to extract from.", len(pages))

    resuming = os.path.exists(args.output)
    start_from = args.output if resuming else str(BASE_TTL_PATH)
    if resuming:
        already_done = {
            str(label).strip().lower() for _, label in Graph().parse(args.output, format="turtle").subject_objects(RDFS.label)
        }
        before = len(pages)
        pages = [p for p in pages if p["title"].strip().lower() not in already_done]
        logger.info("Resuming from %s: skipping %d already-extracted pages (%d remaining).",
                    args.output, before - len(pages), len(pages))

    if args.limit is not None:
        pages = pages[: args.limit]
        logger.info("Limiting to first %d pages for this run.", len(pages))

    if not pages:
        logger.warning("No pages to process. Exiting.")
        return

    from geograph.config import PipelineConfig
    config = PipelineConfig.from_constants()

    builder = OntologyBuilder(
        ontology_path=start_from,
        api_key=config.reasoning.api_key,
        model=args.model,
        max_tokens=args.max_tokens,
    )
    triples_before = len(builder.graph)

    results = builder.process_batch(pages, backup=False)

    saved_path = builder.save(output_path=args.output)

    succeeded = [r for r in results if r.success]
    failed = [r for r in results if not r.success]
    logger.info(
        "Curation run complete: %d/%d pages succeeded, %d triples added (%d -> %d), saved to %s",
        len(succeeded), len(results), len(builder.graph) - triples_before,
        triples_before, len(builder.graph), saved_path,
    )
    if failed:
        logger.warning("Failed pages:")
        for r in failed:
            logger.warning("  %s: %s", r.page_title, r.validation_errors)


if __name__ == "__main__":
    main()
