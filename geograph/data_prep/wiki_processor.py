import json
import logging
import os
import re
from collections import Counter
from typing import Dict, List

from jsonlines import jsonlines

logger = logging.getLogger(__name__)


def clean_wiki_content(content: str, unwanted_sections: List[str]) -> str:
    cleaned = content

    section_pattern = "|".join(re.escape(sec) for sec in unwanted_sections)
    regex = re.compile(
        rf"^\s*==\s*({section_pattern})\s*==.*",
        re.MULTILINE | re.IGNORECASE | re.DOTALL,
    )
    match = regex.search(cleaned)
    if match:
        cleaned = cleaned[:match.start()]

    cleaned = re.sub(r"<ref.*?>.*?</ref>", "", cleaned, flags=re.IGNORECASE | re.DOTALL)
    cleaned = re.sub(r"{{.*?}}", "", cleaned, flags=re.DOTALL)
    cleaned = re.sub(r"\[\[Файл:.*?\]\]", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\[\[Категория:.*?\]\]", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"'''(.*?)'''", r"\1", cleaned)
    cleaned = re.sub(r"''(.*?)''", r"\1", cleaned)
    cleaned = re.sub(r"==+\s*(.*?)\s*==+", r"\n\n\1\n", cleaned)
    cleaned = re.sub(r"\[\[(?:[^|\]]*\|)?([^\]]+)\]\]", r"\1", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    cleaned = re.sub(r"[ \t]+", " ", cleaned)

    return cleaned.strip()


def load_pages_from_file(json_file_path: str) -> List[Dict[str, str]]:
    if not os.path.exists(json_file_path):
        logger.error("File not found: %s", json_file_path)
        return []

    try:
        with open(json_file_path, "r", encoding="utf-8") as f:
            pages = json.load(f)
        if not isinstance(pages, list):
            logger.error("Expected a JSON list in %s", json_file_path)
            return []

        logger.info("Loaded %d pages from %s", len(pages), json_file_path)

        titles = [p.get("title", "") for p in pages]
        if len(titles) != len(set(titles)):
            dupes = {t: c for t, c in Counter(titles).items() if c > 1}
            logger.warning("Duplicate titles found: %s", dupes)

        return pages
    except json.JSONDecodeError as e:
        logger.error("JSON decode error in %s: %s", json_file_path, e)
        return []


def load_pages_from_folder(folder_path: str) -> List[Dict[str, str]]:
    if not os.path.isdir(folder_path):
        logger.error("Not a valid directory: %s", folder_path)
        return []

    all_pages: List[Dict[str, str]] = []

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if filename.endswith(".json"):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, list):
                    all_pages.extend(data)
                    logger.info("Loaded %d pages from %s", len(data), filename)
            except Exception as e:
                logger.error("Error loading %s: %s", filename, e)
        elif filename.endswith(".jsonl"):
            try:
                with jsonlines.open(file_path) as reader:
                    data = list(reader)
                all_pages.extend(data)
                logger.info("Loaded %d pages from %s", len(data), filename)
            except Exception as e:
                logger.error("Error loading %s: %s", filename, e)

    logger.info("Total pages loaded (with potential duplicates): %d", len(all_pages))

    seen = set()
    unique_pages = []
    duplicate_titles = []

    for page in all_pages:
        title = page.get("title")
        if title and title not in seen:
            seen.add(title)
            unique_pages.append(page)
        elif title:
            duplicate_titles.append(title)

    if duplicate_titles:
        dupe_counts = Counter(duplicate_titles)
        for title, count in dupe_counts.items():
            logger.info("Removed duplicate: '%s' (%d extra copies)", title, count)

    logger.info("Total unique pages: %d", len(unique_pages))
    return unique_pages


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    pages = load_pages_from_folder("./data/deduplicated_data")
    print(f"Loaded {len(pages)} unique pages.")
