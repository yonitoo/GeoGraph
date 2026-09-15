import glob
import json
import logging
import os
import re
from typing import Dict, List

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

UNWANTED_SECTIONS = [
    "== Източници ==",
    "== Вижте също ==",
    "== Галерия ==",
    "== Външни препратки ==",
    "== Литература ==",
    "== Бележки ==",
    "== Източници и бележки ==",
]


def clean_page_content(content: str) -> str:
    """Remove unwanted sections from wiki page content."""
    if not content:
        return content

    earliest_pos = len(content)
    matched_section = None

    for section in UNWANTED_SECTIONS:
        pos = content.find(section)
        if pos != -1 and pos < earliest_pos:
            earliest_pos = pos
            matched_section = section

    if earliest_pos < len(content):
        cleaned = content[:earliest_pos].strip()
        logger.debug(f"Truncated at '{matched_section}' (pos {earliest_pos}, removed {len(content) - earliest_pos} chars)")
        return cleaned

    return content


def clean_wiki_files(
    input_dir: str = "../../data/deduplicated_data",
    output_dir: str = "../../data/cleaned_data",
) -> Dict[str, int]:
    """Clean all wiki JSON files and save to output directory."""
    os.makedirs(output_dir, exist_ok=True)

    pattern = f"{input_dir}/fetched-*.json"
    files = glob.glob(pattern)

    stats = {
        "total_files": len(files),
        "total_pages": 0,
        "cleaned_pages": 0,
        "chars_removed": 0,
    }

    logger.info(f"Found {len(files)} wiki data files in {input_dir}")

    for file_path in files:
        filename = os.path.basename(file_path)
        output_path = os.path.join(output_dir, filename)

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                pages = json.load(f)

            cleaned_pages = []
            for page in pages:
                stats["total_pages"] += 1

                original_content = page.get('content', '')
                cleaned_content = clean_page_content(original_content)

                if len(cleaned_content) < len(original_content):
                    stats["cleaned_pages"] += 1
                    stats["chars_removed"] += len(original_content) - len(cleaned_content)

                cleaned_page = {
                    **page,
                    'content': cleaned_content,
                }
                cleaned_pages.append(cleaned_page)

            # Save cleaned pages
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(cleaned_pages, f, ensure_ascii=False, indent=2)

            logger.info(f"Cleaned {filename}: {len(pages)} pages")

        except Exception as e:
            logger.error(f"Failed to process {file_path}: {e}")

    return stats


if __name__ == "__main__":
    import sys

    input_dir = sys.argv[1] if len(sys.argv) > 1 else "../../data/deduplicated_data"
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "../../data/cleaned_data"

    logger.info("=== Wikipedia Page Cleaning ===")
    logger.info(f"Input:  {input_dir}")
    logger.info(f"Output: {output_dir}")
    logger.info("")

    stats = clean_wiki_files(input_dir, output_dir)

    logger.info("")
    logger.info("=== Cleaning Statistics ===")
    logger.info(f"Total files:     {stats['total_files']}")
    logger.info(f"Total pages:     {stats['total_pages']}")
    logger.info(f"Cleaned pages:   {stats['cleaned_pages']} ({100*stats['cleaned_pages']/stats['total_pages']:.1f}%)")
    logger.info(f"Chars removed:   {stats['chars_removed']:,} ({stats['chars_removed']/stats['total_pages']:.0f} avg per page)")
