import json
import os
from typing import List, Set, Dict

from tqdm import tqdm


def deduplicate_wiki_pages(input_json_path: str, output_json_path: str):
    print(f"Starting deduplication process...\n Input file: {input_json_path}")
    print(f"Output file: {output_json_path}")

    if not os.path.exists(input_json_path):
        print(f"Error: Input file not found at '{input_json_path}'")
        return

    unique_pages: List[Dict] = []
    seen_titles: Set[str] = set()
    duplicate_count = 0
    pages_without_title = 0

    try:
        with open(input_json_path, 'r', encoding='utf-8') as f_in:
            all_pages = json.load(f_in)

        if not isinstance(all_pages, list):
            print(f"Error: Input file '{input_json_path}' does not contain a valid JSON list.")
            return

        print(f"Read {len(all_pages)} pages from input file.")

        print("Processing pages and removing duplicates...")
        for page in tqdm(all_pages, desc="Deduplicating"):
            if not isinstance(page, dict):
                print(f"Warning: Skipping non-dictionary item found in the list: {page}")
                continue

            title = page.get("title")

            if not title:
                pages_without_title += 1
                continue

            if title not in seen_titles:
                seen_titles.add(title)
                unique_pages.append(page)
            else:
                duplicate_count += 1

        print(f"Finished processing.")
        if pages_without_title > 0:
            print(f"Skipped {pages_without_title} entries due to missing or empty titles.")
        print(f"Found and removed {duplicate_count} duplicate pages based on title.")
        print(f"Writing {len(unique_pages)} unique pages to '{output_json_path}'...")

        with open(output_json_path, 'w', encoding='utf-8') as f_out:
            json.dump(unique_pages, f_out, ensure_ascii=False, indent=4)

        print("Deduplication complete. Output file saved.")

    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from '{input_json_path}': {e}")
    except IOError as e:
        print(f"Error reading or writing file: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    all_wiki_categories = [filename for filename in os.listdir("./") if filename.startswith("fetched-wiki")]
    for category in all_wiki_categories:
        deduplicate_wiki_pages(category, "./deduplicated_data/" + category)
