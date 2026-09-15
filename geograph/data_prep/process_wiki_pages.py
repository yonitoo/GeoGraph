import json
import os
import re
from typing import List, Dict, Counter

from jsonlines import jsonlines


def clean_wiki_content(content: str, unwanted_sections: List[str]) -> str:
    cleaned_content = content
    section_pattern = "|".join([re.escape(sec) for sec in unwanted_sections])
    regex = re.compile(rf"^\s*==\s*({section_pattern})\s*==.*", re.MULTILINE | re.IGNORECASE | re.DOTALL)

    match = regex.search(cleaned_content)
    if match:
        cleaned_content = cleaned_content[:match.start()]

    # Почистване на често срещани wiki елементи
    cleaned_content = re.sub(r'<ref.*?>.*?</ref>', '', cleaned_content, flags=re.IGNORECASE | re.DOTALL)
    cleaned_content = re.sub(r'{{.*?}}', '', cleaned_content, flags=re.DOTALL)
    cleaned_content = re.sub(r'\[\[Файл:.*?\]\]', '', cleaned_content, flags=re.IGNORECASE)
    cleaned_content = re.sub(r'\[\[Категория:.*?\]\]', '', cleaned_content, flags=re.IGNORECASE)
    cleaned_content = re.sub(r"'''(.*?)'''", r"\1", cleaned_content)
    cleaned_content = re.sub(r"''(.*?)''", r"\1", cleaned_content)
    cleaned_content = re.sub(r'==+\s*(.*?)\s*==+', r'\n\n\1\n', cleaned_content)
    cleaned_content = re.sub(r'\[\[(?:[^|\]]*\|)?([^\]]+)\]\]', r'\1', cleaned_content)
    cleaned_content = re.sub(r'\n{3,}', '\n\n', cleaned_content)
    cleaned_content = re.sub(r'[ \t]+', ' ', cleaned_content)
    cleaned_content = cleaned_content.strip()
    return cleaned_content

def load_all_wiki_pages_from_file(json_file_path: str) -> List[Dict[str, str]]:
    if not os.path.exists(json_file_path):
        print(f"Error: Input JSON file not found at {json_file_path}")
        return []
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            pages = json.load(f)
        if not isinstance(pages, list):
            print(f"Error: Expected a JSON list in {json_file_path}")
            return []
        print(f"Loaded {len(pages)} pages from {json_file_path}")
        # check for duplicate page names
        titles = [p.get("title", "") for p in pages]
        if len(titles) != len(set(titles)):
             print("Warning: Duplicate page titles found in the input JSON file. This might indicate issues.")
             from collections import Counter
             title_counts = Counter(titles)
             duplicates = {title: count for title, count in title_counts.items() if count > 1}
             print(f"Duplicate titles: {duplicates}")
        return pages
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from {json_file_path}: {e}")
        return []
    except Exception as e:
        print(f"An unexpected error occurred loading {json_file_path}: {e}")
        return []

def load_all_wiki_pages_from_folder(folder_path: str) -> List[Dict[str, str]]:
    all_pages = []
    if not os.path.isdir(folder_path):
        print(f"Error: Input path is not a valid directory: {folder_path}")
        return []

    print(f"Scanning for .json and .jsonl files in '{folder_path}'...")
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if filename.endswith(".json"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        all_pages.extend(data)
                        print(f"  - Loaded {len(data)} pages from {filename}")
                    else:
                        print(f"  - Warning: {filename} does not contain a JSON list.")
            except Exception as e:
                print(f"  - Error loading {filename}: {e}")
        elif filename.endswith(".jsonl"):
            try:
                with jsonlines.open(file_path) as reader:
                    data = list(reader)
                    all_pages.extend(data)
                    print(f"  - Loaded {len(data)} pages from {filename}")
            except Exception as e:
                print(f"  - Error loading {filename}: {e}")

    print(f"\nTotal pages loaded (with potential duplicates): {len(all_pages)}")

    seen_titles = set()
    deduplicated_pages = []
    duplicate_titles = []

    for page in all_pages:
        title = page.get("title")
        if title and title not in seen_titles:
            seen_titles.add(title)
            deduplicated_pages.append(page)
        elif title:
            duplicate_titles.append(title)

    if duplicate_titles:
        duplicate_counts = Counter(duplicate_titles)
        print(f"Warning: Found and removed duplicates for the following titles:")
        for title, count in duplicate_counts.items():
            print(f"  - '{title}' (found {count + 1} times, kept 1)")
    else:
        print("No duplicate titles found.")

    print(f"\nTotal unique pages to be processed: {len(deduplicated_pages)}")
    return deduplicated_pages


if __name__ == "__main__":
    all_wiki_pages = load_all_wiki_pages_from_folder('./data/deduplicated_data')
    print('done')
