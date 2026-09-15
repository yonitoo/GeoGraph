import json
from pathlib import Path

dataset_dir = Path(__file__).parent


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def build_url_mapping(mapping):
    return {entry["current_url"]: entry["new_url"] for entry in mapping}


def remap_urls(questions_path, url_map):
    questions = load_json(questions_path)
    changed = 0
    for entry in questions:
        new_url = url_map.get(entry["url"])
        if new_url is not None:
            entry["url"] = new_url
            changed += 1
    save_json(questions_path, questions)
    return changed


def main():
    url_map = build_url_mapping(load_json(dataset_dir / "url_mapping.json"))

    questions_files = sorted(dataset_dir.glob("geography_questions_*.json"))
    questions_files = [p for p in questions_files if "reasoning" not in p.name]

    for questions_path in questions_files:
        changed = remap_urls(questions_path, url_map)
        print(f"Updated {changed} entries in {questions_path.name}")


if __name__ == "__main__":
    main()