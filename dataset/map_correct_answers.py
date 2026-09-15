import json
import glob
from pathlib import Path

dataset_dir = Path(__file__).parent


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def build_answers_index(answers):
    return {(entry["qid"], entry["url"]): entry["correct"] for entry in answers}


def fill_correct_answers(questions_path, answers_index):
    questions = load_json(questions_path)
    for entry in questions:
        key = (entry["qid"], entry["url"])
        correct_index = answers_index.get(key)
        if correct_index is not None:
            entry["correct"] = entry["answers"][correct_index - 1]
    save_json(questions_path, questions)
    return len(questions)


def main():
    questions_files = sorted(dataset_dir.glob("geography_questions_*.json"))
    questions_files = [p for p in questions_files if "reasoning" not in p.name]

    for questions_path in questions_files:
        suffix = questions_path.stem.replace("geography_questions_", "")
        answers_path = dataset_dir / f"answers_{suffix}.json"

        if not answers_path.exists():
            print(f"No answers file for {questions_path.name}, skipping.")
            continue

        answers_index = build_answers_index(load_json(answers_path))
        count = fill_correct_answers(questions_path, answers_index)
        print(f"Updated {count} entries in {questions_path.name}")


if __name__ == "__main__":
    main()
