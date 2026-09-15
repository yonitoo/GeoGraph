import json
import re
import unicodedata
from pathlib import Path

DATASET_DIR = Path(__file__).parent
OUTPUT = DATASET_DIR.parent / "testset" / "geography_777_benchmark.jsonl"
LABELS = ["A", "B", "C", "D"]


def normalise(text: str) -> str:
    text = unicodedata.normalize("NFC", text).lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s]", "", text)
    return text


def to_testset_record(item: dict, source_dataset: str) -> dict:
    answers = item["answers"]
    correct_text = item["correct"].strip()
    correct_label = "-"
    for label, text in zip(LABELS, answers):
        if text.strip() == correct_text:
            correct_label = label
            break

    return {
        "id": item.get("id", ""),
        "qid": item.get("qid", 0),
        "source_url": item.get("url", ""),
        "source_dataset": source_dataset,
        "question": item["question"],
        "options": [{"label": l, "text": t} for l, t in zip(LABELS, answers)],
        "correct_answer": correct_label,
        "model_answers": {},
    }


def main():
    rbg = json.loads((DATASET_DIR / "geography_questions_reasoning_bg.json").read_text(encoding="utf-8"))
    new = json.loads((DATASET_DIR / "geography_questions_new.json").read_text(encoding="utf-8"))

    seen_keys: set[str] = set()
    records = []

    for item in rbg:
        key = normalise(item["question"])
        if key not in seen_keys:
            seen_keys.add(key)
            records.append(to_testset_record(item, "reasoning_bg"))

    for item in new:
        key = normalise(item["question"])
        if key not in seen_keys:
            seen_keys.add(key)
            records.append(to_testset_record(item, "new"))

    unresolved = [r for r in records if r["correct_answer"] == "-"]
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"reasoning_bg loaded : {len(rbg)}")
    print(f"new loaded          : {len(new)}")
    print(f"Total unique        : {len(records)}")
    print(f"Unresolved answers  : {len(unresolved)}")
    if unresolved:
        for r in unresolved[:5]:
            print(f"  UNRESOLVED: {r['question'][:80]}")
    print(f"Written → {OUTPUT}")


if __name__ == "__main__":
    main()
