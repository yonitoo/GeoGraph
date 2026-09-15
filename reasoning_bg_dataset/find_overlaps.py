import json
import re
import unicodedata

TESTSET = "../testset/zeroshot_kg_rag_bggpt_curated_exams.jsonl"
GEO_POOL = "geography_questions.json"
OUTPUT = "geography_questions_new_only.json"


def normalise(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = text.lower().strip()
    text = re.sub(r"[\s]+", " ", text)
    text = re.sub(r"[^\w\s]", "", text)
    return text


testset = []
with open(TESTSET, encoding="utf-8") as f:
    for line in f:
        testset.append(json.loads(line))

testset_questions = {normalise(q["question"]) for q in testset}

with open(GEO_POOL, encoding="utf-8") as f:
    geo_pool = json.load(f)

duplicates = []
new_questions = []

for item in geo_pool:
    key = normalise(item["question"])
    if key in testset_questions:
        duplicates.append(item)
    else:
        new_questions.append(item)

print(f"Testset size        : {len(testset)}")
print(f"Geography pool size : {len(geo_pool)}")
print(f"Duplicates found    : {len(duplicates)}")
print(f"New (unique) items  : {len(new_questions)}")

if duplicates:
    print("\n--- Duplicate questions ---")
    for d in duplicates:
        print(f"  [{d['id'][:8]}] {d['question'][:90]}")

with open(OUTPUT, "w", encoding="utf-8") as f:
    json.dump(new_questions, f, ensure_ascii=False, indent=2)

print(f"\nSaved {len(new_questions)} unique geography questions → {OUTPUT}")
