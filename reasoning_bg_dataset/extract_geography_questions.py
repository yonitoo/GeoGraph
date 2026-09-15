import json
import urllib.request

URL = "https://raw.githubusercontent.com/mhardalov/bg-reason-BERT/master/data/bg_rc-v1.0.json"
OUTPUT = "geography_questions.json"

with urllib.request.urlopen(URL) as response:
    data = json.loads(response.read().decode("utf-8"))

geo_pages = data["data"]["geography-12th"]

questions = []
for page in geo_pages:
    url = page["url"]
    for q in page["questions"]:
        questions.append(
            {
                "id": q["id"],
                "qid": q["qid"],
                "url": url,
                "question": q["question"],
                "answers": q["answers"],
                "correct": q["correct"],
            }
        )

with open(OUTPUT, "w", encoding="utf-8") as f:
    json.dump(questions, f, ensure_ascii=False, indent=2)

print(f"Extracted {len(questions)} geography questions → {OUTPUT}")
