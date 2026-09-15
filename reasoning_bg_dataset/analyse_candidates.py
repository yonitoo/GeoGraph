import json
import re
from collections import Counter, defaultdict

with open("geography_questions_new_only.json", encoding="utf-8") as f:
    questions = json.load(f)

year_re = re.compile(r"(200[89]|201[0-9]|202[0-9])")

def extract_year(url):
    m = year_re.search(url)
    return m.group(1) if m else "unknown"

BUCKETS = {
    "relief":           ["релеф", "хребет", "вър", "планин", "котловин", "низин", "плато",
                         "рид", "проход", "седловин", "дол", "скал", "пещер"],
    "hydrography":      ["река", "реки", "езер", "вод", "поречи", "басейн", "водопад",
                         "канал", "язовир", "крайбрежи", "море", "залив", "нос"],
    "climate":          ["климат", "температур", "валеж", "вятър", "борá", "бриз",
                         "атмосфер", "озон", "циркулац", "брегова", "черноморск"],
    "soils_vegetation": ["почв", "гор", "дърв", "раститeл", "растит", "зона", "пояс",
                         "биом", "лонгоз", "степ", "ливад"],
    "admin_settlements": ["облает", "общин", "област", "районн", "гр.", "град",
                          "населен", "гкпп", "границ", "административн"],
    "economy_transport": ["стопанств", "отрасъл", "промишл", "земедел", "транспорт",
                          "магистрал", "железопът", "пристанищ", "енерги", "туризъм",
                          "селско", "животновъд", "растениевъд", "нефт", "въглищ",
                          "металург", "химическ", "хранит", "текстил"],
    "population":       ["население", "демограф", "раждаем", "смъртност", "гъстот",
                         "урбаниз", "миграц", "естествен прираст", "застарявa"],
    "protected_areas":  ["национален парк", "природен парк", "резерват", "юнеско",
                         "защитен", "еделвайс", "биосфер"],
    "world_geography":  ["континент", "световн", "регион", "африк", "азия", "европ",
                         "америк", "азиатск", "страна", "држав", "пазарно стопанство",
                         "продоволстве", "мегалоп"],
}

def classify(question_text):
    q = question_text.lower()
    hits = {}
    for bucket, keywords in BUCKETS.items():
        count = sum(kw in q for kw in keywords)
        if count:
            hits[bucket] = count
    if not hits:
        return "other"
    return max(hits, key=hits.get)

bg_markers = ["българия", "българия", "балкан", "дунав", "черно море", "марица",
              "искър", "рила", "родоп", "стара планин", "пирин", "вит", "тунджа"]

def is_bulgaria_specific(q_text):
    q = q_text.lower()
    return any(m in q for m in bg_markers)

year_counter = Counter()
bucket_counter = Counter()
bg_count = 0
annotated = []

for item in questions:
    year = extract_year(item["url"])
    bucket = classify(item["question"])
    is_bg = is_bulgaria_specific(item["question"])

    year_counter[year] += 1
    bucket_counter[bucket] += 1
    if is_bg:
        bg_count += 1

    annotated.append({**item, "year": year, "topic_bucket": bucket, "bulgaria_specific": is_bg})

print(f"Total candidates: {len(questions)}")
print(f"Bulgaria-specific: {bg_count} ({100*bg_count/len(questions):.1f}%)")

print("\nBy exam year:")
for yr, cnt in sorted(year_counter.items()):
    print(f"  {yr}: {cnt}")

print("\nBy topic bucket:")
for bucket, cnt in sorted(bucket_counter.items(), key=lambda x: -x[1]):
    print(f"  {bucket:<22}: {cnt}")

# Save annotated version
with open("geography_candidates_annotated.json", "w", encoding="utf-8") as f:
    json.dump(annotated, f, ensure_ascii=False, indent=2)

print("\nSaved annotated candidates → geography_candidates_annotated.json")
