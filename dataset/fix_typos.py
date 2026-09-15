import json
import re
from pathlib import Path

dataset_dir = Path(__file__).parent

LATIN_TO_CYRILLIC = str.maketrans(
    'aeocpxAEKMHOPCTXB',
    'аеосрхАЕКМНОРСТХВ',
)

KNOWN_CORRECTIONS = {
    'centralен': 'централен',
    'socialно-икономическия': 'социално-икономическия',
}

HAS_CYRILLIC = re.compile(r'[а-яА-ЯёЁ]')
HAS_LATIN = re.compile(r'[a-zA-Z]')


def fix_mixed_word(word):
    if HAS_CYRILLIC.search(word) and HAS_LATIN.search(word):
        return word.translate(LATIN_TO_CYRILLIC)
    return word


def fix_text(text):
    for bad, good in KNOWN_CORRECTIONS.items():
        text = text.replace(bad, good)
    return re.sub(r'\S+', lambda m: fix_mixed_word(m.group()), text)


def load_json(path):
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def save_json(path, data):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def fix_questions_file(path):
    questions = load_json(path)
    changed = 0
    for entry in questions:
        original = json.dumps(entry, ensure_ascii=False)
        entry['question'] = fix_text(entry.get('question', ''))
        entry['answers'] = [fix_text(a) for a in entry.get('answers', [])]
        if isinstance(entry.get('correct'), str):
            entry['correct'] = fix_text(entry['correct'])
        if json.dumps(entry, ensure_ascii=False) != original:
            changed += 1
    save_json(path, questions)
    return changed


def main():
    for path in sorted(dataset_dir.glob('geography_questions_*.json')):
        changed = fix_questions_file(path)
        print(f'Fixed {changed} entries in {path.name}')


if __name__ == '__main__':
    main()
