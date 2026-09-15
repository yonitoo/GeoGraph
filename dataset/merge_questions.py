import json
from pathlib import Path

dataset_dir = Path(__file__).parent


def load_json(path):
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def save_json(path, data):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main():
    question_files = sorted(
        p for p in dataset_dir.glob('geography_questions_*.json')
        if 'reasoning' not in p.name
    )

    merged = []
    for path in question_files:
        entries = load_json(path)
        merged.extend(entries)
        print(f'Loaded {len(entries):>3} entries from {path.name}')

    output_path = dataset_dir / 'geography_questions_new.json'
    save_json(output_path, merged)
    print(f'\nMerged {len(merged)} total entries into {output_path.name}')


if __name__ == '__main__':
    main()
