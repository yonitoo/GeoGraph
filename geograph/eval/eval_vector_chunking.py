import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

BASELINE_PATH = "testset/baselines_output.jsonl"
BASELINE_MODEL = "bggpt-gemma3-27b"


@dataclass
class ResultSpec:
    path: str
    answer_key: str


RESULTS: dict[str, ResultSpec] = {
    "fixed256": ResultSpec(
        "testset/vector_chunking/fixed256/pilot_bgem3/merged_0001_0777.jsonl",
        "vector_fixed256_answer",
    ),
    "structure": ResultSpec(
        "testset/vector_chunking/structure/pilot_bgem3/batch.json",
        "vector_structure_answer",
    ),
    "parentchild": ResultSpec(
        "testset/vector_chunking/parentchild/pilot_bgem3/batch.json",
        "vector_parentchild_answer",
    ),
    "kg_rag": ResultSpec(
        "testset/kg_rag/full_bgem3/batch.jsonl",
        "bggpt_kg_rag_answer",
    ),
    "kg_rag_v2": ResultSpec(
        "testset/kg_rag/pilot_bgem3_v2/batch_lexical_curated2_all.jsonl",
        "bggpt_kg_rag_answer",
    ),
    "kg_rag_v3": ResultSpec(
        "testset/kg_rag/full_bgem3_v3/batch.jsonl",
        "bggpt_kg_rag_answer",
    ),
    "kg_rag_v4": ResultSpec(
        "testset/kg_rag/full_bgem3_v3/batch_round_with_res_filter.jsonl",
        "bggpt_kg_rag_answer",
    ),
}


def load_records(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def evaluate(records: list[dict], answer_key: str) -> dict:
    recall_key = f"{answer_key}_recall"
    ctx_key = f"{answer_key}_ctx_n"

    n = len(records)
    correct = 0
    abstained = 0
    recall_hits = 0
    ctx_sum = 0
    by_source = defaultdict(lambda: {"n": 0, "correct": 0})

    for r in records:
        answer = r.get(answer_key)
        source = r.get("source_dataset", "unknown")
        by_source[source]["n"] += 1

        if answer in ("-", "", None):
            abstained += 1
        elif answer == r.get("correct_answer"):
            correct += 1
            by_source[source]["correct"] += 1

        if r.get(recall_key):
            recall_hits += 1
        ctx_sum += r.get(ctx_key, 0) or 0

    return {
        "n": n,
        "accuracy": correct / n,
        "abstention": abstained / n,
        "recall_proxy": recall_hits / n,
        "avg_ctx_chunks": ctx_sum / n,
        "by_source": {
            source: d["correct"] / d["n"] for source, d in by_source.items()
        },
    }


def evaluate_baseline(records: list[dict]) -> dict:
    n = len(records)
    correct = 0
    by_source = defaultdict(lambda: {"n": 0, "correct": 0})

    for r in records:
        answer = r.get("model_answers", {}).get(BASELINE_MODEL)
        source = r.get("source_dataset", "unknown")
        by_source[source]["n"] += 1
        if answer == r.get("correct_answer"):
            correct += 1
            by_source[source]["correct"] += 1

    return {
        "n": n,
        "accuracy": correct / n,
        "abstention": 0.0,
        "recall_proxy": None,
        "avg_ctx_chunks": None,
        "by_source": {
            source: d["correct"] / d["n"] for source, d in by_source.items()
        },
    }


def evaluate_hybrid(records: list[dict], answer_key: str, baseline_by_question: dict[str, str | None]) -> dict:
    n = len(records)
    correct = 0
    by_source = defaultdict(lambda: {"n": 0, "correct": 0})

    for r in records:
        gt = r["correct_answer"]
        source = r.get("source_dataset", "unknown")
        by_source[source]["n"] += 1

        rag_answer = r.get(answer_key)
        answer = baseline_by_question.get(r["question"]) if rag_answer in ("-", "", None) else rag_answer

        if answer == gt:
            correct += 1
            by_source[source]["correct"] += 1

    return {
        "n": n,
        "accuracy": correct / n,
        "abstention": 0.0,
        "recall_proxy": None,
        "avg_ctx_chunks": None,
        "by_source": {
            source: d["correct"] / d["n"] for source, d in by_source.items()
        },
    }


def print_report(results: dict[str, dict]) -> None:
    header = f"{'strategy':<22}{'n':>6}{'accuracy':>10}{'abstained':>11}{'recall':>9}{'avg_ctx':>9}"
    print(header)
    for strategy, m in results.items():
        recall = f"{m['recall_proxy']*100:>8.1f}%" if m["recall_proxy"] is not None else f"{'-':>9}"
        avg_ctx = f"{m['avg_ctx_chunks']:>9.2f}" if m["avg_ctx_chunks"] is not None else f"{'-':>9}"
        print(
            f"{strategy:<22}{m['n']:>6}"
            f"{m['accuracy']*100:>9.1f}%"
            f"{m['abstention']*100:>10.1f}%"
            f"{recall}"
            f"{avg_ctx}"
        )

    print()
    sources = sorted({s for m in results.values() for s in m["by_source"]})
    header = f"{'strategy':<22}" + "".join(f"{s:>16}" for s in sources)
    print(header)
    for strategy, m in results.items():
        row = f"{strategy:<22}"
        for source in sources:
            acc = m["by_source"].get(source)
            row += f"{acc*100:>15.1f}%" if acc is not None else f"{'-':>16}"
        print(row)


def main() -> None:
    root = Path(__file__).resolve().parent.parent.parent
    baseline_records = load_records(str(root / BASELINE_PATH))
    baseline_by_question = {r["question"]: r["model_answers"].get(BASELINE_MODEL) for r in baseline_records}

    results = {"baseline": evaluate_baseline(baseline_records)}
    for name, spec in RESULTS.items():
        full_path = root / spec.path
        if not full_path.exists():
            continue
        records = load_records(str(full_path))
        results[name] = evaluate(records, spec.answer_key)
        results[f"{name}+fallback"] = evaluate_hybrid(records, spec.answer_key, baseline_by_question)
    print_report(results)


if __name__ == "__main__":
    main()
