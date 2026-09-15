import logging
from dataclasses import dataclass, field
from typing import List, Tuple

from jsonlines import jsonlines

logger = logging.getLogger(__name__)


@dataclass
class EvalResult:
    """Evaluation result for a single experiment run."""
    label: str
    total: int = 0
    correct: int = 0
    errors: List[Tuple[str, str, str]] = field(default_factory=list)

    @property
    def accuracy(self) -> float:
        return (self.correct / self.total) * 100 if self.total else 0.0

    def summary(self) -> str:
        return f"{self.label} Accuracy: {self.accuracy:.2f}% ({self.correct}/{self.total})"


def evaluate(input_file: str, answer_key: str, label: str = "") -> EvalResult:
    """Evaluate accuracy by comparing predicted answers to correct answers.

    Args:
        input_file: Path to JSONL file with evaluation data.
        answer_key: Dot-separated path to the predicted answer field.
                    Examples: "bggpt_kg_rag_answer", "bggpt_vector_rag_answer",
                              "model_answers.bggpt-gemma2"
        label: Human-readable label for the result.
    """
    result = EvalResult(label=label or answer_key)

    with jsonlines.open(input_file) as reader:
        for entry in reader:
            correct = entry.get("correct_answer")
            predicted = _get_nested(entry, answer_key)

            if predicted is None:
                logger.warning("Missing answer key '%s' in entry: %s", answer_key, entry.get("question", "?"))
                continue

            result.total += 1
            if predicted == correct:
                result.correct += 1
            else:
                result.errors.append((
                    entry.get("question", ""),
                    correct,
                    predicted,
                ))

    logger.info(result.summary())
    for q, expected, got in result.errors:
        logger.debug("  WRONG: Q='%s', Expected='%s', Got='%s'", q, expected, got)

    return result


def _get_nested(data: dict, key_path: str):
    """Get a value from a dict using a dot-separated key path."""
    parts = key_path.split(".")
    current = data
    for part in parts:
        if isinstance(current, dict):
            current = current.get(part)
        else:
            return None
    return current


if __name__ == "__main__":
    import sys

    file_path = sys.argv[1] if len(sys.argv) > 1 else "testset/zeroshot_kg_rag_bggpt_curated_exams.jsonl"
    logging.basicConfig(level=logging.INFO)

    r1 = evaluate(file_path, "model_answers.bggpt-gemma2", "BgGPT Zero-Shot")
    print(r1.summary())

    r2 = evaluate(file_path, "bggpt_kg_rag_answer", "BgGPT KG-RAG")
    print(r2.summary())