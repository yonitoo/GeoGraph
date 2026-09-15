from fastapi import APIRouter, HTTPException

from geograph.api.schemas import EvaluateRequest, EvaluateResponse
from geograph.eval.evaluator import evaluate

router = APIRouter(prefix="/api/evaluation", tags=["evaluation"])


@router.post("/evaluate", response_model=EvaluateResponse)
def run_evaluation(req: EvaluateRequest):
    """Evaluate prediction accuracy against a testset."""
    try:
        result = evaluate(req.input_file, req.answer_key, req.label)
    except Exception as e:
        raise HTTPException(500, f"Evaluation failed: {e}")

    return EvaluateResponse(
        label=result.label,
        accuracy=result.accuracy,
        total=result.total,
        correct=result.correct,
        errors=[
            {"question": q, "expected": exp, "got": got}
            for q, exp, got in result.errors
        ],
    )
