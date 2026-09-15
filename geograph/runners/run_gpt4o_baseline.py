import argparse
import json
import os
import pathlib
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI

from geograph.eval.evaluator import evaluate
from geograph.llm.llm_client import extract_mcq_letter
from geograph.llm.prompt_builder import build_mcq_prompt

DEFAULT_INPUT = "testset/geography_777_benchmark.jsonl"
DEFAULT_OUTPUT = "testset/baselines/gpt4o_zeroshot.jsonl"
MODEL_KEY = "gpt-4o"


def _load_done(output_file: str, model_key: str) -> dict:
    """Return {question_text: answer} for already-answered questions."""
    done = {}
    p = pathlib.Path(output_file)
    if p.exists():
        with p.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    ans = rec.get("model_answers", {}).get(model_key)
                    if ans not in (None, ""):
                        done[rec["question"]] = ans
                except (json.JSONDecodeError, KeyError):
                    continue
    return done


def _answer_one(client: OpenAI, model: str, exam: dict) -> str:
    prompt = build_mcq_prompt(exam["question"], exam["options"])
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=10,
    )
    text = resp.choices[0].message.content or ""
    return extract_mcq_letter(text) or "-"


def main():
    ap = argparse.ArgumentParser(description="Zero-shot GPT-4o baseline")
    ap.add_argument("--input", default=DEFAULT_INPUT)
    ap.add_argument("--output", default=DEFAULT_OUTPUT)
    ap.add_argument("--model", default=MODEL_KEY, help="OpenAI model id (default gpt-4o)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None, help="For a quick smoke test")
    ap.add_argument("--evaluate", action="store_true")
    args = ap.parse_args()

    try:
        from dotenv import load_dotenv
        load_dotenv(dotenv_path=pathlib.Path(__file__).parent.parent / ".env")
    except ImportError:
        pass
    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not set (geograph/.env)")
        sys.exit(1)

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    questions = []
    with open(args.input, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                questions.append(json.loads(line))
    if args.limit:
        questions = questions[: args.limit]

    done = _load_done(args.output, args.model)
    records = [{**q, "model_answers": {**q.get("model_answers", {})}} for q in questions]
    todo = []
    for idx, q in enumerate(questions):
        prev = done.get(q["question"])
        if prev is not None:
            records[idx]["model_answers"][args.model] = prev
        else:
            todo.append(idx)
    print(f"Total {len(questions)} | already done {len(questions)-len(todo)} | to do {len(todo)}")

    pathlib.Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(_answer_one, client, args.model, questions[idx]): idx for idx in todo}
        for i, fut in enumerate(as_completed(futures), 1):
            idx = futures[fut]
            try:
                ans = fut.result()
            except Exception as exc:
                print(f"  ! idx {idx} error: {exc}")
                ans = "-"
            records[idx]["model_answers"][args.model] = ans
            if i % 25 == 0:
                print(f"  {i}/{len(todo)}  ({time.time()-t0:.0f}s)")

    # Write once, one line per benchmark row, in original order.
    with open(args.output, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"Wrote {len(records)} records -> {args.output} in {time.time()-t0:.0f}s")

    if args.evaluate:
        print("\nEVALUATION\n")
        overall = evaluate(args.output, f"model_answers.{args.model}", args.model)
        print(overall.summary())
        import tempfile
        for src in ("reasoning_bg", "new"):
            tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False, encoding="utf-8")
            with open(args.output, encoding="utf-8") as f:
                for line in f:
                    rec = json.loads(line)
                    if rec.get("source_dataset") == src:
                        tmp.write(line)
            tmp.close()
            count = sum(1 for _ in open(tmp.name))
            if count:
                r = evaluate(tmp.name, f"model_answers.{args.model}", f"{args.model}[{src}]")
                print(f"  {r.summary()}")
            os.unlink(tmp.name)


if __name__ == "__main__":
    main()
