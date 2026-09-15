import argparse
import datetime
import json
import logging
import os
import pathlib
import sys
import time
from typing import Optional

from geograph.config import llm_configs_for_baselines
from geograph.eval.evaluator import evaluate
from geograph.llm.llm_client import BgGPTChatClient
from geograph.llm.prompt_builder import build_mcq_prompt

logger = logging.getLogger(__name__)

DEFAULT_INPUT = "testset/geography_777_benchmark.jsonl"
DEFAULT_OUTPUT_DIR = "testset/baselines"
ALL_MODELS = ["bggpt-gemma3-27b"]
BATCH_SIZE = 100


def _setup_logging(log_dir: str, tag: str = "") -> None:
    pathlib.Path(log_dir).mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%H_%M_%S-%d_%m_%y")
    suffix = f"_{tag}" if tag else ""
    log_file = os.path.join(log_dir, f"baselines{suffix}_{stamp}.log")
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
        filename=log_file,
        filemode="w",
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter("%(levelname)s — %(message)s"))
    logging.getLogger("").addHandler(console)
    logger.info("Logging to %s", log_file)


def _load_existing(output_file: str) -> dict[str, dict]:
    existing: dict[str, dict] = {}
    p = pathlib.Path(output_file)
    if p.exists():
        with p.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    existing[rec["question"]] = rec
                except (json.JSONDecodeError, KeyError):
                    continue
    return existing


def _auto_output(output_dir: str, offset: int, end: int) -> str:
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    return os.path.join(output_dir, f"batch_{offset+1:04d}_{end:04d}.jsonl")


def run_baselines(
    models: list[str],
    input_file: str,
    output_file: str,
    offset: int,
    limit: Optional[int],
    delay: float,
    log_dir: str,
    run_eval: bool,
) -> None:
    try:
        from dotenv import load_dotenv
        load_dotenv(dotenv_path=pathlib.Path(__file__).parent.parent / ".env")
    except ImportError:
        pass

    configs = llm_configs_for_baselines()
    for m in models:
        if not configs[m].api_key:
            logger.error("BGGPT_API_KEY not set. Add it to geograph/.env")
            sys.exit(1)

    clients = {m: BgGPTChatClient(configs[m]) for m in models}

    # Load and slice input
    all_questions = []
    with open(input_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                all_questions.append(json.loads(line))

    end = offset + limit if limit else len(all_questions)
    end = min(end, len(all_questions))
    questions = all_questions[offset:end]

    tag = f"{offset+1}_{end}"
    _setup_logging(log_dir, tag)

    existing = _load_existing(output_file)
    logger.info(
        "Batch Q%d–Q%d (%d questions) | Models: %s | Already answered: %d",
        offset + 1, end, len(questions), models, len(existing),
    )

    pathlib.Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    results = []
    for i, exam in enumerate(questions):
        global_idx = offset + i + 1
        question = exam["question"]
        record = existing.get(question, {**exam})

        if "model_answers" not in record:
            record["model_answers"] = {}

        prompt = build_mcq_prompt(question, exam["options"])

        for model_key, client in clients.items():
            if model_key in record["model_answers"]:
                logger.debug("Q%d — %s already answered, skipping", global_idx, model_key)
                continue

            t0 = time.time()
            try:
                answer = client.generate_answer_letter(prompt)
            except Exception as exc:
                logger.warning("Q%d — %s error: %s", global_idx, model_key, exc)
                answer = None

            elapsed = time.time() - t0
            record["model_answers"][model_key] = answer or "-"
            logger.info(
                "Q%04d | %-22s | answer=%-1s | correct=%s | %.1fs | %s",
                global_idx, model_key, answer or "?",
                exam.get("correct_answer", "?"), elapsed,
                question[:60],
            )

            if delay > 0:
                time.sleep(delay)

        results.append(record)

    with open(output_file, "w", encoding="utf-8") as f:
        for rec in results:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    logger.info("Written → %s (%d records)", output_file, len(results))

    if run_eval:
        _print_eval(output_file, models)


def _print_eval(output_file: str, models: list[str]) -> None:
    import tempfile
    print("EVALUATION\n")
    for model_key in models:
        result = evaluate(output_file, f"model_answers.{model_key}", model_key)
        print(result.summary())

    sources = {"reasoning_bg", "new"}
    for src in sorted(sources):
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl",
                                         delete=False, encoding="utf-8")
        with open(output_file, encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                if rec.get("source_dataset") == src:
                    tmp.write(line)
        tmp.close()
        count = sum(1 for _ in open(tmp.name))
        if count:
            print(f"\n  Slice [{src}] ({count} questions)")
            for model_key in models:
                r = evaluate(tmp.name, f"model_answers.{model_key}", f"{model_key}[{src}]")
                print(f"  {r.summary()}")
        os.unlink(tmp.name)


def main():
    parser = argparse.ArgumentParser(description="Zero-shot BgGPT v3 baseline evaluation")
    parser.add_argument("--models", default=",".join(ALL_MODELS),
                        help=f"Comma-separated model keys (default: {','.join(ALL_MODELS)})")
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output", default=None,
                        help="Output JSONL file (auto-named by range if omitted)")
    parser.add_argument("--offset", type=int, default=0,
                        help="Start index (0-based, default 0)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Number of questions to process (default: all from offset)")
    parser.add_argument("--delay", type=float, default=1.0,
                        help="Seconds between API calls (default: 1.0)")
    parser.add_argument("--log-dir", default="logs")
    parser.add_argument("--evaluate", action="store_true",
                        help="Print accuracy after run")
    args = parser.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    unknown = [m for m in models if m not in ALL_MODELS]
    if unknown:
        print(f"Unknown model(s): {unknown}. Choose from: {ALL_MODELS}")
        sys.exit(1)

    if args.output:
        output_file = args.output
    else:
        total = sum(1 for _ in open(args.input, encoding="utf-8") if _.strip())
        end = min(args.offset + args.limit, total) if args.limit else total
        output_file = _auto_output(DEFAULT_OUTPUT_DIR, args.offset, end)

    run_baselines(
        models=models,
        input_file=args.input,
        output_file=output_file,
        offset=args.offset,
        limit=args.limit,
        delay=args.delay,
        log_dir=args.log_dir,
        run_eval=args.evaluate,
    )


if __name__ == "__main__":
    main()
