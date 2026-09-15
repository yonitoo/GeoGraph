import argparse
import json
import os
import pathlib
import subprocess
import sys
import time

INPUT = "testset/geography_777_benchmark.jsonl"
OUTPUT_DIR = "testset/baselines"
MERGED_OUTPUT = "testset/baselines_output.jsonl"
BATCH_SIZE = 100


def _count_lines(path: str) -> int:
    with open(path, encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def _batches(total: int, batch_size: int) -> list[tuple[int, int]]:
    """Return (offset, limit) pairs covering all questions."""
    batches = []
    offset = 0
    while offset < total:
        limit = min(batch_size, total - offset)
        batches.append((offset, limit))
        offset += limit
    return batches


def _output_path(offset: int, limit: int) -> str:
    end = offset + limit
    return os.path.join(OUTPUT_DIR, f"batch_{offset+1:04d}_{end:04d}.jsonl")


def run_parallel(dry_run: bool, run_eval: bool) -> None:
    total = _count_lines(INPUT)
    batches = _batches(total, BATCH_SIZE)

    print(f"Total questions : {total}")
    print(f"Batch size      : {BATCH_SIZE}")
    print(f"Batches         : {len(batches)}")
    print()

    procs = []
    for offset, limit in batches:
        output_file = _output_path(offset, limit)
        cmd = [
            sys.executable, "-m", "geograph.runners.run_baselines",
            "--offset", str(offset),
            "--limit", str(limit),
            "--output", output_file,
            "--delay", "1.0",
        ]
        label = f"Q{offset+1:04d}–Q{offset+limit:04d}"
        print(f"  {label}  →  {output_file}")
        if not dry_run:
            log_dir = "logs"
            pathlib.Path(log_dir).mkdir(parents=True, exist_ok=True)
            log_path = os.path.join(log_dir, f"batch_{offset+1:04d}_{offset+limit:04d}.stdout")
            log_fh = open(log_path, "w")
            proc = subprocess.Popen(cmd, stdout=log_fh, stderr=subprocess.STDOUT)
            procs.append((label, proc, log_fh, log_path))

    if dry_run:
        print("\n(dry run — no processes started)")
        return

    print(f"\nStarted {len(procs)} parallel processes. Waiting for completion...")
    failed = []
    for label, proc, log_fh, log_path in procs:
        proc.wait()
        log_fh.close()
        status = "OK" if proc.returncode == 0 else f"FAILED (exit {proc.returncode})"
        print(f"  {label}  {status}  (log: {log_path})")
        if proc.returncode != 0:
            failed.append(label)

    if failed:
        print(f"\nFailed batches: {failed}")
    else:
        print("\nAll batches completed successfully.")
        merge_outputs()
        if run_eval:
            evaluate_merged()


def merge_outputs() -> None:
    total = _count_lines(INPUT)
    batches = _batches(total, BATCH_SIZE)
    output_files = [_output_path(o, l) for o, l in batches]

    missing = [f for f in output_files if not pathlib.Path(f).exists()]
    if missing:
        print(f"Cannot merge — missing batch files: {missing}")
        return

    records = []
    for path in output_files:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))

    with open(MERGED_OUTPUT, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"\nMerged {len(records)} records → {MERGED_OUTPUT}")


def evaluate_merged() -> None:
    from geograph.eval.evaluator import evaluate
    print("EVALUATION - merged results\n")
    for model_key in ["bggpt-gemma3-27b"]:
        result = evaluate(MERGED_OUTPUT, f"model_answers.{model_key}", model_key)
        print(result.summary())


def main():
    parser = argparse.ArgumentParser(description="Run all baseline batches in parallel")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands without running them")
    parser.add_argument("--evaluate", action="store_true",
                        help="Evaluate merged results after all batches complete")
    parser.add_argument("--merge-only", action="store_true",
                        help="Merge existing batch files and evaluate, skip running")
    args = parser.parse_args()

    if args.merge_only:
        merge_outputs()
        evaluate_merged()
    else:
        run_parallel(dry_run=args.dry_run, run_eval=args.evaluate)


if __name__ == "__main__":
    main()
