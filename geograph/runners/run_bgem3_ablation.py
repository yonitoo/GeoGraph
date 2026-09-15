import argparse
import subprocess
import sys

ALL_STRATEGIES = ["structure", "parentchild"]
EMBEDDING_MODEL = "BAAI/bge-m3"
COLLECTION_SUFFIX = "_bgem3"
NUM_QUESTIONS = 777
EMBED_BATCH_SIZE = 32


def run_strategy(strategy: str) -> None:
    output_file = (
        f"testset/vector_chunking/{strategy}/pilot_bgem3/"
        f"batch_0001_{NUM_QUESTIONS:04d}.jsonl"
    )
    print(f"\n=== Starting {strategy} (ingest + eval, {EMBEDDING_MODEL}) -> {output_file} ===", flush=True)

    cmd = [
        sys.executable, "-m", "geograph.runners.run_vector_chunking",
        "--strategy", strategy, "--ingest",
        "--embedding-model", EMBEDDING_MODEL,
        "--collection-suffix", COLLECTION_SUFFIX,
        "--embed-batch-size", str(EMBED_BATCH_SIZE),
        "--offset", "0", "--limit", str(NUM_QUESTIONS),
        "--output", output_file,
        "--delay", "0",
        "--evaluate",
    ]
    subprocess.run(cmd, check=True)
    print(f"=== Finished {strategy} ===", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "strategy",
        nargs="?",
        choices=ALL_STRATEGIES,
        default=None,
        help="Run only this strategy (omit to run both sequentially).",
    )
    args = parser.parse_args()

    strategies = [args.strategy] if args.strategy else ALL_STRATEGIES
    for strategy in strategies:
        run_strategy(strategy)
    print("\nAll done.")


if __name__ == "__main__":
    main()
