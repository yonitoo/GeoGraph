import argparse
import collections
import datetime
import json
import logging
import pathlib
import random
import sys
import time
from typing import Dict, List, Optional

from geograph.pipelines.chunking_rag_pipeline import ChunkingRAGPipeline, STRATEGY_REGISTRY, get_strategy
from geograph.config import PipelineConfig
from geograph.llm.llm_client import extract_mcq_letter
from geograph.retrieval.retrieval_recall import recall_proxy

logger = logging.getLogger(__name__)

DEFAULT_INPUT = "testset/geography_777_benchmark.jsonl"
DEFAULT_OUTPUT_ROOT = "testset/vector_chunking"
BATCH_SIZE = 100
PILOT_SEED = 42

ALL_STRATEGIES = ["fixed256", "structure", "parentchild"]


def _answer_key(strategy: str) -> str:
    return f"vector_{strategy}_answer"


def _output_dir(strategy: str) -> str:
    return f"{DEFAULT_OUTPUT_ROOT}/{strategy}"


def _merged_path(strategy: str) -> str:
    return f"{DEFAULT_OUTPUT_ROOT}/{strategy}_output.jsonl"


def _auto_output(strategy: str, offset: int, end: int) -> str:
    d = pathlib.Path(_output_dir(strategy))
    d.mkdir(parents=True, exist_ok=True)
    return str(d / f"batch_{offset + 1:04d}_{end:04d}.jsonl")


def _setup_logging(log_dir: str, tag: str = "") -> None:
    pathlib.Path(log_dir).mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%H_%M_%S-%d_%m_%y")
    suffix = f"_{tag}" if tag else ""
    log_file = f"{log_dir}/vector_chunking{suffix}_{stamp}.log"
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


def _load_benchmark(path: str) -> List[Dict]:
    items = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def _load_existing(output_file: str) -> Dict[str, Dict]:
    existing: Dict[str, Dict] = {}
    p = pathlib.Path(output_file)
    if not p.exists():
        return existing
    with p.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                existing[rec["question"]] = rec
            except (json.JSONDecodeError, KeyError):
                continue
    return existing


def _make_pilot(all_questions: List[Dict], n: int) -> List[Dict]:
    """Stratified sample: preserve the reasoning_bg / new split ratio."""
    by_source: Dict[str, List[Dict]] = collections.defaultdict(list)
    for q in all_questions:
        by_source[q.get("source_dataset", "unknown")].append(q)

    total = len(all_questions)
    sampled: List[Dict] = []
    rng = random.Random(PILOT_SEED)
    for src, qs in by_source.items():
        quota = max(1, round(len(qs) / total * n))
        sampled.extend(rng.sample(qs, min(quota, len(qs))))

    rng.shuffle(sampled)
    if len(sampled) > n:
        sampled = sampled[:n]
    elif len(sampled) < n:
        remaining = [q for q in all_questions if q not in sampled]
        sampled.extend(rng.sample(remaining, min(n - len(sampled), len(remaining))))

    return sampled


def run_strategy(
    strategy: str,
    questions: List[Dict],
    output_file: str,
    offset: int,
    delay: float,
    run_eval: bool,
    do_ingest: bool,
    use_reranker: bool,
    embedding_model: Optional[str] = None,
    collection_suffix: str = "",
    embed_batch_size: int = 32,
) -> None:
    try:
        from dotenv import load_dotenv
        load_dotenv(dotenv_path=pathlib.Path(__file__).parent.parent / ".env")
    except ImportError:
        pass

    config = PipelineConfig.from_constants()
    if embedding_model:
        config.embedding.model_name = embedding_model
        config.embedding.tokenizer_name = embedding_model

    pipeline = ChunkingRAGPipeline(
        config=config,
        strategy_name=strategy,
        use_reranker=use_reranker,
        embed_batch_size=embed_batch_size,
    )
    if collection_suffix:
        pipeline.strategy.collection_name += collection_suffix

    if do_ingest:
        logger.info("Ingesting strategy '%s' — this may take several minutes...", strategy)
        pipeline.ingest()
        logger.info("Ingestion complete.")
    else:
        logger.info("Loading existing collection for strategy '%s'...", strategy)
        pipeline.load_existing()
        logger.info("Pipeline ready.")

    answer_key = _answer_key(strategy)
    existing = _load_existing(output_file)
    logger.info(
        "Strategy '%s' | %d questions | %d already answered",
        strategy, len(questions), len(existing),
    )

    pathlib.Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    new_count = 0

    with open(output_file, "a", encoding="utf-8") as out_fh:
        for i, exam in enumerate(questions):
            global_idx = offset + i + 1
            question = exam["question"]
            options = exam.get("options", [])
            correct = exam.get("correct_answer", "?")

            if question in existing:
                rec = existing[question]
                if answer_key in rec and rec[answer_key] not in ("", None):
                    logger.debug("Q%d already answered (%s), skipping", global_idx, rec[answer_key])
                    continue

            record = existing.get(question, {**exam})

            t0 = time.time()
            try:
                response, context_chunks, timings = pipeline.query(question, options)
                letter = extract_mcq_letter(response)
                if letter is None:
                    letter = "-"
                elapsed = time.time() - t0

                recall = recall_proxy(context_chunks, options, correct)
                logger.info(
                    "Q%04d | pred=%-1s | gold=%s | recall=%s | %.1fs | %s | %s",
                    global_idx, letter, correct, "Y" if recall else "N",
                    elapsed, timings.summary(), question[:55],
                )
                record[answer_key] = letter
                record[f"{answer_key}_recall"] = recall
                record[f"{answer_key}_ctx_n"] = len(context_chunks)
            except Exception as exc:
                logger.warning("Q%d error: %s", global_idx, exc)
                record[answer_key] = "-"
                record[f"{answer_key}_recall"] = False
                record[f"{answer_key}_ctx_n"] = 0

            out_fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            out_fh.flush()
            new_count += 1

            if delay > 0:
                time.sleep(delay)

    logger.info("Done — %d new answers written to %s", new_count, output_file)

    if run_eval:
        _print_eval(output_file, strategy)


def _print_eval(output_file: str, strategy: str) -> None:
    answer_key = _answer_key(strategy)
    records = []
    with open(output_file, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    correct = total = abstained = 0
    recall_hits = recall_total = 0
    ctx_counts: List[int] = []
    by_source: Dict[str, Dict] = collections.defaultdict(lambda: {"correct": 0, "total": 0})

    for rec in records:
        answer = rec.get(answer_key)
        expected = rec.get("correct_answer")
        src = rec.get("source_dataset", "unknown")
        total += 1
        by_source[src]["total"] += 1

        if answer in ("", None, "-"):
            abstained += 1
        elif answer == expected:
            correct += 1
            by_source[src]["correct"] += 1

        recall_key = f"{answer_key}_recall"
        if recall_key in rec:
            recall_total += 1
            if rec[recall_key]:
                recall_hits += 1

        ctx_n = rec.get(f"{answer_key}_ctx_n", 0)
        if ctx_n:
            ctx_counts.append(ctx_n)

    avg_ctx = sum(ctx_counts) / len(ctx_counts) if ctx_counts else 0.0
    recall_pct = (recall_hits / recall_total * 100) if recall_total else 0.0

    print(f"VECTOR CHUNKING — strategy: {strategy}\n")
    print(f"Accuracy: {correct}/{total} = {correct/total*100:.1f}%")
    print(f"Abstention: {abstained}/{total} = {abstained/total*100:.1f}%")
    print(f"Recall-proxy: {recall_hits}/{recall_total} = {recall_pct:.1f}%")
    print(f"Avg ctx chunks: {avg_ctx:.1f}")
    for src, d in sorted(by_source.items()):
        t, c = d["total"], d["correct"]
        print(f"  [{src}]  {c}/{t} = {c/t*100:.1f}%\n")


def merge_outputs(strategy: str, run_eval: bool = False) -> None:
    batch_dir = pathlib.Path(_output_dir(strategy))
    files = sorted(batch_dir.glob("batch_*.jsonl"))
    if not files:
        print(f"No batch files in {batch_dir}")
        return

    records: List[Dict] = []
    for f in files:
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    records.append(json.loads(line))

    questions = [r["question"] for r in records]
    dupes = len(questions) - len(set(questions))
    print(f"Strategy '{strategy}': {len(files)} files | {len(records)} records | {dupes} duplicates")

    merged = _merged_path(strategy)
    with open(merged, "w", encoding="utf-8") as out:
        for rec in records:
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"Written → {merged}")

    if run_eval:
        _print_eval(merged, strategy)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Vector RAG chunking-strategy experiment runner"
    )
    parser.add_argument(
        "--strategy",
        default="structure",
        help=(
            "Chunking strategy: fixed256 | structure | parentchild | fixed500 | all. "
            "'all' runs all three experimental strategies sequentially (useful for pilot)."
        ),
    )
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSONL path (auto-named by range/strategy if omitted)",
    )
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--pilot",
        type=int,
        default=None,
        metavar="N",
        help="Sample N questions (stratified) for a quick pilot run instead of offset/limit",
    )
    parser.add_argument("--delay", type=float, default=1.0)
    parser.add_argument("--log-dir", default="logs")
    parser.add_argument("--ingest", action="store_true")
    parser.add_argument(
        "--no-rerank",
        action="store_true",
        help="Disable cross-encoder reranking (useful as an ablation)",
    )
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--merge-only", action="store_true")
    parser.add_argument(
        "--embedding-model",
        default=None,
        help="Override the embedding model for this run only (e.g. BAAI/bge-m3). "
             "Also overrides the tokenizer to match. Leaves constants.py untouched.",
    )
    parser.add_argument(
        "--collection-suffix",
        default="",
        help="Suffix appended to the strategy's Chroma collection name, so this "
             "run's ingested vectors don't overwrite the default collection "
             "(useful when combined with --embedding-model).",
    )
    parser.add_argument(
        "--embed-batch-size",
        type=int,
        default=32,
        help="Batch size for embedding chunks during --ingest (higher throughput "
             "on GPU/MPS if memory allows; has no effect on retrieval quality).",
    )
    args = parser.parse_args()

    # Ensure strategy registry is populated before we validate names
    _ = get_strategy("fixed256")

    strategies = ALL_STRATEGIES if args.strategy == "all" else [args.strategy]

    # Validate
    for s in strategies:
        if s not in STRATEGY_REGISTRY:
            print(f"Unknown strategy '{s}'. Available: {list(STRATEGY_REGISTRY.keys())}")
            sys.exit(1)

    if args.merge_only:
        for s in strategies:
            merge_outputs(s, run_eval=True)
        return

    all_questions = _load_benchmark(args.input)

    if args.pilot:
        questions = _make_pilot(all_questions, args.pilot)
        logger.info("Pilot sample: %d questions (stratified)", len(questions))
        offset = 0
        end = len(questions)
    else:
        end = min(args.offset + args.limit, len(all_questions)) if args.limit else len(all_questions)
        questions = all_questions[args.offset:end]
        offset = args.offset

    for strategy in strategies:
        tag = f"{strategy}_{offset + 1}_{end}"
        _setup_logging(args.log_dir, tag)

        if args.output and len(strategies) == 1:
            output_file = args.output
        else:
            output_file = _auto_output(strategy, offset, end)

        logger.info(
            "=== Strategy: %s | Questions: %d | Output: %s ===",
            strategy, len(questions), output_file,
        )

        run_strategy(
            strategy=strategy,
            questions=questions,
            output_file=output_file,
            offset=offset,
            delay=args.delay,
            run_eval=args.evaluate,
            do_ingest=args.ingest,
            use_reranker=not args.no_rerank,
            embedding_model=args.embedding_model,
            collection_suffix=args.collection_suffix,
            embed_batch_size=args.embed_batch_size,
        )

    if args.evaluate and len(strategies) > 1:
        print("\n=== CROSS-STRATEGY COMPARISON ===")
        for s in strategies:
            merged = _merged_path(s)
            if pathlib.Path(merged).exists():
                _print_eval(merged, s)


if __name__ == "__main__":
    main()
