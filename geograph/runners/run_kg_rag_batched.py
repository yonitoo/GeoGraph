import argparse
import collections
import datetime
import json
import logging
import os
import pathlib
import time
from typing import Dict, List, Optional

from geograph.config import PipelineConfig
from geograph.pipelines.kg_rag_pipeline import KGRAGPipeline, QueryDiagnostics
from geograph.llm.llm_client import extract_mcq_letter
from geograph.llm.prompt_builder import build_mcq_prompt
from geograph.retrieval.retrieval_recall import recall_proxy

logger = logging.getLogger(__name__)

DEFAULT_INPUT = "testset/geography_777_benchmark.jsonl"
DEFAULT_OUTPUT_DIR = "testset/kg_rag"
MERGED_OUTPUT = "testset/kg_rag_output.jsonl"
ANSWER_KEY = "bggpt_kg_rag_answer"


def _setup_logging(log_dir: str, tag: str = "") -> None:
    pathlib.Path(log_dir).mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%H_%M_%S-%d_%m_%y")
    suffix = f"_{tag}" if tag else ""
    log_file = os.path.join(log_dir, f"kg_rag{suffix}_{stamp}.log")
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
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def _load_existing(output_file: str) -> Dict[str, Dict]:
    existing: Dict[str, Dict] = {}
    p = pathlib.Path(output_file)
    if not p.exists():
        return existing
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


def _diagnostics_to_kg_diag(diagnostics: QueryDiagnostics, options: List[Dict], correct: str, timings) -> Dict:
    """Reduce full-text diagnostics to compact, JSONL-safe counts and recall flags."""
    return {
        "n_seeds": diagnostics.n_seeds,
        "n_linked_entities": diagnostics.n_linked_entities,
        "n_expanded": diagnostics.n_expanded,
        "n_protected": diagnostics.n_protected,
        "n_reranked": diagnostics.n_reranked,
        "n_prefiltered": diagnostics.n_prefiltered,
        "n_filtered": diagnostics.n_filtered,
        "filter_llm_used": diagnostics.filter_llm_used,
        "used_fallback_context": diagnostics.used_fallback_context,
        "seed_recall": recall_proxy(diagnostics.seed_docs, options, correct),
        "expanded_recall": recall_proxy(diagnostics.expanded_triples, options, correct),
        "timings_s": {
            "retrieval": round(timings.retrieval_s, 2),
            "expansion": round(timings.expansion_s, 2),
            "rerank": round(timings.rerank_s, 2),
            "filtering": round(timings.filtering_s, 2),
            "generation": round(timings.generation_s, 2),
        },
    }


def run_kg_rag(
    input_file: str,
    output_file: str,
    offset: int,
    limit: Optional[int],
    delay: float,
    log_dir: str,
    run_eval: bool,
    do_ingest: bool,
    embedding_model: Optional[str] = None,
    collection_suffix: str = "",
    embed_batch_size: int = 32,
    empty_context_policy: str = "fallback",
    reranker: str = "cross_encoder",
    no_reasoning_filter: bool = False,
) -> None:
    try:
        from dotenv import load_dotenv
        load_dotenv(dotenv_path=pathlib.Path(__file__).parent.parent / ".env")
    except ImportError:
        pass

    all_questions = _load_benchmark(input_file)
    end = offset + limit if limit else len(all_questions)
    end = min(end, len(all_questions))
    questions = all_questions[offset:end]

    tag = f"{offset+1}_{end}"
    _setup_logging(log_dir, tag)

    config = PipelineConfig.from_constants()
    if embedding_model:
        # Override just for this run — leaves constants.py untouched.
        config.embedding.model_name = embedding_model
        config.embedding.tokenizer_name = embedding_model
    if collection_suffix:
        # Keep this run's ingested vectors in a separate Chroma collection so
        # they don't clobber the default (differently-embedded) collection.
        config.retrieval.chroma_collection_name += collection_suffix
    if no_reasoning_filter:
        config.reasoning.enabled = False
        logger.info("Reasoning filter disabled — using pre-filtered triples directly (ablation).")

    pipeline = KGRAGPipeline(
        config, embed_batch_size=embed_batch_size, empty_context_policy=empty_context_policy,
    )
    if reranker == "lexical":
        from geograph.retrieval.reranker import LexicalReranker
        pipeline.expansion_reranker = LexicalReranker()
        logger.info("Using LexicalReranker for expansion-pruning stage (ablation).")
    elif reranker == "stem_lexical":
        from geograph.retrieval.reranker import StemLexicalReranker
        pipeline.expansion_reranker = StemLexicalReranker()
        logger.info("Using StemLexicalReranker for expansion-pruning stage (ablation).")
    if do_ingest:
        logger.info("Ingesting huge KG into Chroma — this may take a while...")
        pipeline.ingest()
        logger.info("Ingestion complete.")
    else:
        logger.info("Loading existing Chroma collection...")
        pipeline.load_graph_only()
        logger.info("Pipeline ready.")

    existing = _load_existing(output_file)
    logger.info(
        "Batch Q%d–Q%d (%d questions) | Already answered: %d",
        offset + 1, end, len(questions), len(existing),
    )

    pathlib.Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    new_count = 0

    # Append mode: already-answered records stay in the file untouched;
    # each new answer is flushed immediately so a crash loses at most one question.
    with open(output_file, "a", encoding="utf-8") as out_fh:
        for i, exam in enumerate(questions):
            global_idx = offset + i + 1
            question = exam["question"]
            options = exam.get("options", [])
            correct = exam.get("correct_answer", "?")

            if question in existing:
                rec = existing[question]
                if ANSWER_KEY in rec and rec[ANSWER_KEY] not in ("", None):
                    logger.debug("Q%d already answered (%s), skipping", global_idx, rec[ANSWER_KEY])
                    continue

            record = existing.get(question, {**exam})
            mcq_prompt = build_mcq_prompt(question, options)

            t0 = time.time()
            try:
                response, timings, diagnostics = pipeline.query(mcq_prompt)
                letter = extract_mcq_letter(response)
                if letter is None:
                    letter = "-"
                elapsed = time.time() - t0

                answer_recall = recall_proxy(diagnostics.context_triples, options, correct)
                logger.info(
                    "Q%04d | pred=%-1s | gold=%s | recall=%s | %.1fs | %s | %s",
                    global_idx, letter, correct, "Y" if answer_recall else "N",
                    elapsed, timings.summary(), question[:55],
                )
                record[ANSWER_KEY] = letter
                record[f"{ANSWER_KEY}_recall"] = answer_recall
                record[f"{ANSWER_KEY}_ctx_n"] = len(diagnostics.context_triples)
                record["kg_diag"] = _diagnostics_to_kg_diag(diagnostics, options, correct, timings)
            except Exception as exc:
                logger.warning("Q%d error: %s", global_idx, exc)
                record[ANSWER_KEY] = "-"
                record[f"{ANSWER_KEY}_recall"] = False
                record[f"{ANSWER_KEY}_ctx_n"] = 0
                record["kg_diag"] = {"error": str(exc)}

            out_fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            out_fh.flush()
            new_count += 1

            if delay > 0:
                time.sleep(delay)

    logger.info("Done — %d new answers written to %s", new_count, output_file)
    if pipeline.embedding_cache is not None:
        logger.info("Embedding cache hit rate: %.1f%%", pipeline.embedding_cache.hit_rate * 100)

    if run_eval:
        _print_eval(output_file)


def _print_eval(output_file: str) -> None:
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
        answer = rec.get(ANSWER_KEY)
        expected = rec.get("correct_answer")
        src = rec.get("source_dataset", "unknown")
        total += 1
        by_source[src]["total"] += 1

        if answer in ("", None, "-"):
            abstained += 1
        elif answer == expected:
            correct += 1
            by_source[src]["correct"] += 1

        recall_key = f"{ANSWER_KEY}_recall"
        if recall_key in rec:
            recall_total += 1
            if rec[recall_key]:
                recall_hits += 1

        ctx_n = rec.get(f"{ANSWER_KEY}_ctx_n", 0)
        if ctx_n:
            ctx_counts.append(ctx_n)

    avg_ctx = sum(ctx_counts) / len(ctx_counts) if ctx_counts else 0.0
    recall_pct = (recall_hits / recall_total * 100) if recall_total else 0.0

    print("KG-RAG EVALUATION\n")
    print(f"Accuracy: {correct}/{total} = {correct/total*100:.1f}%")
    print(f"Abstained: {abstained}/{total} = {abstained/total*100:.1f}%")
    print(f"Recall-proxy: {recall_hits}/{recall_total} = {recall_pct:.1f}%")
    print(f"Avg ctx triples: {avg_ctx:.1f}")
    for src, d in sorted(by_source.items()):
        t, c = d["total"], d["correct"]
        print(f"  [{src}]  {c}/{t} = {c/t*100:.1f}%")


def merge_outputs(run_eval: bool = False) -> None:
    batch_dir = pathlib.Path(DEFAULT_OUTPUT_DIR)
    files = sorted(batch_dir.glob("batch_*.jsonl"))
    if not files:
        print(f"No batch files found in {batch_dir}")
        return

    records = []
    for f in files:
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    records.append(json.loads(line))

    questions = [r["question"] for r in records]
    dupes = len(questions) - len(set(questions))
    print(f"Files: {len(files)}  |  Records: {len(records)}  |  Duplicates: {dupes}")

    with open(MERGED_OUTPUT, "w", encoding="utf-8") as out:
        for rec in records:
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Written → {MERGED_OUTPUT}")
    if run_eval:
        _print_eval(MERGED_OUTPUT)


def main():
    parser = argparse.ArgumentParser(description="KG-RAG batched evaluation on 777 questions")
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output", default=None,
                        help="Output JSONL file (auto-named by range if omitted)")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--delay", type=float, default=1.0,
                        help="Seconds to sleep between questions (default: 1.0)")
    parser.add_argument("--log-dir", default="logs")
    parser.add_argument("--evaluate", action="store_true",
                        help="Print accuracy after the batch completes")
    parser.add_argument("--ingest", action="store_true",
                        help="Ingest the huge KG into Chroma before querying (one-time)")
    parser.add_argument("--merge-only", action="store_true",
                        help="Merge existing batch outputs and evaluate")
    parser.add_argument(
        "--embedding-model",
        default=None,
        help="Override the embedding model for this run only (e.g. BAAI/bge-m3). "
             "Also overrides the tokenizer to match. Leaves constants.py untouched.",
    )
    parser.add_argument(
        "--collection-suffix",
        default="",
        help="Suffix appended to the Chroma collection name, so this run's "
             "ingested vectors don't overwrite the default collection "
             "(useful when combined with --embedding-model).",
    )
    parser.add_argument(
        "--embed-batch-size",
        type=int,
        default=32,
        help="Batch size for embedding triples during --ingest (higher throughput "
             "on GPU/MPS if memory allows; has no effect on retrieval quality).",
    )
    parser.add_argument(
        "--empty-context-policy",
        choices=["fallback", "abstain"],
        default="fallback",
        help="What to do when the reasoning filter returns no triples: 'fallback' "
             "(default) uses the top reranked triples as context; 'abstain' skips "
             "generation entirely and answers '-'.",
    )
    parser.add_argument(
        "--reranker",
        choices=["cross_encoder", "lexical", "stem_lexical"],
        default="cross_encoder",
        help="Expansion-pruning reranker: 'cross_encoder' (default, production), "
             "'lexical' (keyword-overlap heuristic), or 'stem_lexical' (keyword "
             "overlap + Bulgarian suffix-stem matching).",
    )
    parser.add_argument(
        "--no-reasoning-filter",
        action="store_true",
        help="Skip the o4-mini reasoning-selector call entirely and pass the "
             "heuristically pre-filtered triples straight to generation "
             "(ablation — no OpenAI cost for this stage).",
    )
    args = parser.parse_args()

    if args.merge_only:
        merge_outputs(run_eval=True)
        return

    total = sum(1 for _ in open(args.input, encoding="utf-8") if _.strip())
    end = min(args.offset + args.limit, total) if args.limit else total

    output_file = args.output or _auto_output(DEFAULT_OUTPUT_DIR, args.offset, end)

    run_kg_rag(
        input_file=args.input,
        output_file=output_file,
        offset=args.offset,
        limit=args.limit,
        delay=args.delay,
        log_dir=args.log_dir,
        run_eval=args.evaluate,
        do_ingest=args.ingest,
        embedding_model=args.embedding_model,
        collection_suffix=args.collection_suffix,
        embed_batch_size=args.embed_batch_size,
        no_reasoning_filter=args.no_reasoning_filter,
        empty_context_policy=args.empty_context_policy,
        reranker=args.reranker,
    )


if __name__ == "__main__":
    main()
