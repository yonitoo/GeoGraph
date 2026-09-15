import argparse
import datetime
import logging
import os
import pathlib
import time

from jsonlines import jsonlines

from geograph.config import PipelineConfig
from geograph.eval.evaluator import evaluate
from geograph.pipelines.kg_rag_pipeline import KGRAGPipeline
from geograph.llm.prompt_builder import build_mcq_prompt
from geograph.pipelines.vector_rag_pipeline import VectorRAGPipeline

logger = logging.getLogger(__name__)


def setup_logging(log_dir: str) -> str:
    """Configure logging to both file and console."""
    pathlib.Path(log_dir).mkdir(parents=True, exist_ok=True)
    log_file = os.path.join(
        log_dir,
        f'experiment_{datetime.datetime.now().strftime("%H_%M_%S-%d_%m_%y")}.log',
    )

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        filename=log_file,
        filemode="w",
    )

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter("%(levelname)s - %(message)s"))
    logging.getLogger("").addHandler(console)

    logger.info("Logging to %s", log_file)
    return log_file


def run_kg_rag(config: PipelineConfig, input_file: str, output_file: str, run_ingestion: bool) -> None:
    """Run the KG-RAG experiment pipeline."""
    pipeline = KGRAGPipeline(config)

    if run_ingestion:
        pipeline.ingest()
    else:
        pipeline.load_graph_only()

    logger.info("Running KG-RAG on %s → %s", input_file, output_file)

    with jsonlines.open(input_file) as reader:
        with jsonlines.open(output_file, mode="w") as writer:
            for i, exam in enumerate(reader):
                query = build_mcq_prompt(exam["question"], exam["options"])
                logger.info("Q%d: %s", i + 1, exam["question"][:80])

                # Zero-shot baseline (same LLM, no context)
                t0 = time.time()
                zeroshot_response = pipeline.llm_client.generate(query)
                zeroshot_time = time.time() - t0
                logger.info("Zero-shot (%.2fs): %s", zeroshot_time, zeroshot_response[:50])

                # KG-RAG
                kg_rag_response, timings, _diagnostics = pipeline.query(query)
                logger.info("KG-RAG (%s): %s", timings.summary(), kg_rag_response[:50])

                exam["bggpt_kg_rag_answer"] = kg_rag_response
                writer.write(exam)

    logger.info("KG-RAG experiment complete. Results: %s", output_file)


def run_vector_rag(config: PipelineConfig, input_file: str, output_file: str, run_ingestion: bool) -> None:
    """Run the Vector RAG baseline experiment."""
    pipeline = VectorRAGPipeline(config, wiki_data_dir="data/cleaned_data")

    if run_ingestion:
        pipeline.ingest()
    else:
        pipeline.load_existing()

    logger.info("Running Vector RAG on %s → %s", input_file, output_file)

    with jsonlines.open(input_file) as reader:
        with jsonlines.open(output_file, mode="w") as writer:
            for i, exam in enumerate(reader):
                query = build_mcq_prompt(exam["question"], exam["options"])
                logger.info("Q%d: %s", i + 1, exam["question"][:80])

                response, timings = pipeline.query(query)
                logger.info("Vector RAG (%s): %s", timings.summary(), response[:50])

                exam["bggpt_vector_rag_answer"] = response
                writer.write(exam)

    logger.info("Vector RAG experiment complete. Results: %s", output_file)


def main():
    parser = argparse.ArgumentParser(description="Run KG-RAG / Vector RAG experiments")
    parser.add_argument(
        "--pipeline",
        choices=["kg_rag", "vector_rag", "both"],
        default="kg_rag",
        help="Which pipeline to run (default: kg_rag)",
    )
    parser.add_argument("--input", default="testset/demo.jsonl", help="Input JSONL file")
    parser.add_argument("--output", default=None, help="Output JSONL file (auto-named if not set)")
    parser.add_argument("--no-ingest", action="store_true", help="Skip data ingestion, use existing collection")
    parser.add_argument("--log-dir", default="logs", help="Log directory")
    parser.add_argument("--evaluate", action="store_true", help="Run evaluation after experiment")

    args = parser.parse_args()
    setup_logging(args.log_dir)

    config = PipelineConfig.from_constants()
    run_ingestion = not args.no_ingest

    if args.pipeline in ("kg_rag", "both"):
        output = args.output or f"testset/kg_rag_output.jsonl"
        run_kg_rag(config, args.input, output, run_ingestion)
        if args.evaluate:
            result = evaluate(output, "bggpt_kg_rag_answer", "KG-RAG")
            print(result.summary())

    if args.pipeline in ("vector_rag", "both"):
        output = args.output or f"testset/vector_rag_output.jsonl"
        run_vector_rag(config, args.input, output, run_ingestion)
        if args.evaluate:
            result = evaluate(output, "bggpt_vector_rag_answer", "Vector RAG")
            print(result.summary())


if __name__ == "__main__":
    main()