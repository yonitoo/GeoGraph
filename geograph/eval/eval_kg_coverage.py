from __future__ import annotations

import json
import logging
import pathlib
import re
import time
from dataclasses import asdict, dataclass
from typing import List

logger = logging.getLogger(__name__)

BENCHMARK_PATH = pathlib.Path(__file__).parent.parent.parent / "testset" / "geography_777_benchmark.jsonl"
OUTPUT_PATH = pathlib.Path(__file__).parent.parent.parent / "testset" / "kg_rag" / "coverage_777_v3.json"
FINAL_TOP_K = 60
FALLBACK_TOP_K = 15

_OPTION_LABEL_PREFIX = re.compile(r"^[А-ГA-D][\)\.]\s*")

# World geography is out of scope for a Bulgaria-only KG (user decision);
# these questions stay in the hybrid-fallback bucket instead of being curated.
_WORLD_GEO_KEYWORDS = [
    "индия", "япония", "китай", "бразилия", "сащ", "африк", "азия", "австрал",
    "американ", "европ", "франция", "германия", "русия", "иран", "египет",
    "аржентина", "канада", "мексико", "нигерия", "конго", "амазонка", "нил",
    "хималаи", "алпи", "анди", "сахара", "океан", "екватор", "нафта",
]
_ECON_SOCIAL_KEYWORDS = [
    "стопанство", "промишленост", "въдство", "земеделие", "отглежда",
    "царевица", "слънчоглед", "памук", "цвекло", "тютюн", "пшеница", "лоз",
    "овощ", "туризъм", "транспорт", "енергетика", "руди", "въглища",
    "население на българия", "демограф", "раждаемост", "смъртност",
    "миграц", "етнос", "религи", "град", "село", "урбаниз", "регион",
    "ньойски", "договор", "промишлен",
]


def _option_text(option) -> str:
    raw = str(option.get("text", "")) if isinstance(option, dict) else str(option)
    return _OPTION_LABEL_PREFIX.sub("", raw.strip()).strip()


def _classify_domain(question: str, option_texts: List[str]) -> str:
    text = (question + " " + " ".join(option_texts)).lower()
    if any(keyword in text for keyword in _WORLD_GEO_KEYWORDS):
        return "world_geo"
    if any(keyword in text for keyword in _ECON_SOCIAL_KEYWORDS):
        return "econ_social_geo"
    return "bg_physical_geo"


@dataclass
class QuestionCoverage:
    id: str
    qid: int
    domain: str
    expanded_recall: bool
    survival_60: bool
    survival_15: bool
    n_expanded: int


def _load_pipeline():
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=pathlib.Path(__file__).parent.parent / ".env")

    from geograph.config import PipelineConfig
    from geograph.pipelines.kg_rag_pipeline import KGRAGPipeline
    from geograph.retrieval.reranker import LexicalReranker

    config = PipelineConfig.from_constants()
    config.embedding.model_name = "BAAI/bge-m3"
    config.embedding.tokenizer_name = "BAAI/bge-m3"
    config.retrieval.chroma_collection_name += "_bgem3_v3"

    pipeline = KGRAGPipeline(config)
    pipeline.load_graph_only()
    pipeline.expansion_reranker = LexicalReranker()
    return pipeline


def _load_questions() -> List[dict]:
    with open(BENCHMARK_PATH, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def run_coverage_sweep() -> None:
    from geograph.llm.prompt_builder import build_mcq_prompt
    from geograph.retrieval.retrieval_recall import recall_proxy

    pipeline = _load_pipeline()
    questions = _load_questions()

    results: List[QuestionCoverage] = []
    t_start = time.time()

    for i, record in enumerate(questions, 1):
        qid = record.get("qid", i)
        rec_id = record.get("id", str(qid))
        question = record["question"]
        options = record["options"]
        correct_answer = record["correct_answer"]

        option_texts = [_option_text(o) for o in options]
        domain = _classify_domain(question, option_texts)

        mcq_prompt = build_mcq_prompt(question, options)

        retrieved_metadata = pipeline.vector_store.retrieve_and_rerank(
            mcq_prompt, pipeline.embedder, pipeline.config.ontology, pipeline.rdf_processor,
        )
        linked_entities = pipeline.entity_linker.link(mcq_prompt)

        expansion = pipeline.sparql_expander.expand(
            pipeline.rdf_processor, retrieved_metadata, mcq_prompt, pipeline._score_relevance,
            extra_seed_entities=set(linked_entities),
        )
        expanded_triples, protected_triples = expansion.semantic_strings, expansion.protected_strings

        expanded_recall = recall_proxy(expanded_triples, options, correct_answer)

        protected_kept = [t for t in expanded_triples if t in protected_triples]
        prunable = [t for t in expanded_triples if t not in protected_triples]

        coarse_budget = max(pipeline.config.sparql.expansion_coarse_top_k - len(protected_kept), 0)
        coarse_survivors = pipeline._coarse_filter_by_similarity(mcq_prompt, prunable, coarse_budget)

        rerank_budget = max(FINAL_TOP_K - len(protected_kept), 0)
        ranked_indices = pipeline.expansion_reranker.rerank(mcq_prompt, coarse_survivors, rerank_budget)
        kept = protected_kept + [coarse_survivors[i] for i in ranked_indices]

        survival_60 = recall_proxy(kept, options, correct_answer) if expanded_recall else False
        survival_15 = recall_proxy(kept[:FALLBACK_TOP_K], options, correct_answer) if expanded_recall else False

        results.append(QuestionCoverage(
            id=rec_id, qid=qid, domain=domain,
            expanded_recall=expanded_recall, survival_60=survival_60, survival_15=survival_15,
            n_expanded=len(expanded_triples),
        ))

        if i % 20 == 0 or i == len(questions):
            elapsed = time.time() - t_start
            logger.info("[%d/%d] elapsed=%.1fs expanded_recall=%s", i, len(questions), elapsed, expanded_recall)

    n = len(results)
    n_recall = sum(1 for r in results if r.expanded_recall)
    by_domain_total = {}
    by_domain_recall = {}
    for r in results:
        by_domain_total[r.domain] = by_domain_total.get(r.domain, 0) + 1
        by_domain_recall[r.domain] = by_domain_recall.get(r.domain, 0) + int(r.expanded_recall)

    print(f"KG COVERAGE SWEEP: {n} questions. \n Overall expanded_recall (gold evidence in KG expansion): {n_recall}/{n} = {100 * n_recall / n:.1f}% \n Overall gap rate: {n - n_recall}/{n} = {100 * (n - n_recall) / n:.1f}%")
    print("\nBy domain:")
    for domain in sorted(by_domain_total):
        total = by_domain_total[domain]
        recall = by_domain_recall[domain]
        print(f"  {domain:<18}{recall}/{total} = {100 * recall / total:.1f}% covered")

    in_scope = [r for r in results if r.domain != "world_geo"]
    n_in_scope = len(in_scope)
    n_in_scope_recall = sum(1 for r in in_scope if r.expanded_recall)
    print(f"\nIn-scope (non-world-geo) coverage: {n_in_scope_recall}/{n_in_scope} = "
          f"{100 * n_in_scope_recall / n_in_scope:.1f}%")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    output = {
        "n_questions": n,
        "expanded_recall_pct": 100 * n_recall / n,
        "by_domain": {
            domain: {
                "total": by_domain_total[domain],
                "recall": by_domain_recall[domain],
                "recall_pct": 100 * by_domain_recall[domain] / by_domain_total[domain],
            }
            for domain in by_domain_total
        },
        "questions": [asdict(r) for r in results],
    }
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\nResults written to {OUTPUT_PATH}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
    run_coverage_sweep()
