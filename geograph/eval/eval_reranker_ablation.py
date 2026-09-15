from __future__ import annotations

import json
import pathlib
import time
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

BENCHMARK_PATH = pathlib.Path(__file__).parent.parent.parent / "testset" / "geography_777_benchmark.jsonl"
OUTPUT_PATH = pathlib.Path(__file__).parent.parent.parent / "testset" / "kg_rag" / "reranker_ablation.json"
N_QUESTIONS = 40
FINAL_TOP_K = 60
FALLBACK_TOP_K = 15

_STOP_WORDS = {
    "в", "на", "са", "кои", "е", "с", "и", "или", "до", "от", "за", "под", "над", "през",
}


@dataclass
class ArmResult:
    kept_60: List[str]
    seconds: float


@dataclass
class QuestionResult:
    qid: int
    oracle: bool
    ce: bool = False
    cos: bool = False
    lex: bool = False
    ce15: bool = False
    cos15: bool = False
    lex15: bool = False


def _lex_keywords(query: str) -> set:
    return {
        kw for kw in query.lower().split()
        if kw not in _STOP_WORDS and len(kw) > 2
    }


def _lex_rank(query: str, candidates: List[str], top_k: int) -> List[str]:
    """Rank by deduplicated keyword-overlap score, descending. Mirrors
    `_ReRanker._keyword_bonus`'s scoring (Σ len(kw)*2 over matched keywords),
    but as a standalone ranker rather than a tie-breaking bonus term."""
    keywords = _lex_keywords(query)
    scored = []
    for c in candidates:
        c_lower = c.lower()
        score = sum(len(kw) * 2 for kw in keywords if kw in c_lower)
        scored.append(score)
    order = np.argsort(scored)[::-1][:top_k]
    return [candidates[i] for i in order]


def _load_pipeline():
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=pathlib.Path(__file__).parent.parent / ".env")

    from geograph.config import PipelineConfig
    from geograph.pipelines.kg_rag_pipeline import KGRAGPipeline

    config = PipelineConfig.from_constants()
    config.embedding.model_name = "BAAI/bge-m3"
    config.embedding.tokenizer_name = "BAAI/bge-m3"
    config.retrieval.chroma_collection_name += "_bgem3_v2"

    pipeline = KGRAGPipeline(config)
    pipeline.load_graph_only()
    return pipeline


def _load_questions(n: int) -> List[Dict]:
    questions = []
    with open(BENCHMARK_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            questions.append(json.loads(line))
            if len(questions) >= n:
                break
    return questions


def run_ablation() -> None:
    from geograph.llm.prompt_builder import build_mcq_prompt
    from geograph.retrieval.retrieval_recall import recall_proxy

    pipeline = _load_pipeline()
    questions = _load_questions(N_QUESTIONS)

    ce_total_s = 0.0
    cos_total_s = 0.0
    lex_total_s = 0.0

    ce_hits_60 = ce_hits_15 = 0
    cos_hits_60 = cos_hits_15 = 0
    lex_hits_60 = lex_hits_15 = 0
    n_oracle = 0

    per_question: List[QuestionResult] = []

    for i, record in enumerate(questions, 1):
        qid = record.get("qid", i)
        question = record["question"]
        options = record["options"]
        correct_answer = record["correct_answer"]

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

        oracle = recall_proxy(expanded_triples, options, correct_answer)
        n_oracle += int(oracle)

        protected_kept = [t for t in expanded_triples if t in protected_triples]
        prunable = [t for t in expanded_triples if t not in protected_triples]

        coarse_budget = max(pipeline.config.sparql.expansion_coarse_top_k - len(protected_kept), 0)
        coarse_survivors = pipeline._coarse_filter_by_similarity(mcq_prompt, prunable, coarse_budget)

        rerank_budget = max(FINAL_TOP_K - len(protected_kept), 0)

        # Arm CE: current production stage.
        t0 = time.time()
        ranked_indices = pipeline.expansion_reranker.rerank(mcq_prompt, coarse_survivors, rerank_budget)
        ce_total_s += time.time() - t0
        ce_kept = protected_kept + [coarse_survivors[i] for i in ranked_indices]

        # Arm COS: cosine-only, no cross-encoder.
        t0 = time.time()
        cos_kept = protected_kept + pipeline._coarse_filter_by_similarity(
            mcq_prompt, coarse_survivors, rerank_budget,
        )
        cos_total_s += time.time() - t0

        # Arm LEX: deduplicated keyword-overlap score.
        t0 = time.time()
        lex_kept = protected_kept + _lex_rank(mcq_prompt, coarse_survivors, rerank_budget)
        lex_total_s += time.time() - t0

        qr = QuestionResult(qid=qid, oracle=oracle)
        if oracle:
            qr.ce = recall_proxy(ce_kept, options, correct_answer)
            qr.cos = recall_proxy(cos_kept, options, correct_answer)
            qr.lex = recall_proxy(lex_kept, options, correct_answer)
            qr.ce15 = recall_proxy(ce_kept[:FALLBACK_TOP_K], options, correct_answer)
            qr.cos15 = recall_proxy(cos_kept[:FALLBACK_TOP_K], options, correct_answer)
            qr.lex15 = recall_proxy(lex_kept[:FALLBACK_TOP_K], options, correct_answer)
            ce_hits_60 += int(qr.ce)
            cos_hits_60 += int(qr.cos)
            lex_hits_60 += int(qr.lex)
            ce_hits_15 += int(qr.ce15)
            cos_hits_15 += int(qr.cos15)
            lex_hits_15 += int(qr.lex15)
        per_question.append(qr)

        print(
            f"[{i}/{len(questions)}] qid={qid} oracle={oracle} "
            f"CE={qr.ce}/{qr.ce15} COS={qr.cos}/{qr.cos15} LEX={qr.lex}/{qr.lex15}"
        )

    def pct(hits: int) -> float:
        return 100.0 * hits / n_oracle if n_oracle else 0.0

    print(f"\nOracle-True subset: {n_oracle}/{len(questions)} questions")
    print(f"{'Arm':<6}{'survival@60':<14}{'survival@15':<14}{'s/q':<10}")
    print(f"{'CE':<6}{pct(ce_hits_60):<14.1f}{pct(ce_hits_15):<14.1f}{ce_total_s / len(questions):<10.2f}")
    print(f"{'COS':<6}{pct(cos_hits_60):<14.1f}{pct(cos_hits_15):<14.1f}{cos_total_s / len(questions):<10.2f}")
    print(f"{'LEX':<6}{pct(lex_hits_60):<14.1f}{pct(lex_hits_15):<14.1f}{lex_total_s / len(questions):<10.2f}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    output = {
        "n_questions": len(questions),
        "n_oracle": n_oracle,
        "arms": {
            "CE": {
                "survival_60_pct": pct(ce_hits_60), "survival_15_pct": pct(ce_hits_15),
                "seconds_per_question": ce_total_s / len(questions),
            },
            "COS": {
                "survival_60_pct": pct(cos_hits_60), "survival_15_pct": pct(cos_hits_15),
                "seconds_per_question": cos_total_s / len(questions),
            },
            "LEX": {
                "survival_60_pct": pct(lex_hits_60), "survival_15_pct": pct(lex_hits_15),
                "seconds_per_question": lex_total_s / len(questions),
            },
        },
        "per_question": [
            {
                "qid": qr.qid, "oracle": qr.oracle,
                "ce_60": qr.ce, "cos_60": qr.cos, "lex_60": qr.lex,
                "ce_15": qr.ce15, "cos_15": qr.cos15, "lex_15": qr.lex15,
            }
            for qr in per_question
        ],
    }
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\nResults written to {OUTPUT_PATH}")


if __name__ == "__main__":
    run_ablation()
