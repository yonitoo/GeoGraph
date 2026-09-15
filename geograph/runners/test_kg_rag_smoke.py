import pathlib
import sys

QUERY = "Кои са десетте най-високи върхове в България?"
HEIGHT_PREDICATE = "надморскаВисочина"

EXPECTED_TOP_10_PEAKS = [
    "Мусала", "Вихрен", "Кутело", "Малка Мусала", "Бански суходол",
    "Иречек", "Полежан", "Малък Полежан", "Каменица", "Баюви дупки",
]

MAX_SANE_EXPANDED = 1500


def _load_pipeline(reranker: str = "cross_encoder"):
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=pathlib.Path(__file__).parent.parent / ".env")

    from geograph.config import PipelineConfig
    from geograph.pipelines.kg_rag_pipeline import KGRAGPipeline

    config = PipelineConfig.from_constants()
    config.embedding.model_name = "BAAI/bge-m3"
    config.embedding.tokenizer_name = "BAAI/bge-m3"
    config.retrieval.chroma_collection_name += "_bgem3_v3"

    pipeline = KGRAGPipeline(config)
    pipeline.load_graph_only()

    if reranker == "lexical":
        from geograph.retrieval.reranker import LexicalReranker
        pipeline.expansion_reranker = LexicalReranker()
    elif reranker == "stem_lexical":
        from geograph.retrieval.reranker import StemLexicalReranker
        pipeline.expansion_reranker = StemLexicalReranker()

    return pipeline


def run_smoke_test(reranker: str = "cross_encoder") -> bool:
    pipeline = _load_pipeline(reranker)
    print(f"Reranker: {reranker}\n")

    print(f"Query: {QUERY}\n")
    response, timings, diagnostics = pipeline.query(QUERY)

    checks = [
        ("retrieval found seeds", diagnostics.n_seeds > 0),
        ("expansion produced candidates", diagnostics.n_expanded > 0),
        (
            f"expansion stayed bounded (<= {MAX_SANE_EXPANDED}, got {diagnostics.n_expanded})",
            diagnostics.n_expanded <= MAX_SANE_EXPANDED,
        ),
        ("reranking kept candidates", diagnostics.n_reranked > 0),
        ("final context is non-empty", len(diagnostics.context_triples) > 0),
    ]

    height_facts = [t for t in diagnostics.context_triples if HEIGHT_PREDICATE in t]
    checks.append((f"context has height facts (got {len(height_facts)})", len(height_facts) > 0))

    missing_peaks = [
        peak for peak in EXPECTED_TOP_10_PEAKS
        if not any(peak in t for t in diagnostics.context_triples)
    ]
    checks.append((
        f"context contains all {len(EXPECTED_TOP_10_PEAKS)} correct peaks"
        + (f" (missing: {missing_peaks})" if missing_peaks else ""),
        not missing_peaks,
    ))

    print("Stage diagnostics:")
    print(f"  n_seeds={diagnostics.n_seeds} n_linked_entities={diagnostics.n_linked_entities} "
          f"n_expanded={diagnostics.n_expanded} n_reranked={diagnostics.n_reranked} "
          f"n_prefiltered={diagnostics.n_prefiltered} n_filtered={diagnostics.n_filtered}")
    print(f"  filter_llm_used={diagnostics.filter_llm_used} used_fallback_context={diagnostics.used_fallback_context}")
    print(f"  timings: {timings.summary()}")
    print(f"\nFinal context ({len(diagnostics.context_triples)} triples):")
    for i, triple in enumerate(diagnostics.context_triples, 1):
        print(f"  {i}. {triple}")
    print(f"\nResponse:\n{response}\n")

    print("Checks:")
    all_passed = True
    for description, passed in checks:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {description}")
        all_passed = all_passed and passed

    return all_passed


if __name__ == "__main__":
    if "--stem-lexical" in sys.argv:
        reranker = "stem_lexical"
    elif "--lexical" in sys.argv:
        reranker = "lexical"
    else:
        reranker = "cross_encoder"
    success = run_smoke_test(reranker)
    print("\nSMOKE TEST " + ("PASSED" if success else "FAILED"))
    sys.exit(0 if success else 1)
