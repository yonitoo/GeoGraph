import argparse
import json
import logging
from collections import Counter, defaultdict
from typing import Dict, List, Set

from rdflib import BNode, Graph, Literal, RDF, RDFS, OWL, URIRef

from geograph.config import PipelineConfig
from geograph.ontology.entity_linker import EntityLinker
from geograph.ontology.rdf_processor import RDFProcessor

logger = logging.getLogger(__name__)

DEFAULT_BENCHMARK = "testset/geography_777_benchmark.jsonl"
DEFAULT_SOURCE_TTL = "ontologies/huge_ontology_v2.ttl"
DEFAULT_OUTPUT = "ontologies/huge_ontology_v3.ttl"
HUB_DEGREE_CAP = 50
MAX_LINKED_PER_QUESTION = 16
DESCRIPTION_CHAR_CAP = 300

NOISE_TYPES = (
    "Село", "Музей", "КултуренОбект", "Събитие", "ЕтнографскаОбласт",
    "Квартал", "Хижа", "Организация", "КандидатЗаОбщинскиСъветник",
    "ЖивотинскиВид", "РастителенВид", "Крепост",
)


def _load_benchmark(path: str) -> List[Dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _question_text(item: Dict) -> str:
    option_texts = [opt.get("text", "") for opt in item.get("options", [])]
    return " ".join([item["question"], *option_texts])


_STEM_MIN = 4
_STEM_TRIM = 3


def _stem(word: str) -> str:
    return word[: max(_STEM_MIN, len(word) - _STEM_TRIM)]


def _word_prefixes(text: str) -> Set[str]:
    """All prefixes (length ≥ _STEM_MIN) of every word in `text`."""
    prefixes: Set[str] = set()
    for word in text.lower().split():
        word = word.strip('.,;:!?()"„“–-')
        for end in range(_STEM_MIN, len(word) + 1):
            prefixes.add(word[:end])
    return prefixes


def _fuzzy_label_match(label: str, question_prefixes: Set[str]) -> bool:
    """True if the label's content words appear in the question text modulo
    Bulgarian suffix inflection (prefix-stem matching)."""
    words = [w.strip('.,;:!?()"„“–-') for w in label.split()]
    content = [w for w in words if len(w) >= _STEM_MIN]
    if not content:
        return False
    hits = [_stem(w) in question_prefixes for w in content]
    n = len(content)
    if all(hits):
        return True
    if n >= 3 and sum(hits) >= n - 1:
        return True
    return n >= 3 and hits[0] and hits[1]


def collect_seed_entities(graph: Graph, questions: List[Dict]) -> Set[URIRef]:
    """Union of KG entities linked in any question or answer option."""
    linker = EntityLinker.from_graph(graph)
    seeds: Set[URIRef] = set()
    fuzzy_hits = 0
    for item in questions:
        text = _question_text(item)
        seeds.update(linker.link(text, max_entities=MAX_LINKED_PER_QUESTION))
        prefixes = _word_prefixes(text)
        for label, uri in linker.label_index.items():
            if uri not in seeds and _fuzzy_label_match(label, prefixes):
                seeds.add(uri)
                fuzzy_hits += 1
    logger.info(
        "Linked %d distinct seed entities across %d questions (%d via fuzzy stems).",
        len(seeds), len(questions), fuzzy_hits,
    )
    return seeds


def _build_adjacency(graph: Graph, namespace: str) -> Dict[URIRef, Set[URIRef]]:
    """Undirected entity adjacency over object-property edges."""
    adjacency: Dict[URIRef, Set[URIRef]] = defaultdict(set)
    for s, p, o in graph:
        if p == RDF.type:
            continue
        if isinstance(s, URIRef) and isinstance(o, URIRef) and str(o).startswith(namespace):
            adjacency[s].add(o)
            adjacency[o].add(s)
    return adjacency


def compute_kept_nodes(
    graph: Graph, seeds: Set[URIRef], namespace: str, hub_degree_cap: int,
) -> Dict[URIRef, int]:
    """BFS from the seeds; returns node → hop distance (0 or 1)."""
    adjacency = _build_adjacency(graph, namespace)
    hops: Dict[URIRef, int] = {seed: 0 for seed in seeds}

    for seed in seeds:
        for neighbor in adjacency.get(seed, ()):
            hops.setdefault(neighbor, 1)

    # Second hop: expand only from non-hub hop-1 nodes — hubs are kept
    # but don't gate open their entire neighborhood.
    frontier = [n for n, h in hops.items() if h == 1 and len(adjacency.get(n, ())) <= hub_degree_cap]
    for node in frontier:
        for neighbor in adjacency.get(node, ()):
            hops.setdefault(neighbor, 2)
    return hops


def _schema_subjects(graph: Graph) -> Set[URIRef]:
    subjects: Set[URIRef] = set()
    for cls in (OWL.Class, RDFS.Class, OWL.ObjectProperty, OWL.DatatypeProperty, RDF.Property):
        subjects.update(s for s in graph.subjects(RDF.type, cls) if isinstance(s, URIRef))
    subjects.update(o for o in graph.objects(None, RDF.type) if isinstance(o, URIRef))
    return subjects


def build_subgraph(
    processor: RDFProcessor,
    seeds: Set[URIRef],
    namespace: str,
    hub_degree_cap: int = HUB_DEGREE_CAP,
    description_char_cap: int = DESCRIPTION_CHAR_CAP,
) -> Graph:
    graph = processor.graph
    hops = compute_kept_nodes(graph, seeds, namespace, hub_degree_cap)
    schema = _schema_subjects(graph)
    describe_uri = URIRef(namespace + "имаОписание")

    noise_type_uris = {URIRef(namespace + t) for t in NOISE_TYPES}
    noise_subjects = {
        s for t in noise_type_uris for s in graph.subjects(RDF.type, t)
        if isinstance(s, URIRef) and s not in seeds
    }

    out = Graph()
    for prefix, ns in graph.namespace_manager.namespaces():
        out.bind(prefix, ns)

    dropped_blobs = 0
    kept_bnodes: Set[BNode] = set()

    for s, p, o in graph:
        if isinstance(s, BNode):
            continue
        if s in schema:
            out.add((s, p, o))
            if isinstance(o, BNode):
                kept_bnodes.add(o)
            continue
        s_hop = hops.get(s)
        if s_hop is None or s_hop > 1 or s in noise_subjects:
            if (s_hop is not None or s in noise_subjects) and p == RDFS.label:
                out.add((s, p, o))
            continue
        if p == describe_uri and isinstance(o, Literal) and s not in seeds \
                and len(str(o)) > description_char_cap:
            dropped_blobs += 1
            continue
        if isinstance(o, URIRef) and str(o).startswith(namespace) \
                and o not in hops and o not in schema and p != RDF.type:
            continue
        out.add((s, p, o))
        if isinstance(o, BNode):
            kept_bnodes.add(o)

    worklist = list(kept_bnodes)
    closed: Set[BNode] = set()
    while worklist:
        bnode = worklist.pop()
        if bnode in closed:
            continue
        closed.add(bnode)
        for p, o in graph.predicate_objects(bnode):
            out.add((bnode, p, o))
            if isinstance(o, BNode):
                worklist.append(o)

    logger.info(
        "Subgraph: %d → %d triples | seeds=%d, hop≤1 nodes=%d, schema subjects=%d, "
        "noise-typed subjects reduced to labels=%d, long descriptions dropped=%d",
        len(graph), len(out), len(seeds),
        sum(1 for h in hops.values() if h <= 1), len(schema),
        len(noise_subjects), dropped_blobs,
    )
    return out


def report_type_census(graph: Graph, label: str) -> Counter:
    census = Counter(
        str(o).rsplit("#", 1)[-1]
        for o in graph.objects(None, RDF.type)
        if isinstance(o, URIRef)
    )
    logger.info("%s: %d typed instances across %d types.", label, sum(census.values()), len(census))
    return census


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s — %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ttl", default=DEFAULT_SOURCE_TTL)
    parser.add_argument("--benchmark", default=DEFAULT_BENCHMARK)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--hub-degree-cap", type=int, default=HUB_DEGREE_CAP)
    parser.add_argument("--description-char-cap", type=int, default=DESCRIPTION_CHAR_CAP)
    args = parser.parse_args()

    config = PipelineConfig.from_constants()
    processor = RDFProcessor.load(args.ttl, config.ontology)
    namespace = str(config.ontology.namespace)

    questions = _load_benchmark(args.benchmark)
    seeds = collect_seed_entities(processor.graph, questions)

    before = report_type_census(processor.graph, "Before")
    subgraph = build_subgraph(
        processor, seeds, namespace,
        hub_degree_cap=args.hub_degree_cap,
        description_char_cap=args.description_char_cap,
    )
    after = report_type_census(subgraph, "After")

    print(f"\n{'type':<40}{'before':>8}{'after':>8}")
    for t, n in before.most_common(40):
        print(f"{t:<40}{n:>8}{after.get(t, 0):>8}")

    subgraph.serialize(destination=args.output, format="turtle")
    logger.info("Wrote %s (%d triples).", args.output, len(subgraph))


if __name__ == "__main__":
    main()
