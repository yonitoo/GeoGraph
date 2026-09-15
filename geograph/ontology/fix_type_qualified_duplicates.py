from __future__ import annotations

import argparse
import logging
import re
import shutil
from collections import defaultdict
from typing import Dict, List, Set, Tuple

from rdflib import Graph, RDF, RDFS, URIRef

from geograph.ontology.curate_ontology import MergeReport, _connectivity, _merge_node, _shared_type

logger = logging.getLogger(__name__)

_PAREN_RE = re.compile(r"^(.*?)\s*\(([^)]+)\)$")

_SCHEMA_TYPES = {
    "http://www.w3.org/2002/07/owl#Class",
    "http://www.w3.org/2002/07/owl#ObjectProperty",
    "http://www.w3.org/2002/07/owl#DatatypeProperty",
    "http://www.w3.org/2002/07/owl#SymmetricProperty",
    "http://www.w3.org/2002/07/owl#TransitiveProperty",
    "http://www.w3.org/1999/02/22-rdf-syntax-ns#Property",
    "http://www.w3.org/2000/01/rdf-schema#Class",
}


def _type_words(graph: Graph) -> Set[str]:
    words = set()
    for t in set(graph.objects(None, RDF.type)):
        if isinstance(t, URIRef):
            local = str(t).split("#")[-1].split("/")[-1]
            if local.isalpha():
                words.add(local.lower())
    return words


def find_type_qualified_pairs(graph: Graph) -> List[Tuple[URIRef, URIRef, Set]]:
    type_words = _type_words(graph)
    raw_label_subjects: Dict[str, Set] = defaultdict(set)
    for s, l in graph.subject_objects(RDFS.label):
        raw_label_subjects[str(l).strip().lower()].add(s)

    pairs = []
    seen: Set[Tuple[str, str]] = set()
    for s, l in graph.subject_objects(RDFS.label):
        text = str(l).strip()
        low = text.lower()
        base = None
        m = _PAREN_RE.match(text)
        if m and m.group(2).strip().lower() in type_words:
            base = m.group(1).strip().lower()
        else:
            parts = low.rsplit(" ", 1)
            if len(parts) == 2 and parts[1] in type_words and parts[0]:
                base = parts[0]
        if not base or base == low:
            continue
        for other in raw_label_subjects.get(base, set()):
            if other == s:
                continue
            shared = _shared_type(graph, [s, other])
            shared = {t for t in shared if str(t) not in _SCHEMA_TYPES}
            if not shared:
                continue
            key = tuple(sorted((str(s), str(other))))
            if key in seen:
                continue
            seen.add(key)
            pairs.append((s, other, shared))
    return pairs


def merge_type_qualified_duplicates(graph: Graph, report: MergeReport) -> None:
    for a, b, shared_types in find_type_qualified_pairs(graph):
        ranked = sorted([a, b], key=lambda n: (isinstance(n, URIRef), _connectivity(graph, n)), reverse=True)
        keep, drop = ranked[0], ranked[1]
        rewritten = _merge_node(graph, keep, drop)
        report.merged_groups.append({
            "label": f"type-qualified: {a} / {b}",
            "kept": str(keep),
            "merged": [str(drop)],
            "shared_types": [str(t) for t in shared_types],
            "triples_rewritten": rewritten,
        })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ttl_path")
    parser.add_argument("--output", default=None, help="Defaults to overwriting ttl_path (a .backup copy is made first).")
    parser.add_argument("--dry-run", action="store_true", help="Report only, write nothing.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")

    graph = Graph()
    graph.parse(args.ttl_path, format="turtle")
    triples_before = len(graph)
    logger.info("Loaded %d triples from %s", triples_before, args.ttl_path)

    report = MergeReport()
    merge_type_qualified_duplicates(graph, report)

    logger.info("Found/merged %d type-qualified duplicate pairs:", len(report.merged_groups))
    for g in report.merged_groups:
        logger.info("  KEEP %s  <-  DROP %s  (%d triples rewritten, shared types: %s)",
                    g["kept"], g["merged"][0], g["triples_rewritten"], g["shared_types"])
    logger.info("Triples: %d -> %d", triples_before, len(graph))

    if args.dry_run:
        logger.info("Dry run: nothing written.")
        return

    output_path = args.output or args.ttl_path
    if output_path == args.ttl_path:
        backup_path = args.ttl_path + ".pre_type_dedup.backup"
        shutil.copy(args.ttl_path, backup_path)
        logger.info("Backed up original to %s", backup_path)
    graph.serialize(destination=output_path, format="turtle")
    logger.info("Wrote %s", output_path)


if __name__ == "__main__":
    main()
