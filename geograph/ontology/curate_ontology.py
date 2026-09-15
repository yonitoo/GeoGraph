from __future__ import annotations

import argparse
import logging
import shutil
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Set

from rdflib import Graph, Literal, Namespace, RDF, RDFS, URIRef, XSD

from geograph.config import OntologyConfig
from geograph.ontology.rdf_processor import RDFProcessor

logger = logging.getLogger(__name__)

GEOGRAPHBG = Namespace("http://example.org/geographbg#")


VALUE_PROPERTY = GEOGRAPHBG.имаСтойност
UNIT_PROPERTY = GEOGRAPHBG.имаМернаЕдиница
QUANTITY_TYPE = GEOGRAPHBG.КоличественаСтойност

UNIT_URI_ALIASES: Dict[URIRef, URIRef] = {
    URIRef("http://example.org/geographbg#‰"): GEOGRAPHBG.Промил,
}


@dataclass
class MergeReport:
    merged_groups: List[Dict] = field(default_factory=list)
    ambiguous_groups: List[Dict] = field(default_factory=list)
    labels_added: List[str] = field(default_factory=list)
    placeholder_values_removed: List[str] = field(default_factory=list)
    units_canonicalized: List[str] = field(default_factory=list)
    quantity_duplicates_removed: List[str] = field(default_factory=list)
    conflicting_quantities: List[Dict] = field(default_factory=list)


def _label_groups(graph: Graph) -> Dict[str, List]:
    """label text -> distinct subjects carrying it. A single entity can have
    more than one rdfs:label triple (e.g. differing only by @bg tag or
    casing) without being a duplicate entity, so subjects are deduplicated
    before deciding a group is worth merging."""
    groups: Dict[str, Set] = defaultdict(set)
    for subj, label in graph.subject_objects(RDFS.label):
        groups[str(label).strip().lower()].add(subj)
    return {k: sorted(v, key=str) for k, v in groups.items() if len(v) > 1}


def _shared_type(graph: Graph, nodes: List) -> Set:
    type_sets = [set(graph.objects(n, RDF.type)) for n in nodes]
    return set.intersection(*type_sets) if type_sets else set()


def _connectivity(graph: Graph, node) -> int:
    return len(list(graph.predicate_objects(node))) + len(list(graph.subject_predicates(node)))


def _merge_node(graph: Graph, keep, drop) -> int:
    """Rewrite every triple referencing `drop` to reference `keep` instead."""
    rewritten = 0
    for p, o in list(graph.predicate_objects(drop)):
        graph.remove((drop, p, o))
        graph.add((keep, p, o))
        rewritten += 1
    for s, p in list(graph.subject_predicates(drop)):
        graph.remove((s, p, drop))
        graph.add((s, p, keep))
        rewritten += 1
    return rewritten


def merge_duplicate_labels(graph: Graph, report: MergeReport) -> None:
    for label_text, nodes in _label_groups(graph).items():
        shared_types = _shared_type(graph, nodes)
        if not shared_types:
            report.ambiguous_groups.append({
                "label": label_text,
                "nodes": [str(n) for n in nodes],
                "types": [[str(t) for t in graph.objects(n, RDF.type)] for n in nodes],
            })
            continue

        # Prefer a real URI as the merge target over a blank node; break
        # remaining ties by how connected each candidate already is.
        ranked = sorted(nodes, key=lambda n: (isinstance(n, URIRef), _connectivity(graph, n)), reverse=True)
        keep, drops = ranked[0], ranked[1:]
        rewritten = sum(_merge_node(graph, keep, drop) for drop in drops)
        report.merged_groups.append({
            "label": label_text,
            "kept": str(keep),
            "merged": [str(d) for d in drops],
            "triples_rewritten": rewritten,
        })


_NUMERIC_DATATYPES = {XSD.decimal, XSD.integer, XSD.float, XSD.double}


def remove_placeholder_quantity_values(graph: Graph, report: MergeReport) -> None:
    """Drop quantity blank nodes whose имаСтойност is typed as a number but
    isn't one.
    """
    for node, value in list(graph.subject_objects(VALUE_PROPERTY)):
        if value.datatype not in _NUMERIC_DATATYPES:
            continue
        try:
            float(str(value))
        except ValueError:
            for p, o in list(graph.predicate_objects(node)):
                graph.remove((node, p, o))
            for s, p in list(graph.subject_predicates(node)):
                graph.remove((s, p, node))
            report.placeholder_values_removed.append(f"{node} имаСтойност {value!r}")


def canonicalize_units(graph: Graph, report: MergeReport) -> None:
    for alias, canonical in UNIT_URI_ALIASES.items():
        for bnode in list(graph.subjects(UNIT_PROPERTY, alias)):
            graph.remove((bnode, UNIT_PROPERTY, alias))
            graph.add((bnode, UNIT_PROPERTY, canonical))
            report.units_canonicalized.append(f"{bnode} {UNIT_PROPERTY} {alias} -> {canonical}")


def _quantity_signature(graph: Graph, bnode) -> tuple:
    value = graph.value(bnode, VALUE_PROPERTY)
    unit = graph.value(bnode, UNIT_PROPERTY)
    return (str(value) if value is not None else None, str(unit) if unit is not None else None)


def _is_quantity_bnode(graph: Graph, o) -> bool:
    return not isinstance(o, URIRef) and graph.value(o, VALUE_PROPERTY) is not None


def dedupe_quantity_values(graph: Graph, report: MergeReport) -> None:
    by_sp: Dict[tuple, List] = defaultdict(list)
    for s, p, o in graph:
        if not _is_quantity_bnode(graph, o):
            continue
        by_sp[(s, p)].append(o)

    for (s, p), bnodes in by_sp.items():
        if len(bnodes) < 2:
            continue
        by_signature: Dict[tuple, List] = defaultdict(list)
        for bn in bnodes:
            by_signature[_quantity_signature(graph, bn)].append(bn)
        for signature, dupes in by_signature.items():
            if len(dupes) < 2 or signature == (None, None):
                continue
            keep, drops = dupes[0], dupes[1:]
            for drop in drops:
                graph.remove((s, p, drop))
                for dp, do in list(graph.predicate_objects(drop)):
                    graph.remove((drop, dp, do))
                report.quantity_duplicates_removed.append(f"{s} {p} [{signature[0]} {signature[1]}] (kept {keep})")


def report_conflicting_quantities(graph: Graph, report: MergeReport) -> None:
    by_sp: Dict[tuple, List] = defaultdict(list)
    for s, p, o in graph:
        if not _is_quantity_bnode(graph, o):
            continue
        by_sp[(s, p)].append(o)

    for (s, p), bnodes in by_sp.items():
        signatures = {_quantity_signature(graph, bn) for bn in bnodes}
        if len(signatures) > 1:
            report.conflicting_quantities.append({
                "subject": str(s), "predicate": str(p),
                "values": sorted(f"{v} {u}" if u else str(v) for v, u in signatures if v),
            })


def label_unlabeled_entities(graph: Graph, report: MergeReport) -> None:
    processor = RDFProcessor(graph, OntologyConfig(namespace=GEOGRAPHBG))
    typed_entities = {
        s for s in graph.subjects(RDF.type, None)
        if isinstance(s, URIRef) and str(s).startswith(str(GEOGRAPHBG))
    }
    for entity in typed_entities:
        if (entity, RDFS.label, None) in graph:
            continue
        label_text = processor._label_for_uri(entity)
        graph.add((entity, RDFS.label, Literal(label_text, lang="bg")))
        report.labels_added.append(f"{entity} -> {label_text}")


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
    merge_duplicate_labels(graph, report)
    remove_placeholder_quantity_values(graph, report)
    canonicalize_units(graph, report)
    dedupe_quantity_values(graph, report)
    report_conflicting_quantities(graph, report)
    label_unlabeled_entities(graph, report)

    triples_rewritten = sum(g["triples_rewritten"] for g in report.merged_groups)
    logger.info("Merged %d duplicate-label groups (%d triples rewritten).", len(report.merged_groups), triples_rewritten)
    logger.info("Ambiguous groups left untouched (different types — needs manual review): %d", len(report.ambiguous_groups))
    for group in report.ambiguous_groups:
        logger.info("  AMBIGUOUS %r: %s", group["label"], group["nodes"])
    logger.info("Added labels to %d previously-unlabeled entities.", len(report.labels_added))
    logger.info("Removed %d placeholder (non-numeric) quantity values.", len(report.placeholder_values_removed))
    for entry in report.placeholder_values_removed:
        logger.info("  REMOVED %s", entry)
    logger.info("Canonicalized %d aliased unit URIs.", len(report.units_canonicalized))
    logger.info("Removed %d duplicate quantity blank nodes (same subject/predicate/value/unit).", len(report.quantity_duplicates_removed))
    logger.info("Conflicting quantities left untouched (different values, needs manual review): %d", len(report.conflicting_quantities))
    for entry in report.conflicting_quantities:
        logger.info("  CONFLICT %s %s: %s", entry["subject"], entry["predicate"], entry["values"])
    logger.info("Triples: %d -> %d", triples_before, len(graph))

    if args.dry_run:
        logger.info("Dry run: no file written.")
        return

    output_path = args.output or args.ttl_path
    if output_path == args.ttl_path:
        backup_path = args.ttl_path + ".backup"
        shutil.copy2(args.ttl_path, backup_path)
        logger.info("Backup saved to %s", backup_path)

    graph.serialize(destination=output_path, format="turtle")
    logger.info("Wrote cleaned ontology (%d triples) -> %s", len(graph), output_path)


if __name__ == "__main__":
    main()
