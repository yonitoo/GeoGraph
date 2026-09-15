from __future__ import annotations

import argparse
import logging
import re
import shutil
from collections import defaultdict
from typing import Dict, List, Set, Tuple

from rdflib import Graph, Literal, RDF, RDFS, URIRef

logger = logging.getLogger(__name__)

_PAREN_RE = re.compile(r"^(.*?)\s*\(([^)]+)\)$")


def _type_words(graph: Graph) -> Set[str]:
    words = set()
    for t in set(graph.objects(None, RDF.type)):
        if isinstance(t, URIRef):
            local = str(t).split("#")[-1].split("/")[-1]
            if local.isalpha():
                words.add(local.lower())
    return words


def find_missing_aliases(graph: Graph) -> List[Tuple[URIRef, str, str]]:
    type_words = _type_words(graph)
    labels_by_subj: Dict = defaultdict(set)
    raw_label_subjects: Dict[str, Set] = defaultdict(set)
    for s, l in graph.subject_objects(RDFS.label):
        labels_by_subj[s].add(str(l).strip())
        raw_label_subjects[str(l).strip().lower()].add(s)

    candidates = []
    for s, labs in labels_by_subj.items():
        if len(labs) != 1:
            continue
        text = next(iter(labs))
        m = _PAREN_RE.match(text)
        if not m or m.group(2).strip().lower() not in type_words:
            continue
        base = m.group(1).strip()
        if not base or base.lower() == text.lower():
            continue
        others = raw_label_subjects.get(base.lower(), set()) - {s}
        if others:
            continue
        candidates.append((s, text, base))
    return candidates


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

    candidates = find_missing_aliases(graph)
    logger.info("Adding %d bare-name aliases:", len(candidates))
    for s, text, base in candidates:
        logger.info("  %s: %r -> add alias %r", str(s).split("#")[-1], text, base)
        graph.add((s, RDFS.label, Literal(base, lang="bg")))

    logger.info("Triples: %d -> %d", triples_before, len(graph))

    if args.dry_run:
        logger.info("Dry run: nothing written.")
        return

    output_path = args.output or args.ttl_path
    if output_path == args.ttl_path:
        backup_path = args.ttl_path + ".pre_bare_alias.backup"
        shutil.copy(args.ttl_path, backup_path)
        logger.info("Backed up original to %s", backup_path)
    graph.serialize(destination=output_path, format="turtle")
    logger.info("Wrote %s", output_path)


if __name__ == "__main__":
    main()
