from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Set, Tuple

from SPARQLWrapper import JSON, POST, SPARQLWrapper
from rdflib import BNode, Graph, Literal, Node, RDF, URIRef
from tqdm import tqdm

from geograph.config import OntologyConfig, SPARQLConfig
from geograph.ontology.rdf_processor import RDFProcessor

logger = logging.getLogger(__name__)

RelevanceScorer = Callable[[str, List[str]], List[float]]


@dataclass
class ExpansionResult:
    semantic_strings: List[str] = field(default_factory=list)
    protected_strings: Set[str] = field(default_factory=set)


class SPARQLExpander:
    """Expands retrieval context via SPARQL n-hop traversal and type-based expansion."""

    def __init__(self, sparql_config: SPARQLConfig, ont_config: OntologyConfig):
        self.sparql_config = sparql_config
        self.ont = ont_config
        self._quantity_uris = ont_config.quantity_property_uris
        self._value_uri = ont_config.value_property_uri
        self._unit_uri = ont_config.unit_property_uri

    def expand(
        self,
        rdf_processor: RDFProcessor,
        retrieved_metadata: List[Dict[str, Optional[str]]],
        user_query: str,
        relevance_scorer: RelevanceScorer,
        extra_seed_entities: Optional[Set[URIRef]] = None,
    ) -> ExpansionResult:
        """Expand context from retrieved metadata via per-seed relevance-guided
        SPARQL traversal + type-based expansion.
        """
        graph = rdf_processor.graph
        logger.info("SPARQL Expansion: processing %d metadata entries.", len(retrieved_metadata))

        # Extract seed entities and reconstruct original triples
        seeds = self._extract_seeds(graph, retrieved_metadata)
        if extra_seed_entities:
            seeds.entities |= extra_seed_entities
            logger.info("Added %d linked-entity seeds.", len(extra_seed_entities))

        # Per-seed, relevance-guided traversal (divide-and-conquer)
        collected_triples, bnode_details = self._expand_per_seed(
            rdf_processor, seeds.entities, seeds.original_triples, seeds.all_processed,
            user_query, relevance_scorer,
        )

        # Type-based expansion
        pre_type_expansion_triples = set(collected_triples)
        self._type_based_expansion(
            graph, seeds.entity_types, seeds.predicates,
            collected_triples, bnode_details,
        )
        type_expansion_triples = collected_triples - pre_type_expansion_triples

        logger.info("Total %d unique RDF triples after all expansions.", len(collected_triples))

        # Convert to semantic strings
        semantic_strings = self._to_semantic_strings(rdf_processor, collected_triples, bnode_details)
        protected_strings = set(self._to_semantic_strings(rdf_processor, type_expansion_triples, bnode_details))

        return ExpansionResult(semantic_strings=semantic_strings, protected_strings=protected_strings)

    def _extract_seeds(
        self,
        graph: Graph,
        retrieved_metadata: List[Dict[str, Optional[str]]],
    ) -> _SeedData:
        """Extract initial entities, types, predicates, and reconstruct triples from metadata."""
        entities: Set[URIRef] = set()
        original_triples: Set[Tuple[Node, Node, Node]] = set()
        entity_types: List[URIRef] = []
        predicates: List[URIRef] = []

        for meta in retrieved_metadata:
            subj_str = meta.get("subject_uri")
            pred_str = meta.get("predicate_uri")
            obj_str = meta.get("object_uri")
            obj_val = meta.get("object_value")
            obj_dt = meta.get("object_datatype")
            obj_bnode = meta.get("object_bnode_id")

            if subj_str:
                try:
                    subj_uri = URIRef(subj_str)
                    entities.add(subj_uri)

                    for s_type in graph.objects(subject=subj_uri, predicate=RDF.type):
                        if isinstance(s_type, URIRef):
                            entity_types.append(s_type)

                    if pred_str:
                        pred_uri = URIRef(pred_str)
                        predicates.append(pred_uri)

                        obj_node = self._reconstruct_object(
                            graph, subj_uri, pred_uri, obj_str, obj_val, obj_dt, obj_bnode,
                            entities, entity_types,
                        )
                        if obj_node:
                            original_triples.add((subj_uri, pred_uri, obj_node))
                        else:
                            for o_local in graph.objects(subject=subj_uri, predicate=pred_uri):
                                original_triples.add((subj_uri, pred_uri, o_local))
                                if isinstance(o_local, URIRef) and pred_uri != RDF.type:
                                    entities.add(o_local)
                                    for ot in graph.objects(subject=o_local, predicate=RDF.type):
                                        if isinstance(ot, URIRef):
                                            entity_types.append(ot)
                except Exception as e:
                    logger.warning("Error processing metadata: %s. Error: %s", meta, e)
            elif obj_str:
                try:
                    obj_uri = URIRef(obj_str)
                    entities.add(obj_uri)
                    for ot in graph.objects(subject=obj_uri, predicate=RDF.type):
                        if isinstance(ot, URIRef):
                            entity_types.append(ot)
                except Exception as e:
                    logger.warning("Error processing object URI: %s. Error: %s", obj_str, e)

        logger.info("Seeds: %d entities, %d original triples.", len(entities), len(original_triples))
        return _SeedData(
            entities=entities,
            original_triples=original_triples,
            entity_types=entity_types,
            predicates=predicates,
            all_processed=set(entities),
        )

    def _reconstruct_object(
        self,
        graph: Graph,
        subj: URIRef,
        pred: URIRef,
        obj_str: Optional[str],
        obj_val: Optional[str],
        obj_dt: Optional[str],
        obj_bnode: Optional[str],
        entities: Set[URIRef],
        entity_types: List[URIRef],
    ) -> Optional[Node]:
        """Reconstruct the object node of a triple from metadata."""
        if obj_bnode and pred in self._quantity_uris:
            return BNode(obj_bnode)
        if obj_str:
            obj_uri = URIRef(obj_str)
            if pred != RDF.type:
                entities.add(obj_uri)
            for ot in graph.objects(subject=obj_uri, predicate=RDF.type):
                if isinstance(ot, URIRef):
                    entity_types.append(ot)
            return obj_uri
        if obj_val:
            dt_uri = URIRef(obj_dt) if obj_dt else None
            return Literal(obj_val, datatype=dt_uri)
        return None

    def _expand_per_seed(
        self,
        rdf_processor: RDFProcessor,
        seed_entities: Set[URIRef],
        original_triples: Set[Tuple[Node, Node, Node]],
        all_processed: Set[URIRef],
        user_query: str,
        relevance_scorer: RelevanceScorer,
    ) -> Tuple[Set[Tuple[Node, Node, Node]], Dict[str, Dict[str, str]]]:
        collected: Set[Tuple[Node, Node, Node]] = set(original_triples)
        bnode_details: Dict[str, Dict[str, str]] = {}

        if not seed_entities:
            logger.info("No entities for SPARQL expansion.")
            return collected, bnode_details

        sparql = SPARQLWrapper(self.sparql_config.endpoint_url)
        sparql.setReturnFormat(JSON)
        sparql.setTimeout(self.sparql_config.timeout)
        sparql.setMethod(POST)

        beam_width = self.sparql_config.expansion_beam_width
        frontier_cap = self.sparql_config.expansion_frontier_per_seed

        for seed in seed_entities:
            frontier = {seed}
            for hop in range(1, self.sparql_config.max_hops + 1):
                if not frontier:
                    break

                raw_triples = self._fetch_hop(sparql, frontier, bnode_details)
                new_triples = [t for t in raw_triples if t not in collected]
                if not new_triples:
                    break

                kept = self._select_relevant(
                    rdf_processor, new_triples, bnode_details, user_query, relevance_scorer, beam_width,
                )
                collected.update(kept)
                logger.info(
                    "Seed \"%s\", hop %d: %d raw → %d kept.",
                    rdf_processor.get_readable_label(seed), hop, len(raw_triples), len(kept),
                )

                next_frontier: Set[URIRef] = set()
                for s_node, p_node, o_node in kept:
                    candidates = (s_node,) if p_node == RDF.type else (s_node, o_node)
                    for node in candidates:
                        if len(next_frontier) >= frontier_cap:
                            break
                        if isinstance(node, URIRef) and node not in all_processed:
                            next_frontier.add(node)
                            all_processed.add(node)
                frontier = next_frontier

        logger.info(
            "Per-seed expansion complete: %d triples collected from %d seeds.",
            len(collected), len(seed_entities),
        )
        return collected, bnode_details

    def _fetch_hop(
        self,
        sparql: SPARQLWrapper,
        frontier: Set[URIRef],
        bnode_details: Dict[str, Dict[str, str]],
    ) -> List[Tuple[Node, Node, Node]]:
        values_list = [f"<{uri}>" for uri in frontier if isinstance(uri, URIRef)]
        if not values_list:
            return []

        quantity_filter = self._build_quantity_filter()
        value_prop = f"<{self._value_uri}>" if self._value_uri else "rdf:nil"
        unit_prop = f"<{self._unit_uri}>" if self._unit_uri else "rdf:nil"
        query = self._build_nhop_query(
            " ".join(values_list), quantity_filter, value_prop, unit_prop,
            self.sparql_config.expansion_per_seed_limit, str(self.ont.namespace),
        )
        sparql.setQuery(query)

        triples: List[Tuple[Node, Node, Node]] = []
        try:
            results = sparql.queryAndConvert()
            bindings = results["results"]["bindings"]
            for binding in bindings:
                try:
                    s_node = _parse_binding(binding["s"])
                    p_node = _parse_binding(binding["p"])
                    o_node = _parse_binding(binding["o"])

                    is_quant = binding.get("is_quantity_bnode_object", {}).get("value", "false").lower() == "true"
                    if is_quant and isinstance(o_node, BNode):
                        bnode_details[str(o_node)] = {
                            "value": binding.get("o_value", {}).get("value", self.ont.missing_value_string),
                            "unit_uri": binding.get("o_unit_uri", {}).get("value", ""),
                            "unit_label": binding.get("o_unit_label", {}).get("value", ""),
                        }

                    triples.append((s_node, p_node, o_node))
                except Exception as e:
                    logger.error("Error parsing SPARQL binding: %s", e)
        except Exception as e:
            logger.error("SPARQL query error: %s", e)

        return triples

    def _select_relevant(
        self,
        rdf_processor: RDFProcessor,
        triples: List[Tuple[Node, Node, Node]],
        bnode_details: Dict[str, Dict[str, str]],
        user_query: str,
        relevance_scorer: RelevanceScorer,
        beam_width: int,
    ) -> List[Tuple[Node, Node, Node]]:
        if len(triples) <= beam_width:
            return triples

        processed_bnodes: Set[BNode] = set()
        verbalized: List[Tuple[Tuple[Node, Node, Node], str]] = []
        for triple in triples:
            s, p, o = triple
            if isinstance(s, BNode):
                continue
            text = rdf_processor.convert_triple_to_semantic(s, p, o, processed_bnodes, bnode_details)
            if text:
                verbalized.append((triple, text))

        if len(verbalized) <= beam_width:
            return [triple for triple, _text in verbalized]

        scores = relevance_scorer(user_query, [text for _triple, text in verbalized])
        scored = sorted(zip(verbalized, scores), key=lambda pair: -pair[1])
        return [triple for (triple, _text), _score in scored[:beam_width]]

    def _type_based_expansion(
        self,
        graph: Graph,
        entity_types: List[URIRef],
        predicates: List[URIRef],
        collected: Set[Tuple[Node, Node, Node]],
        bnode_details: Dict[str, Dict[str, str]],
    ) -> None:
        if not entity_types or not predicates:
            logger.info("Insufficient data for type-based expansion.")
            return

        type_counts = Counter(entity_types)
        pred_counts = Counter(predicates)

        ns = self.ont.namespace
        excluded_types = {ns.Местоположение, ns.КоличественаСтойност, ns.МернаЕдиница}

        most_common_type = None
        for t, _ in type_counts.most_common():
            if str(t).startswith(str(ns)) and t not in excluded_types:
                most_common_type = t
                break

        most_common_quant_pred = None
        for p, _ in pred_counts.most_common():
            if p in self._quantity_uris:
                most_common_quant_pred = p
                break

        if not most_common_type or not most_common_quant_pred:
            logger.info("No suitable type/predicate pair for expansion.")
            return

        value_prop = f"<{self._value_uri}>" if self._value_uri else "rdf:nil"
        unit_prop = f"<{self._unit_uri}>" if self._unit_uri else "rdf:nil"

        query = f"""
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
            PREFIX geographbg: <{str(ns)}>

            SELECT DISTINCT ?s ?p ?o ?o_value ?o_unit_uri ?o_unit_label ?is_quantity_bnode_object
            WHERE {{
                BIND(<{str(most_common_quant_pred)}> AS ?targetProperty)
                ?s_raw ?targetProperty ?o_raw .
                FILTER(isBlank(?o_raw)) .
                ?o_raw {value_prop} ?val_raw .
                ?o_raw {unit_prop} ?unit_uri_raw .
                OPTIONAL {{ ?unit_uri_raw rdfs:label ?unit_label_raw . }}
                BIND(?s_raw AS ?s)
                BIND(?targetProperty AS ?p)
                BIND(?o_raw AS ?o)
                BIND(STR(?val_raw) AS ?o_value)
                BIND(STR(?unit_uri_raw) AS ?o_unit_uri)
                BIND(STR(?unit_label_raw) AS ?o_unit_label)
                BIND(true AS ?is_quantity_bnode_object)
            }}
            LIMIT {self.sparql_config.max_related_by_type}
        """

        sparql = SPARQLWrapper(self.sparql_config.endpoint_url)
        sparql.setReturnFormat(JSON)
        sparql.setTimeout(self.sparql_config.timeout)
        sparql.setMethod(POST)
        sparql.setQuery(query)

        try:
            results = sparql.queryAndConvert()
            bindings = results["results"]["bindings"]
            logger.info("Type-based expansion: %d results.", len(bindings))

            for binding in bindings:
                try:
                    s_node = _parse_binding(binding["s"])
                    p_node = _parse_binding(binding["p"])
                    o_node = _parse_binding(binding["o"])

                    if isinstance(o_node, BNode):
                        is_quant = binding.get("is_quantity_bnode_object", {}).get("value", "false").lower() == "true"
                        if is_quant:
                            bnode_details[str(o_node)] = {
                                "value": binding.get("o_value", {}).get("value", self.ont.missing_value_string),
                                "unit_uri": binding.get("o_unit_uri", {}).get("value", ""),
                                "unit_label": binding.get("o_unit_label", {}).get("value", ""),
                            }

                    triple = (s_node, p_node, o_node)
                    collected.add(triple)
                except Exception as e:
                    logger.error("Error parsing type-expansion binding: %s", e)
        except Exception as e:
            logger.error("Type-based expansion query error: %s", e)

    def _to_semantic_strings(
        self,
        rdf_processor: RDFProcessor,
        collected: Set[Tuple[Node, Node, Node]],
        bnode_details: Dict[str, Dict[str, str]],
    ) -> List[str]:
        from rdflib import BNode as RDFBNode
        results: List[str] = []
        seen: Set[str] = set()
        processed_bnodes: Set[RDFBNode] = set()

        for s, p, o in tqdm(list(collected), desc="Converting expanded triples"):
            if isinstance(s, RDFBNode):
                continue
            text = rdf_processor.convert_triple_to_semantic(s, p, o, processed_bnodes, bnode_details)
            if text and text not in seen:
                results.append(text)
                seen.add(text)

        logger.info("SPARQL expansion produced %d unique semantic triples.", len(results))
        return results

    def _build_quantity_filter(self) -> str:
        parts = [f"?p_b1 = <{uri}>" for uri in self._quantity_uris if uri]
        return " || ".join(parts) if parts else "false"

    @staticmethod
    def _build_nhop_query(
        values_clause: str,
        quantity_filter: str,
        value_prop: str,
        unit_prop: str,
        limit: int,
        namespace: str,
    ) -> str:
        return f"""
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
            PREFIX owl: <http://www.w3.org/2002/07/owl#>
            PREFIX geographbg: <{namespace}>
            PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>

            SELECT DISTINCT ?s ?p ?o ?o_value ?o_unit_uri ?o_unit_label ?is_quantity_bnode_object
            WHERE {{
              VALUES ?entity {{ {values_clause} }}
              {{
                BIND(?entity AS ?s_b1)
                ?s_b1 ?p_b1 ?o_b1 .
                FILTER (?p_b1 NOT IN (rdfs:subClassOf, rdfs:domain, rdfs:range, rdfs:label, rdfs:comment,
                                     owl:sameAs, owl:inverseOf, owl:equivalentClass, owl:disjointWith)) .
                OPTIONAL {{
                    FILTER(isBlank(?o_b1) && ({quantity_filter})) .
                    ?o_b1 {value_prop} ?val_raw_b1 .
                    ?o_b1 {unit_prop} ?unit_uri_raw_b1 .
                    OPTIONAL {{ ?unit_uri_raw_b1 rdfs:label ?unit_label_raw_b1 . }}
                    BIND(STR(?val_raw_b1) AS ?val_final_b1)
                    BIND(STR(?unit_uri_raw_b1) AS ?unit_uri_final_b1)
                    BIND(STR(?unit_label_raw_b1) AS ?unit_label_final_b1)
                    BIND(true AS ?is_quant_final_b1)
                }}
                BIND(?s_b1 AS ?s)
                BIND(?p_b1 AS ?p)
                BIND(?o_b1 AS ?o)
                BIND(COALESCE(?val_final_b1, "") AS ?o_value)
                BIND(COALESCE(?unit_uri_final_b1, "") AS ?o_unit_uri)
                BIND(COALESCE(?unit_label_final_b1, "") AS ?o_unit_label)
                BIND(COALESCE(?is_quant_final_b1, false) AS ?is_quantity_bnode_object)
              }}
              UNION
              {{
                BIND(?entity AS ?o_b2)
                ?s_b2 ?p_b2 ?o_b2 .
                FILTER (!isBlank(?s_b2)) .
                FILTER (?p_b2 NOT IN (rdfs:subClassOf, rdfs:domain, rdfs:range, rdfs:label, rdfs:comment,
                                     owl:sameAs, owl:inverseOf, owl:equivalentClass, owl:disjointWith)) .
                BIND(?s_b2 AS ?s)
                BIND(?p_b2 AS ?p)
                BIND(?o_b2 AS ?o)
                BIND("" AS ?o_value)
                BIND("" AS ?o_unit_uri)
                BIND("" AS ?o_unit_label)
                BIND(false AS ?is_quantity_bnode_object)
              }}
            }}
            LIMIT {limit}
        """


class _SeedData:
    __slots__ = ("entities", "original_triples", "entity_types", "predicates", "all_processed")

    def __init__(
        self,
        entities: Set[URIRef],
        original_triples: Set[Tuple[Node, Node, Node]],
        entity_types: List[URIRef],
        predicates: List[URIRef],
        all_processed: Set[URIRef],
    ):
        self.entities = entities
        self.original_triples = original_triples
        self.entity_types = entity_types
        self.predicates = predicates
        self.all_processed = all_processed


def _parse_binding(binding_value: dict) -> Node:
    vtype = binding_value.get("type")
    value = binding_value.get("value")

    if vtype == "uri":
        return URIRef(value)
    if vtype in ("literal", "typed-literal"):
        lang = binding_value.get("xml:lang")
        datatype = binding_value.get("datatype")
        dt_uri = URIRef(datatype) if datatype else None
        return Literal(value, lang=lang, datatype=dt_uri)
    if vtype == "bnode":
        return BNode(value)

    logger.warning("Unknown SPARQL value type: %s", vtype)
    return Literal(str(value))
