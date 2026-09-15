import ast
import datetime
import logging
import os
import pathlib
import re
import time
import urllib.parse
import warnings
from collections import Counter
from typing import List, Any, Dict, Set, Tuple, Optional

import chromadb
from SPARQLWrapper import SPARQLWrapper, JSON, POST
from jsonlines import jsonlines
from openai import OpenAI
from rdflib import Graph, BNode, RDFS, RDF, Literal, OWL, Node, XSD, URIRef
from tqdm import tqdm

from geograph.constants import TOP_K, CHROMA_COLLECTION_NAME, TTL_FILE_PATH, GEOGRAPHBG, EMBEDDING_MODEL_NAME, \
    ONT_QUANTITY_PROPERTIES, ONT_VALUE_PROPERTY, ONT_UNIT_PROPERTY, REASONING_MODEL_PROMPT, \
    OPENAI_API_KEY, GRAPHDB_REPO_ENDPOINT, OPENAI_REASONING_MODEL_SYSTEM_PROMPT, MISSING_VALUE_STRING, \
    UNWANTED_TRIPLE_ENDINGS
from geograph.retrieval.embed import embed, load_model_and_tokenizer
from geograph.llm.generate_response_bggpt import generate_single_response_bggpt, create_prompt

warnings.filterwarnings("ignore")

ONT_QUANTITY_PROPERTIES_URIS = {GEOGRAPHBG[prop_name] for prop_name in ONT_QUANTITY_PROPERTIES}
ONT_VALUE_PROPERTY_URI = GEOGRAPHBG[ONT_VALUE_PROPERTY] if ONT_VALUE_PROPERTY else None
ONT_UNIT_PROPERTY_URI = GEOGRAPHBG[ONT_UNIT_PROPERTY] if ONT_UNIT_PROPERTY else None

def get_readable_label(graph: Graph, resource: Any) -> str:
    """
    Tries to get a human-readable label for an RDF resource.
    Prefers rdfs:label, falls back to local name (decoded and de-camelcased)
    or full URI/Literal value.
    """
    if isinstance(resource, Literal):
        if resource.datatype in {XSD.decimal, XSD.integer, XSD.float, XSD.double}:
            try:
                val = resource.value
                if val is None:
                    return MISSING_VALUE_STRING
                if isinstance(val, float) and val.is_integer():
                    return str(int(val))
                return str(val)
            except (TypeError, ValueError):
                return str(resource.value) if resource.value is not None else MISSING_VALUE_STRING
        return str(resource.value) if resource.value is not None else ""
    if isinstance(resource, BNode):
        return "[празен възел]"
    if not isinstance(resource, URIRef):
        return str(resource)

    try:
        label_literal = graph.value(subject=resource, predicate=RDFS.label)
        if label_literal and isinstance(label_literal, Literal):
            return str(label_literal.value)
    except Exception as e:
        logging.debug(f"Error getting rdfs:label for {resource}: {e}")
        pass

    try:
        full_uri_str = str(resource)
        if "#" in full_uri_str:
            local_name = full_uri_str.split("#")[-1]
        else:
            local_name = full_uri_str.split("/")[-1]

        if local_name:
            decoded_name = urllib.parse.unquote(local_name)
            spaced_name = decoded_name.replace("_", " ")
            de_camel_cased_name = re.sub(r"([А-ЯЯУЕИАОЪЬЮ])", r" \1", spaced_name).strip()
            final_readable_name = re.sub(r"\s+", " ", de_camel_cased_name).strip()
            if final_readable_name:
                return final_readable_name
            elif decoded_name:
                return decoded_name
    except Exception:
        pass
    return str(resource)


def convert_rdf_triple_to_semantic(graph: Graph, s: Node, p: Node, o: Node, processed_bnodes: Set[BNode],
                                   sparql_bnode_details: Optional[Dict[str, Dict[str, str]]] = None) -> Optional[str]:
    if p in ONT_QUANTITY_PROPERTIES_URIS and isinstance(o, BNode):
        if o in processed_bnodes:
            return None

        value_str = MISSING_VALUE_STRING
        unit_label_str = ""

        bnode_id_key = str(o)
        details_from_sparql_used = False

        if sparql_bnode_details and bnode_id_key in sparql_bnode_details:
            details = sparql_bnode_details[bnode_id_key]
            value_str = details.get("value", MISSING_VALUE_STRING)
            unit_label_str = details.get("unit_label", "")
            unit_uri_from_sparql = details.get("unit_uri", "")
            details_from_sparql_used = True

            if not unit_label_str and unit_uri_from_sparql:
                try:
                    readable_unit = get_readable_label(graph, URIRef(unit_uri_from_sparql))
                    if readable_unit == unit_uri_from_sparql and ('#' in readable_unit or '/' in readable_unit):
                        unit_label_str = ""
                    else:
                        unit_label_str = readable_unit
                except Exception:
                    unit_label_str = ""

            if value_str != MISSING_VALUE_STRING:
                try:
                    val_float = float(value_str)
                    if val_float.is_integer():
                        value_str = str(int(val_float))
                    else:
                        value_str = str(val_float)
                except ValueError:
                    pass

        if not details_from_sparql_used:
            value_literal = graph.value(subject=o, predicate=ONT_VALUE_PROPERTY_URI) if ONT_VALUE_PROPERTY_URI else None
            unit_uri_node = graph.value(subject=o, predicate=ONT_UNIT_PROPERTY_URI) if ONT_UNIT_PROPERTY_URI else None

            value_str = get_readable_label(graph, value_literal) if value_literal else MISSING_VALUE_STRING
            unit_label_str = get_readable_label(graph, unit_uri_node) if unit_uri_node else ""

        o_combined = f"{value_str} {unit_label_str}".strip()

        if (value_str == MISSING_VALUE_STRING and not unit_label_str) or o_combined == MISSING_VALUE_STRING or \
                (value_str == MISSING_VALUE_STRING and o_combined == unit_label_str.strip()):
            logging.debug(f"Skipping quantitative BNode {o} for {s} {p} due to effectively missing value (o_combined: '{o_combined}').")
            return None

        processed_bnodes.add(o)
        s_label = get_readable_label(graph, s)
        p_label = get_readable_label(graph, p)
        if s_label == str(s) or p_label == str(p):
            logging.debug(f"Skipping quantitative triple due to non-readable subject/predicate: S='{s_label}', P='{p_label}'")
            return None
        return f"{s_label} {p_label} {o_combined}"

    elif p == RDF.type:
        schema_classes_to_ignore = {
            OWL.Class, RDFS.Class, OWL.Ontology, OWL.AnnotationProperty,
            OWL.ObjectProperty, OWL.DatatypeProperty, OWL.SymmetricProperty,
            OWL.TransitiveProperty, OWL.FunctionalProperty, OWL.InverseFunctionalProperty,
            RDF.Property, RDFS.Resource, OWL.Thing,
            GEOGRAPHBG.КоличественаСтойност,
        }
        if not isinstance(o, URIRef) or o in schema_classes_to_ignore:
            return None
        s_label = get_readable_label(graph, s)
        o_label = get_readable_label(graph, o)

        if s_label == str(s) or o_label == str(o):
            logging.debug(f"Skipping rdf:type triple due to non-readable labels: {s_label} is type {o_label}")
            return None
        return f"{s_label} е тип {o_label}"

    elif p in {RDFS.comment, RDFS.subClassOf, RDFS.label, RDFS.domain, RDFS.range, OWL.sameAs, OWL.inverseOf, OWL.equivalentClass, OWL.disjointWith}:
        return None

    else:
        if p in ONT_QUANTITY_PROPERTIES_URIS and isinstance(o, URIRef) and "Entity" in str(o):
            logging.warning(
                f"Encountered a quantitative property {get_readable_label(graph, p)} "
                f"with a URIRef object '{get_readable_label(graph, o)}' that looks like a fallback URI. "
                f"This might indicate an issue in KG construction for subject {get_readable_label(graph, s)}. Skipping triple."
            )
            return None

        if isinstance(o, BNode):
            logging.debug(f"Skipping triple with unhandled BNode object: {s} {p} {o}")
            return None

        s_label = get_readable_label(graph, s)
        o_label = get_readable_label(graph, o)
        p_label = get_readable_label(graph, p)

        if s_label == str(s) or p_label == str(p):
            logging.debug(f"Skipping triple due to non-readable subject or predicate: S='{s_label}', P='{p_label}', O='{o_label}'")
            return None

        if isinstance(o, URIRef) and o_label == str(o):
            logging.debug(f"Skipping triple due to non-readable object URI: S='{s_label}', P='{p_label}', O='{o_label}'")
            return None

        if isinstance(o, Literal) and not o_label and o.value is None:
            logging.debug(f"Skipping triple with empty literal object: {s} {p} {o}")
            return None
        return f"{s_label} {p_label} {o_label}"


def load_data(ttl_path: str) -> Graph:
    if not os.path.exists(ttl_path):
        raise FileNotFoundError(f"TTL file not found at: {ttl_path}")

    graph = Graph()
    logging.info(f"Loading RDF data from {ttl_path}...")
    graph.parse(ttl_path, format="turtle", publicID=str(GEOGRAPHBG))
    logging.info(f"Loaded {len(graph)} triples from {ttl_path}")
    return graph


def _get_readable_types_for_node(graph: Graph, node: Node, ignored_types: Set[URIRef]) -> str:
    """
    Помощна функция за получаване на низ от четими етикети на типове за даден URI.
    Връща празен низ, ако не са намерени релевантни типове или ако възелът не е URIRef.
    """
    if not isinstance(node, URIRef):
        return ""

    type_labels = [
        get_readable_label(graph, type_uri)
        for type_uri in graph.objects(node, RDF.type)
        if isinstance(type_uri, URIRef) and type_uri not in ignored_types
    ]
    if type_labels:
        return ", ".join(sorted(list(set(type_labels))))
    return ""


def process_graph_data(graph: Graph) -> tuple[list[str], list[dict]]:
    """
    Обработва RDF граф, за да генерира семантични низове и съответните им метаданни.

    Args:
        graph: rdflib.Graph обектът за обработка.

    Returns:
        Кортеж, съдържащ:
            - Списък от семантични низове, представящи тройките.
            - Списък от речници с метаданни за всяка тройка.
    """
    semantic_strings: List[str] = []
    metadata_list: List[Dict[str, Any]] = []
    processed_bnodes_globally: Set[BNode] = set()

    logging.info("Processing triples into semantic strings and creating metadata...")
    for s, p, o in tqdm(graph, desc="Processing triples"):
        if isinstance(s, BNode):
            continue

        semantic_triple_str = convert_rdf_triple_to_semantic(graph, s, p, o, processed_bnodes_globally)
        if semantic_triple_str:
            semantic_strings.append(semantic_triple_str)

            meta: Dict[str, Any] = {
                "subject_uri": str(s) if isinstance(s, URIRef) else "",
                "predicate_uri": str(p) if isinstance(p, URIRef) else "",
                "predicate_importance": get_predicate_importance(p) if isinstance(p, URIRef) else 99
            }

            if p in ONT_QUANTITY_PROPERTIES_URIS and isinstance(o, BNode):
                value_lit = graph.value(subject=o, predicate=ONT_VALUE_PROPERTY_URI)
                unit_ref = graph.value(subject=o, predicate=ONT_UNIT_PROPERTY_URI)

                meta["object_bnode_id"] = str(o)
                meta["object_uri"] = ""
                meta["object_value"] = str(value_lit.value) if value_lit and hasattr(value_lit,
                                                                                     'value') else ""
                meta["object_datatype"] = str(value_lit.datatype) if value_lit and value_lit.datatype else str(XSD.decimal)
                meta["object_unit_uri"] = str(unit_ref) if unit_ref else ""
                meta["object_unit_label"] = get_readable_label(graph, unit_ref) if unit_ref else ""
            else:
                meta["object_uri"] = str(o) if isinstance(o, URIRef) else ""
                meta["object_value"] = str(o.value) if isinstance(o, Literal) else ""
                meta["object_datatype"] = str(o.datatype) if isinstance(o, Literal) and o.datatype else ""
                if isinstance(o, BNode):
                    meta["object_bnode_id"] = str(o)

            if isinstance(s, URIRef):
                s_types_labels = [
                    get_readable_label(graph, st)
                    for st in graph.objects(s, RDF.type)
                    if isinstance(st, URIRef) and st not in {OWL.NamedIndividual, OWL.Thing, RDFS.Resource}
                ]
                if s_types_labels: meta["subject_types"] = ", ".join(sorted(list(set(s_types_labels))))

            if isinstance(o, URIRef):
                o_types_labels = [
                    get_readable_label(graph, ot)
                    for ot in graph.objects(o, RDF.type)
                    if isinstance(ot, URIRef) and ot not in {OWL.NamedIndividual, OWL.Thing, RDFS.Resource, OWL.Class,
                                                             RDFS.Class}
                ]
                if o_types_labels:
                    meta["object_types"] = ", ".join(sorted(list(set(o_types_labels))))

            metadata_list.append(meta)

    logging.info(f"Processed into {len(semantic_strings)} semantic fact strings with corresponding metadata.")
    return semantic_strings, metadata_list


def _parse_sparql_result_value(binding_value: dict) -> Node:
    """Converts a SPARQL JSON result binding value to an rdflib Node."""
    value_type = binding_value.get("type")
    value = binding_value.get("value")

    if value_type == "uri":
        return URIRef(value)
    elif value_type == "literal" or value_type == "typed-literal":
        lang = binding_value.get("xml:lang")
        datatype = binding_value.get("datatype")
        datatype_uri = URIRef(datatype) if datatype else None
        return Literal(value, lang=lang, datatype=datatype_uri)
    elif value_type == "bnode":
        return BNode(value)
    else:
        logging.warning(f"Unknown SPARQL result value type: {value_type}. Returning as Literal.")
        return Literal(str(value))


def search_graph_sparql(sparql_endpoint_url: str, graph: Graph, retrieved_metadata: List[Dict[str, Optional[str]]], max_hops: int = 2, max_results_per_hop_expansion: int = 30, max_related_by_type_expansion: int = 30) -> List[str]:
    """
    Разширява контекста чрез SPARQL:
    1. N-хоп обхождане от първоначалните същности.
    2. Допълнително извличане на свързани същности от същия тип с ключови свойства.
    """
    print(f"Step: Graph Search (SPARQL) - Input metadata: {len(retrieved_metadata)}, Hops: {max_hops}")
    print(f"Retrieved metadata: {retrieved_metadata}")
    initial_entities: Set[URIRef] = set()
    original_rdf_triples_from_retrieval: Set[Tuple[URIRef, URIRef, Node]] = set()
    initial_entity_types: List[URIRef] = []
    initial_predicates: List[URIRef] = []

    logging.info("Identifying initial entities, types, predicates and reconstructing original triples from retrieval...")
    for meta in retrieved_metadata:
        subj_uri_str = meta.get("subject_uri")
        pred_uri_str = meta.get("predicate_uri")
        obj_uri_str = meta.get("object_uri")
        obj_val_str = meta.get("object_value")
        obj_dt_str = meta.get("object_datatype")
        obj_bnode_id = meta.get("object_bnode_id")

        if subj_uri_str:
            try:
                subj_uri = URIRef(subj_uri_str)
                initial_entities.add(subj_uri)
                current_subj_uri = subj_uri

                for s_type in graph.objects(subject=subj_uri, predicate=RDF.type):
                    if isinstance(s_type, URIRef):
                        initial_entity_types.append(s_type)

                if pred_uri_str:
                    pred_uri = URIRef(pred_uri_str)
                    initial_predicates.append(pred_uri)
                    current_pred_uri = pred_uri
                    obj_node: Optional[Node] = None

                    if obj_bnode_id and pred_uri in ONT_QUANTITY_PROPERTIES_URIS:
                        obj_node = BNode(obj_bnode_id)
                    elif obj_uri_str:
                        obj_node = URIRef(obj_uri_str)
                        initial_entities.add(obj_node)
                        for o_type in graph.objects(subject=obj_node, predicate=RDF.type):
                            if isinstance(o_type, URIRef):
                                initial_entity_types.append(o_type)
                    elif obj_val_str:
                        obj_dt_uri = URIRef(obj_dt_str) if obj_dt_str else None
                        obj_node = Literal(obj_val_str, datatype=obj_dt_uri)

                    if obj_node:
                        original_rdf_triples_from_retrieval.add((subj_uri, pred_uri, obj_node))
                    elif current_subj_uri and current_pred_uri:
                        for o_local in graph.objects(
                                subject=current_subj_uri, predicate=current_pred_uri
                        ):
                            original_rdf_triples_from_retrieval.add(
                                (current_subj_uri, current_pred_uri, o_local)
                            )
                            if isinstance(o_local, URIRef):
                                initial_entities.add(o_local)
                                for o_type in graph.objects(subject=o_local, predicate=RDF.type):
                                    if isinstance(o_type, URIRef):
                                        initial_entity_types.append(o_type)
            except Exception as e:
                logging.warning(f"Error processing metadata URI: {meta}. Error: {e}")
        elif obj_uri_str:
            try:
                obj_uri = URIRef(obj_uri_str)
                initial_entities.add(obj_uri)
                for o_type in graph.objects(subject=obj_uri, predicate=RDF.type):
                    if isinstance(o_type, URIRef):
                        initial_entity_types.append(o_type)
            except Exception as e:
                logging.warning(f"Error processing object URI from metadata: {obj_uri_str}. Error: {e}")

    logging.info(f"Identified {len(initial_entities)} initial entities for expansion.")
    logging.info(f"Reconstructed {len(original_rdf_triples_from_retrieval)} original triples from retrieval.")

    visited_triples: Set[Tuple[Node, Node, Node]] = set(original_rdf_triples_from_retrieval)
    all_collected_triples: Set[Tuple[Node, Node, Node]] = set(original_rdf_triples_from_retrieval)
    sparql_bnode_quantity_details: Dict[str, Dict[str, str]] = {}
    entities_to_expand_in_current_hop: Set[URIRef] = set(initial_entities)
    all_processed_entities_for_nhop: Set[URIRef] = set(initial_entities)

    ont_quantity_props_filter_list = [f"?p_b1 = <{uri}>" for uri in ONT_QUANTITY_PROPERTIES_URIS if uri]
    ont_quantity_props_filter_str = " || ".join(ont_quantity_props_filter_list) if ont_quantity_props_filter_list else "false"
    ont_value_prop_uri_str = (f"<{ONT_VALUE_PROPERTY_URI}>" if ONT_VALUE_PROPERTY_URI else "rdf:nil")
    ont_unit_prop_uri_str = (f"<{ONT_UNIT_PROPERTY_URI}>" if ONT_UNIT_PROPERTY_URI else "rdf:nil")

    if not entities_to_expand_in_current_hop:
        logging.info("No valid entities for N-hop SPARQL expansion.")
    else:
        sparql = SPARQLWrapper(sparql_endpoint_url)
        sparql.setReturnFormat(JSON)
        sparql.setTimeout(60)
        sparql.setMethod(POST)

        for hop in range(1, max_hops + 1):
            if not entities_to_expand_in_current_hop:
                logging.info(f"N-hop: No new entities to expand at hop {hop}.")
                break
            logging.info(f"N-hop {hop}: Expanding {len(entities_to_expand_in_current_hop)} entities via SPARQL...")
            values_list = [
                f"<{uri}>"
                for uri in entities_to_expand_in_current_hop
                if isinstance(uri, URIRef)
            ]
            if not values_list:
                logging.info(f"N-hop: No valid URIs in frontier for hop {hop}.")
                break
            values_clause = " ".join(values_list)
            limit_value = max_results_per_hop_expansion * len(values_list)

            query_string = f"""
                PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
                PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
                PREFIX owl: <http://www.w3.org/2002/07/owl#>
                PREFIX geographbg: <{str(GEOGRAPHBG)}>
                PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>

                SELECT DISTINCT ?s ?p ?o ?o_value ?o_unit_uri ?o_unit_label ?is_quantity_bnode_object
                WHERE {{
                  VALUES ?entity {{ {values_clause} }}
                  {{ # Клон 1: Изходящи свойства от ?entity
                    BIND(?entity AS ?s_b1)
                    ?s_b1 ?p_b1 ?o_b1 .

                    FILTER (?p_b1 NOT IN (rdfs:subClassOf, rdfs:domain, rdfs:range, rdfs:label, rdfs:comment,
                                       owl:sameAs, owl:inverseOf, owl:equivalentClass, owl:disjointWith)) .

                    OPTIONAL {{
                        FILTER(isBlank(?o_b1) && ({ont_quantity_props_filter_str})) .

                        ?o_b1 {ont_value_prop_uri_str} ?val_raw_b1 .
                        ?o_b1 {ont_unit_prop_uri_str} ?unit_uri_raw_b1 .
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
                  {{ # Клон 2: Входящи свойства към ?entity
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
                LIMIT {limit_value}
                """
            logging.debug(f"N-hop SPARQL Query (Hop {hop}):\n{query_string}")
            sparql.setQuery(query_string)
            entities_for_next_hop: Set[URIRef] = set()
            triples_found_this_hop: Set[Tuple[Node, Node, Node]] = set()

            try:
                results = sparql.queryAndConvert()
                bindings = results["results"]["bindings"]
                logging.info(f"N-hop SPARQL query (Hop {hop}) returned {len(bindings)} potential neighbor triples.")
                for binding in bindings:
                    try:
                        s_node = _parse_sparql_result_value(binding["s"])
                        p_node = _parse_sparql_result_value(binding["p"])
                        o_node = _parse_sparql_result_value(binding["o"])

                        is_quant_obj_str = binding.get("is_quantity_bnode_object", {}).get("value", "false")
                        is_quant_obj = is_quant_obj_str.lower() == "true"

                        if is_quant_obj and isinstance(o_node, BNode):
                            bnode_id_str = str(o_node)
                            sparql_bnode_quantity_details[bnode_id_str] = {
                                "value": binding.get("o_value", {}).get("value", MISSING_VALUE_STRING),
                                "unit_uri": binding.get("o_unit_uri", {}).get("value", ""),
                                "unit_label": binding.get("o_unit_label", {}).get("value", ""),
                            }

                        if isinstance(s_node, BNode) and (s_node, p_node, o_node) not in original_rdf_triples_from_retrieval:
                            continue

                        triple = (s_node, p_node, o_node)
                        if triple not in visited_triples:
                            triples_found_this_hop.add(triple)
                            visited_triples.add(triple)
                            if isinstance(s_node, URIRef) and s_node not in entities_to_expand_in_current_hop and s_node not in all_processed_entities_for_nhop:
                                entities_for_next_hop.add(s_node)
                                all_processed_entities_for_nhop.add(s_node)
                            if isinstance(o_node, URIRef) and o_node not in entities_to_expand_in_current_hop and o_node not in all_processed_entities_for_nhop:
                                entities_for_next_hop.add(o_node)
                                all_processed_entities_for_nhop.add(o_node)
                    except Exception as e:
                        logging.error(f"Error parsing N-hop SPARQL result binding (Hop {hop}): {binding} - Error: {e}")
            except Exception as e:
                logging.error(f"Error querying N-hop SPARQL endpoint (Hop {hop}): {e}. Query was:\n{query_string}")
                break
            if triples_found_this_hop:
                all_collected_triples.update(triples_found_this_hop)
            entities_to_expand_in_current_hop = entities_for_next_hop

    logging.info(f"N-hop: Total {len(all_collected_triples)} unique RDF triples collected after N-hop expansion.")
    logging.info("Attempting additional type-based and property-based expansion...")

    if initial_entity_types and initial_predicates:
        type_counts = Counter(initial_entity_types)
        predicate_counts = Counter(initial_predicates)

        most_common_type: Optional[URIRef] = None
        for t, count in type_counts.most_common():
            if str(t).startswith(str(GEOGRAPHBG)) and t not in {GEOGRAPHBG.Местоположение, GEOGRAPHBG.КоличественаСтойност, GEOGRAPHBG.МернаЕдиница}:
                most_common_type = t
                break

        most_common_quantity_predicate: Optional[URIRef] = None
        for p, count in predicate_counts.most_common():
            if p in ONT_QUANTITY_PROPERTIES_URIS:
                most_common_quantity_predicate = p
                break

        if most_common_type and most_common_quantity_predicate:
            logging.info(f"Identified most common type: {get_readable_label(graph, most_common_type)} and quantity predicate: {get_readable_label(graph, most_common_quantity_predicate)} for additional expansion.")

            # Конструиране на заявка за извличане на други инстанции с това свойство
            type_expansion_query = f"""
                PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
                PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
                PREFIX owl: <http://www.w3.org/2002/07/owl#>
                PREFIX geographbg: <{str(GEOGRAPHBG)}>
                PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>

                SELECT DISTINCT ?s ?p ?o ?o_value ?o_unit_uri ?o_unit_label ?is_quantity_bnode_object
                WHERE {{
                    # BIND(<{str(most_common_type)}> AS ?targetClass)
                    BIND(<{str(most_common_quantity_predicate)}> AS ?targetProperty)

                    # ?s_raw rdf:type ?targetClass .
                    ?s_raw ?targetProperty ?o_raw .

                    FILTER(isBlank(?o_raw)) .
                    ?o_raw {ont_value_prop_uri_str} ?val_raw_type_exp .
                    ?o_raw {ont_unit_prop_uri_str} ?unit_uri_raw_type_exp .
                    OPTIONAL {{ ?unit_uri_raw_type_exp rdfs:label ?unit_label_raw_type_exp . }}

                    BIND(?s_raw AS ?s)
                    BIND(?targetProperty AS ?p)
                    BIND(?o_raw AS ?o)
                    BIND(STR(?val_raw_type_exp) AS ?o_value)
                    BIND(STR(?unit_uri_raw_type_exp) AS ?o_unit_uri)
                    BIND(STR(?unit_label_raw_type_exp) AS ?o_unit_label)
                    BIND(true AS ?is_quantity_bnode_object)
                }}
                LIMIT {max_related_by_type_expansion}
                """
            logging.debug(f"Type-based expansion SPARQL Query:\n{type_expansion_query}")
            sparql.setQuery(type_expansion_query)
            try:
                results_type_exp = sparql.queryAndConvert()
                bindings_type_exp = results_type_exp["results"]["bindings"]
                logging.info(f"Type-based expansion returned {len(bindings_type_exp)} triples.")

                for binding in bindings_type_exp:
                    try:
                        s_node = _parse_sparql_result_value(binding["s"])
                        p_node = _parse_sparql_result_value(binding["p"])
                        o_node = _parse_sparql_result_value(binding["o"])

                        if isinstance(o_node, BNode) and binding.get("is_quantity_bnode_object", {}).get("value", "false").lower() == "true":
                            bnode_id_str = str(o_node)
                            sparql_bnode_quantity_details[bnode_id_str] = {
                                "value": binding.get("o_value", {}).get("value", MISSING_VALUE_STRING),
                                "unit_uri": binding.get("o_unit_uri", {}).get("value", ""),
                                "unit_label": binding.get("o_unit_label", {}).get("value", ""),
                            }

                        triple = (s_node, p_node, o_node)
                        if triple not in visited_triples:
                            all_collected_triples.add(triple)
                            visited_triples.add(triple)
                            if isinstance(s_node, URIRef):
                                all_processed_entities_for_nhop.add(s_node)


                    except Exception as e:
                        logging.error(f"Error parsing type-based expansion result: {binding} - Error: {e}")
            except Exception as e:
                logging.error(f"Error executing type-based expansion query: {e}\nQuery:\n{type_expansion_query}")
        else:
            logging.info("Not enough information (common type/quantity predicate) for type-based expansion.")

    logging.info(f"Total {len(all_collected_triples)} unique RDF triples collected after all expansions.")

    final_semantic_triples: List[str] = []
    processed_bnodes_for_semantic_conversion: Set[BNode] = set()
    unique_semantic_strings: Set[str] = set()

    logging.info("Converting all collected RDF triples to semantic format...")
    for s, p, o in tqdm(list(all_collected_triples), desc="Converting all collected triples"):
        if isinstance(s, BNode):
            continue
        semantic_triple = convert_rdf_triple_to_semantic(graph, s, p, o, processed_bnodes_for_semantic_conversion, sparql_bnode_quantity_details)
        if semantic_triple and semantic_triple not in unique_semantic_strings:
            final_semantic_triples.append(semantic_triple)
            unique_semantic_strings.add(semantic_triple)

    logging.info(f"Step: Graph Search (SPARQL) - Output {len(final_semantic_triples)} unique semantic triples.")
    return final_semantic_triples


def is_valid_pre_filtered_triple(triple: str) -> bool:
    """Проверява дали семантичната тройка отговаря на критериите за предварително филтриране."""
    if not triple:
        return False
    if MISSING_VALUE_STRING in triple:
        return False

    for ending in UNWANTED_TRIPLE_ENDINGS:
        if triple.endswith(ending):
            return False
    if triple.endswith("държава България"):
        return False
    return True


def filter_with_reasoning_model(client: OpenAI, semantic_triples_to_filter: list[str], user_query: str) -> list[str]:
    logging.info(f"Step: Reasoning Filter - Input: {len(semantic_triples_to_filter)} semantic triples.")
    if not semantic_triples_to_filter:
        logging.info("Step: Reasoning Filter - No triples to filter.")
        return []

    pre_filtered_semantic_triples = [triple for triple in semantic_triples_to_filter if is_valid_pre_filtered_triple(triple)]

    if not pre_filtered_semantic_triples:
        logging.info("Step: Reasoning Filter - All triples removed by pre-filtering.")
        return []

    formatted_triples = "\n".join([f"{i + 1}. \"{triple}\"" for i, triple in enumerate(pre_filtered_semantic_triples)])
    prompt_content = REASONING_MODEL_PROMPT.format(
        kg_triples=formatted_triples,
        user_query=user_query
    )
    print(f"Reasoning Prompt:\n{prompt_content}")

    try:
        response = client.responses.create(
            model="o4-mini",
            reasoning={"effort": "medium"},
            input=[
                {"role": "system", "content": OPENAI_REASONING_MODEL_SYSTEM_PROMPT},
                {"role": "user", "content": prompt_content}
            ],
        )

        response_text = response.output_text
        logging.info(f"Reasoning Model Raw Output:\n{response_text}")

        if response_text.startswith("```python"):
            response_text = response_text[len("```python"):].strip()
        elif response_text.startswith("```"):
            response_text = response_text[3:].strip()
        if response_text.endswith("```"):
            response_text = response_text[:-3].strip()

        if not response_text.startswith("[") or not response_text.endswith("]"):
            logging.warning(
                "Reasoning model output does not look like a list. "
                f"Output: '{response_text}'. Returning pre-filtered triples."
            )
            return pre_filtered_semantic_triples

        try:
            parsed_list = ast.literal_eval(response_text)
            if not isinstance(parsed_list, list):
                raise ValueError("Parsed result is not a list.")

            filtered_triples_from_model: List[str] = []
            for item in parsed_list:
                if not isinstance(item, str):
                    logging.warning(f"Reasoning model returned non-string item in list: {item}. Skipping.")
                cleaned_item = re.sub(r"^\d+\.\s*\"", "\"", item).strip()
                if cleaned_item.startswith("\"") and cleaned_item.endswith("\""):
                    cleaned_item = cleaned_item[1:-1]

                if cleaned_item in pre_filtered_semantic_triples:
                    filtered_triples_from_model.append(cleaned_item)
                else:
                    logging.warning(f"Reasoning model returned a triple not in the original pre-filtered list: '{cleaned_item}'. Skipping.")

        except (SyntaxError, ValueError, TypeError) as e:
            logging.error(
                f"Could not parse reasoning model output as Python list: {e}. "
                f"Problematic output: '{response_text}'. Returning pre-filtered triples."
            )
            return pre_filtered_semantic_triples

    except Exception as e:
        logging.error(f"Error calling OpenAI API for reasoning: {e}. Returning pre-filtered triples.")
        return pre_filtered_semantic_triples

    logging.info(f"Step: Reasoning Filter - Output {len(filtered_triples_from_model)} semantic triples.")
    return filtered_triples_from_model


def embed_and_ingest_data(client: chromadb.Client, ttl_path: str, mtd: Dict[str, Any]) -> tuple[Graph, chromadb.Collection]:
    """
    Fetches and processes RDF data, creates label map and metadata,
    embeds semantic strings, and ingests them into ChromaDB.

    Args:
        client: Initialized ChromaDB client.
        ttl_path: Path to the Turtle file.
        mtd: Dictionary containing the loaded model, tokenizer, and device.

    Returns:
        A tuple containing the rdflib Graph, the ChromaDB collection and a label map.
    """
    graph = load_data(ttl_path)
    semantic_strings, metadata_list = process_graph_data(graph)

    if not semantic_strings:
        logging.error("No semantic strings generated from the graph. Cannot proceed with embedding.")
        collection = client.get_or_create_collection(name=CHROMA_COLLECTION_NAME)
        return graph, collection

    logging.info(f"Embedding {len(semantic_strings)} semantic triples using {EMBEDDING_MODEL_NAME}...")
    embeddings = []
    for text_triple in tqdm(semantic_strings, desc="Embedding triples"):
        embedding = embed(mtd, text_triple)
        embeddings.append(embedding)

    ids = [str(i) for i in range(len(semantic_strings))]

    logging.info(f"Ingesting embeddings into ChromaDB collection '{CHROMA_COLLECTION_NAME}'...")
    try:
        collection = client.get_collection(name=CHROMA_COLLECTION_NAME)
        logging.info(f"Collection '{CHROMA_COLLECTION_NAME}' already exists. Deleting and recreating.")
        client.delete_collection(name=collection.name)
        collection = client.create_collection(name=CHROMA_COLLECTION_NAME)
    except:
        logging.info(f"Collection '{CHROMA_COLLECTION_NAME}' not found or error accessing. Creating new one.")
        collection = client.create_collection(name=CHROMA_COLLECTION_NAME)

    batch_size = 5000
    for i in tqdm(range(0, len(ids), batch_size), desc="Ingesting batches to ChromaDB"):
        batch_ids = ids[i:i + batch_size]
        batch_documents = semantic_strings[i:i + batch_size]
        batch_embeddings = embeddings[i:i + batch_size]
        batch_metadatas = metadata_list[i:i + batch_size]

        if batch_ids:
            collection.add(
                ids=batch_ids,
                documents=batch_documents,
                embeddings=batch_embeddings,
                metadatas=batch_metadatas
            )
    logging.info("Ingestion complete.")
    return graph, collection


def retrieve_relevant_triples(collection: chromadb.Collection, user_query: str, mtd: Dict[str, Any], top_k: int = TOP_K) -> List[dict]:
    """
    Embeds a user query and retrieves relevant semantic triples metadata from ChromaDB.

    Args:
        collection: The ChromaDB collection containing the embeddings.
        user_query: The user's query string.
        mtd: Dictionary containing the loaded model, tokenizer, and device.
        top_k: The number of top results to retrieve.

    Returns:
        A list of relevant semantic triple strings.
    """
    logging.info(f"Retrieving top {top_k} relevant semantic triples for query: '{user_query}'")
    query_embedding = embed(mtd, user_query)
    try:
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["metadatas", "documents", "distances"],
        )
        retrieved_metadata = results.get("metadatas", [[]])[0]
        retrieved_documents = results.get("documents", [[]])[0]
        retrieved_distances = results.get("distances", [[]])[0]

        logging.info(f"Retrieved {len(retrieved_metadata)} metadata entries from vector DB.")
        for i, doc in enumerate(retrieved_documents):
            logging.info(f"  Retrieved doc: \"{doc}\" (Distance: {retrieved_distances[i]:.4f})")

        return retrieved_metadata
    except Exception as e:
        print(f"Error querying ChromaDB collection: {e}")
        return []


def get_predicate_importance(p_uri: URIRef) -> int:
    """
    Връща числова стойност за важността на предиката в контекста на географията на България.
    По-ниско число = по-важно/специфично за директен отговор на типични въпроси.
    """

    # Ключови количествени характеристики, дефиниции и основни атрибути
    if p_uri in ONT_QUANTITY_PROPERTIES_URIS:
        return 1
    if p_uri == RDF.type:
        return 2

    # Основни географски, структурни и топологични връзки
    if p_uri in {
        GEOGRAPHBG.находящСеВ, GEOGRAPHBG.извираОт, GEOGRAPHBG.вливаСеВ,
        GEOGRAPHBG.притокНа, GEOGRAPHBG.имаПриток, GEOGRAPHBG.найВисокВръхНа,
        GEOGRAPHBG.столица, GEOGRAPHBG.еСтолицаНа, GEOGRAPHBG.областенЦентър,
        GEOGRAPHBG.еОбластенЦентърНа, GEOGRAPHBG.общинскиЦентър, GEOGRAPHBG.еОбщинскиЦентърНа,
        GEOGRAPHBG.мястоНаВливане, GEOGRAPHBG.находящСеВПодножиетоНа,
        GEOGRAPHBG.находящСеВПоречиетоНа, GEOGRAPHBG.имаЯзовир, GEOGRAPHBG.еСъставящаЧастНа, GEOGRAPHBG.имаСъставящаЧаст
    }:
        return 3

    # Важни географски и други ключови връзки/атрибути
    if p_uri in {
        GEOGRAPHBG.намираСеНаРека, GEOGRAPHBG.частОтПрироденПарк, GEOGRAPHBG.частОтКаскада,
        GEOGRAPHBG.странаНаПритока, GEOGRAPHBG.свързанС, GEOGRAPHBG.близоДо, GEOGRAPHBG.захранваСеОт
    }:
        return 4

    # Други важни описателни свойства
    if p_uri in {
        GEOGRAPHBG.имаАлтернативноИме, GEOGRAPHBG.основанПрезГодина, GEOGRAPHBG.еЗавършенПрез,
        GEOGRAPHBG.имаТипСтена, GEOGRAPHBG.имаСтатутНаЗащитенаМестност,
        GEOGRAPHBG.обявенаСъсЗаповед, GEOGRAPHBG.официаленЕзик, GEOGRAPHBG.датаНаОбявяване,
        GEOGRAPHBG.датаНаЗавършване, GEOGRAPHBG.датаНаПровеждане
    }:
        return 5

    # По-общи или по-рядко търсени връзки/атрибути
    if p_uri in {
        GEOGRAPHBG.намираСеВЗемлищетоНа, GEOGRAPHBG.разположенВМестност,
        GEOGRAPHBG.проведеноЗа, GEOGRAPHBG.обхващаПериодОтДо, GEOGRAPHBG.строител,
        GEOGRAPHBG.частОтИнфраструктуренОбект
    }:
        return 6

    # Още по-общи връзки или атрибути
    if p_uri in {
        GEOGRAPHBG.граничиС, GEOGRAPHBG.преминаваПрез, GEOGRAPHBG.протичаПрез,
        GEOGRAPHBG.свързва, GEOGRAPHBG.стопанисваСеОт, GEOGRAPHBG.образува,
        GEOGRAPHBG.използваСеЗа
    }:
        return 7

    # Дълги описателни свойства
    if p_uri in {
        GEOGRAPHBG.имаОписание, GEOGRAPHBG.имаИстория, GEOGRAPHBG.имаЕтимология,
        GEOGRAPHBG.имаКлимат, GEOGRAPHBG.имаГеология, GEOGRAPHBG.имаФлора,
        GEOGRAPHBG.имаФауна, GEOGRAPHBG.частОтИсториятаНа
    }:
        return 8

    # Свойства, свързани с класификации (почви, полезни изкопаеми и други)
    if p_uri in {
        GEOGRAPHBG.имаПолезноИзкопаемо, GEOGRAPHBG.имаТипПочва,
        GEOGRAPHBG.имаХарактеренРастителенВид, GEOGRAPHBG.имаХарактеренЖивотинскиВид,
        GEOGRAPHBG.принадлежиКъмКлиматичнаЗона
    }:
        return 9

    # Най-общи или контекстуални (като "държава България")
    if p_uri == GEOGRAPHBG.държава:
        return 10

    # Свойства, свързани с дефиницията на КоличественаСтойност
    if p_uri == GEOGRAPHBG.имаСтойност or p_uri == GEOGRAPHBG.имаМернаЕдиница:
        return 98

    # Схематични свойства
    if str(p_uri).startswith(str(RDFS)) or str(p_uri).startswith(str(OWL)):
        return 99

    if str(p_uri).startswith(str(GEOGRAPHBG)):
        return 15

    return 20

def retrieve_and_rerank_triples(collection: chromadb.Collection, user_query: str, mtd: Dict[str, Any], graph_for_labels: Optional[Graph] = None, top_k_initial: int = TOP_K * 15, final_top_k: int = TOP_K) -> List[dict]:
    logging.info(f"Retrieving top {top_k_initial} for query: '{user_query}'")

    query_for_embedding = user_query
    user_query_lower = user_query.lower()

    if "българия" in user_query_lower:
        temp_query = user_query_lower.replace("в българия", "").replace("българия", "").strip()
        if len(temp_query.split()) > 1 or (len(temp_query.split()) == 1 and len(temp_query) > 3):
            query_for_embedding = temp_query
            logging.info(f"Original query: '{user_query}', Modified for embedding: '{query_for_embedding}'")

    query_embedding = embed(mtd, query_for_embedding)

    try:
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k_initial,
            include=["metadatas", "documents", "distances"],
        )

        all_metadatas = results.get("metadatas", [[]])[0]
        all_documents = results.get("documents", [[]])[0]
        all_distances = results.get("distances", [[]])[0]

        if not all_metadatas:
            return []

        combined_results = []
        for i in range(len(all_metadatas)):
            combined_results.append({
                "metadata": all_metadatas[i],
                "document": all_documents[i],
                "distance": all_distances[i]
            })

        def custom_sort_key(item):
            distance = item["distance"]
            metadata = item["metadata"]
            document_lower = item["document"].lower()
            predicate_importance_score = metadata.get("predicate_importance", 99) * 1.5
            penalty_score = 0.0
            predicate_uri_str = metadata.get("predicate_uri")
            object_uri_str = metadata.get("object_uri")

            is_country_bulgaria_triple = predicate_uri_str == str(GEOGRAPHBG.държава) and \
                                         object_uri_str == str(GEOGRAPHBG.България)

            if is_country_bulgaria_triple:
                if not (
                        "държава" in user_query_lower or "българия" in user_query_lower or "къде се намира" in user_query_lower):
                    if any(kw in user_query_lower for kw in ["височина", "дължина", "площ", "колко е", "най-", "тип"]):
                        penalty_score += 60
                    else:
                        penalty_score += 30

            bonus_score = 0.0

            quantity_keywords = ["колко", "висок", "дълъг", "площ", "дълбок", "обем", "население", "дебит",
                                 "температура", "капацитет", "най-"]
            is_quantity_related_query = any(kw in user_query_lower for kw in quantity_keywords)

            if is_quantity_related_query and predicate_uri_str in {str(uri) for uri in ONT_QUANTITY_PROPERTIES_URIS}:
                bonus_score -= 60
                subject_types_str = metadata.get("subject_types", "").lower()
                for query_word in user_query_lower.split():
                    if len(query_word) > 3 and query_word in subject_types_str and query_word not in ["река", "планина", "езеро", "връх"]:
                        if query_word in document_lower:
                            bonus_score -= 5
                            break

            query_non_stopwords = [
                kw for kw in user_query_lower.split()
                if kw not in {"в", "на", "са", "кои", "е", "с", "и", "или", "до", "от", "за", "под", "над",
                              "през"} and len(kw) > 2
            ]
            keyword_match_bonus = 0
            for kw in query_non_stopwords:
                if kw in document_lower:
                    keyword_match_bonus -= (len(kw) * 2)

            bonus_score += keyword_match_bonus
            scaled_distance = distance * 0.8
            final_score = scaled_distance + predicate_importance_score + penalty_score + bonus_score

            if graph_for_labels and metadata.get('subject_uri'):
                s_readable = get_readable_label(graph_for_labels, URIRef(metadata['subject_uri'])) if metadata.get(
                    'subject_uri') else "N/A"
                p_readable = get_readable_label(graph_for_labels,
                                                URIRef(predicate_uri_str)) if predicate_uri_str else "N/A"
                logging.debug(
                    f"RankEval: Doc='{item['document'][:60]}...' | "
                    f"Dist={distance:.1f} (Eff:{scaled_distance:.1f}) | "
                    f"PredImp={metadata.get('predicate_importance')}(Score:{predicate_importance_score:.1f}) | "
                    f"Penalty={penalty_score:.1f} | Bonus={bonus_score:.1f} (KW:{keyword_match_bonus:.1f}) | FinalScore={final_score:.1f} | "
                    f"S='{s_readable}', P='{p_readable}'"
                )
            return final_score

        combined_results.sort(key=custom_sort_key)

        reranked_metadatas = [res["metadata"] for res in combined_results[:final_top_k]]

        logging.info(f"Retrieved and re-ranked. Initial: {len(all_metadatas)}, Final: {len(reranked_metadatas)}.")
        if graph_for_labels:
            for i, res in enumerate(combined_results[:final_top_k + 5]):
                s_uri_str = res['metadata'].get('subject_uri')
                s_readable = get_readable_label(graph_for_labels, URIRef(s_uri_str)) if s_uri_str else "N/A"
                p_uri_str = res['metadata'].get('predicate_uri')
                p_readable = get_readable_label(graph_for_labels, URIRef(p_uri_str)) if p_uri_str else "N/A"
                final_score_val = custom_sort_key(res)
                logging.info(
                    f"  Top Reranked {i + 1}: \"{res['document']}\" (Score: {final_score_val:.2f}) "
                    f"S: {s_readable}, P: {p_readable}, Imp: {res['metadata'].get('predicate_importance')}"
                )
        return reranked_metadatas

    except Exception as e:
        logging.error(f"Error querying or reranking ChromaDB results: {e}", exc_info=True)
        return []


def construct_prompt_with_context(reasoning_client: OpenAI, sparql_endpoint: str, graph: Graph, collection: chromadb.Collection, user_query: str, mtd: Dict[str, Any],) -> str:
    """
    Retrieves relevant metadata, applies graph search and filtering,
    and constructs the final prompt for an LLM.

    Args:
        graph: The rdflib Graph object.
        collection: The ChromaDB collection.
        user_query: The user's query.
        mtd: Dictionary containing the loaded model, tokenizer, and device.

    Returns:
        The constructed prompt string including context.
    """
    ls = []
    st = time.time()
    retrieved_metadata = retrieve_and_rerank_triples(collection, user_query, mtd, top_k_initial=TOP_K * 30, final_top_k=TOP_K)
    if not retrieved_metadata:
        logging.warning("No metadata retrieved from vector search. Context will be empty for SPARQL.")
    ls.append(float(f'{time.time() - st:0.2f}'))
    print(f'* retrieving metadata took: {time.time() - st:0.2f}s')
    st = time.time()
    expanded_semantic_triples = search_graph_sparql(sparql_endpoint, graph, retrieved_metadata, max_hops=2)
    if not expanded_semantic_triples:
        logging.warning("No triples after SPARQL expansion. Context will be empty for reasoning model.")
    ls.append(float(f'{time.time() - st:0.2f}'))
    print(f'* expanding context took: {time.time() - st:0.2f}s')
    st = time.time()
    filtered_triples = filter_with_reasoning_model(reasoning_client, expanded_semantic_triples, user_query)
    ls.append(float(f'{time.time() - st:0.2f}'))
    print(f'* filtering triples took: {time.time() - st:0.2f}s')

    context = "\n".join([f"{i+1}. {triple}" for i, triple in enumerate(filtered_triples)]) if filtered_triples else "Няма намерен специфичен контекст."

    prompt = (
        f"Използвай САМО предоставения контекст, за да отговориш на въпроса.\n\n"
        f"Контекст:\n"
        f"---------------------\n"
        f"{context}\n"
        f"---------------------\n\n"
        f"Въпрос:\n"
        f"---------------------\n"
        f"{user_query}\n"
        f"---------------------\n\n"
        f"Отговор: (САМО С ГОРНИЯ КОНТЕКСТ - ако няма достатъчен контекст, отговори с '-')"
    )
    return prompt, ls


if __name__ == "__main__":
    log_file_path = f'logs/experiment_run_{datetime.datetime.now().strftime("%H_%M_%S-%d_%m_%y")}.log'
    if not os.path.exists(os.path.dirname(log_file_path)):
        pathlib.Path(os.path.dirname(log_file_path)).mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s',
        filename=log_file_path,
        filemode='w'
    )

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)

    logging.getLogger('').addHandler(console_handler)
    logging.info(f"Logging configured. Detailed output will be saved to {log_file_path}")

    try:
        if not OPENAI_API_KEY:
            raise ValueError("OpenAI API key is not set in constants.py or environment variables.")
        reasoning_client = OpenAI(api_key=OPENAI_API_KEY)
    except Exception as e:
        logging.warning(f"Failed to initialize OpenAI client: {e}")
        logging.warning("Please ensure the OpenAI API key is configured correctly.")
        exit(1)

    chroma_client = chromadb.Client()
    print("Loading embedding model and tokenizer...")
    model_tokenizer_device = load_model_and_tokenizer()

    RUN_INGESTION = True
    graph = None
    collection = None

    if RUN_INGESTION:
        try:
            graph, collection = embed_and_ingest_data(chroma_client, TTL_FILE_PATH, model_tokenizer_device)
            print("--- Data processing, embedding and ingestion finished. ---")
        except FileNotFoundError as e:
            print(e)
            exit()
        except Exception as e:
            print(f"An error occurred during data ingestion: {e}")
            exit()
    else:
        print("--- Skipping Data Ingestion ---")
        try:
            graph = load_data(TTL_FILE_PATH)
            collection = chroma_client.get_collection(name=CHROMA_COLLECTION_NAME)
            print(f"Using existing ChromaDB collection '{CHROMA_COLLECTION_NAME}' with {collection.count()} items.")
        except FileNotFoundError as e:
            print(f"TTL file not found at {TTL_FILE_PATH}, cannot proceed without graph data.")
        except Exception as e:
            logging.warning(f"Error accessing existing collection '{CHROMA_COLLECTION_NAME}' or loading graph: {e}")
            logging.warning("Please ensure the collection exists and TTL file is available, or set RUN_INGESTION=True.")

    if graph is None or collection is None:
        print("Graph or Collection could not be loaded/initialized. Exiting.")

    input_file = 'testset/demo.jsonl'
    output_file = 'testset/demo_output.jsonl'
    try:
        with jsonlines.open(input_file) as input_reader:
            with jsonlines.open(output_file, mode='w') as outfile:
                i = 0
                for exam in input_reader:
                    query = create_prompt(exam['question'], exam['options'])
                    print(query)
                    final_prompt, times_ls = construct_prompt_with_context(reasoning_client, GRAPHDB_REPO_ENDPOINT, graph, collection, query, model_tokenizer_device)
                    print("\n--- Constructed Prompt for LLM ---")
                    print(final_prompt)
                    print("--- End of Prompt ---")
                    st = time.time()
                    bggpt_response_data_no_context = generate_single_response_bggpt(query)
                    print(f"Zero-shot:\n {bggpt_response_data_no_context['response']}")
                    print(f'* Zero-shot took: {time.time() - st:0.2f} s')
                    st = time.time()
                    bggpt_response_data = generate_single_response_bggpt(final_prompt)
                    times_ls.append(float(f'{time.time() - st:0.2f}'))
                    print(f"GraphRAG response:\n {bggpt_response_data['response']}")
                    print(f'* retrieving metadata took: {times_ls[0]} s')
                    print(f'* expanding triples took: {times_ls[1]} s')
                    print(f'* filtering context took: {times_ls[2]} s')
                    print(f'* BgGPT generation took: {times_ls[3]} s')
                    print(f'* GraphRAG took: {sum(times_ls)} s')
                    exam['bggpt_kg_rag_answer'] = bggpt_response_data['response']
                    print(f'iter: {i}')
                    i = i + 1
                    outfile.write(exam)
    except Exception as e:
        logging.warning(f"An error occurred during the RAG pipeline execution: {e}")

    print("\n--- RAG Pipeline Script Finished ---")
