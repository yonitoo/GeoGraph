from __future__ import annotations

import logging
import os
import re
import urllib.parse
from typing import Any, Dict, List, Optional, Set, Tuple

from rdflib import BNode, Graph, Literal, Node, OWL, RDF, RDFS, URIRef, XSD
from tqdm import tqdm

from geograph.config import OntologyConfig

logger = logging.getLogger(__name__)

_SCHEMA_TYPES_TO_IGNORE = {
    OWL.Class, RDFS.Class, OWL.Ontology, OWL.AnnotationProperty,
    OWL.ObjectProperty, OWL.DatatypeProperty, OWL.SymmetricProperty,
    OWL.TransitiveProperty, OWL.FunctionalProperty, OWL.InverseFunctionalProperty,
    RDF.Property, RDFS.Resource, OWL.Thing,
}

_SKIP_PREDICATES = {
    RDFS.comment, RDFS.subClassOf, RDFS.label, RDFS.domain, RDFS.range,
    OWL.sameAs, OWL.inverseOf, OWL.equivalentClass, OWL.disjointWith,
}

_IGNORED_SUBJECT_TYPES = {OWL.NamedIndividual, OWL.Thing, RDFS.Resource}
_IGNORED_OBJECT_TYPES = _IGNORED_SUBJECT_TYPES | {OWL.Class, RDFS.Class}


class RDFProcessor:
    def __init__(self, graph: Graph, ontology_config: OntologyConfig):
        self.graph = graph
        self.ont = ontology_config
        self._quantity_uris = ontology_config.quantity_property_uris
        self._value_uri = ontology_config.value_property_uri
        self._unit_uri = ontology_config.unit_property_uri
        self._schema_ignore = _SCHEMA_TYPES_TO_IGNORE | {
            ontology_config.namespace.КоличественаСтойност,
        }

    @classmethod
    def load(cls, ttl_path: str, ontology_config: OntologyConfig) -> RDFProcessor:
        if not os.path.exists(ttl_path):
            raise FileNotFoundError(f"TTL file not found: {ttl_path}")
        graph = Graph()
        logger.info("Loading RDF data from %s...", ttl_path)
        graph.parse(ttl_path, format="turtle", publicID=str(ontology_config.namespace))
        logger.info("Loaded %d triples from %s", len(graph), ttl_path)
        return cls(graph, ontology_config)

    def get_readable_label(self, resource: Any) -> str:
        if isinstance(resource, Literal):
            return self._label_for_literal(resource)
        if isinstance(resource, BNode):
            return "[празен възел]"
        if not isinstance(resource, URIRef):
            return str(resource)
        return self._label_for_uri(resource)

    def _label_for_literal(self, literal: Literal) -> str:
        if literal.datatype in {XSD.decimal, XSD.integer, XSD.float, XSD.double}:
            try:
                val = literal.value
                if val is None:
                    return self.ont.missing_value_string
                if isinstance(val, float) and val.is_integer():
                    return str(int(val))
                return str(val)
            except (TypeError, ValueError):
                return str(literal.value) if literal.value is not None else self.ont.missing_value_string
        return str(literal.value) if literal.value is not None else ""

    def _label_for_uri(self, uri: URIRef) -> str:
        try:
            label_lit = self.graph.value(subject=uri, predicate=RDFS.label)
            if label_lit and isinstance(label_lit, Literal):
                return str(label_lit.value)
        except Exception:
            pass

        try:
            full = str(uri)
            local = full.split("#")[-1] if "#" in full else full.split("/")[-1]
            if local:
                decoded = urllib.parse.unquote(local).replace("_", " ")
                spaced = re.sub(r"([А-ЯЯУЕИАОЪЬЮ])", r" \1", decoded).strip()
                cleaned = re.sub(r"\s+", " ", spaced).strip()
                return cleaned or decoded
        except Exception:
            pass
        return str(uri)

    def convert_triple_to_semantic(
        self,
        s: Node,
        p: Node,
        o: Node,
        processed_bnodes: Set[BNode],
        sparql_bnode_details: Optional[Dict[str, Dict[str, str]]] = None,
    ) -> Optional[str]:
        """Convert a single RDF triple to a human-readable semantic string."""
        if p in self._quantity_uris and isinstance(o, BNode):
            return self._convert_quantity_triple(s, p, o, processed_bnodes, sparql_bnode_details)

        if p == RDF.type:
            return self._convert_type_triple(s, o)

        if p in _SKIP_PREDICATES:
            return None

        if p in self._quantity_uris and isinstance(o, URIRef) and "Entity" in str(o):
            logger.warning(
                "Quantitative property %s with fallback URI object %s for %s. Skipping.",
                self.get_readable_label(p), self.get_readable_label(o), self.get_readable_label(s),
            )
            return None

        if isinstance(o, BNode):
            logger.debug("Skipping triple with unhandled BNode object: %s %s %s", s, p, o)
            return None

        s_label = self.get_readable_label(s)
        p_label = self.get_readable_label(p)
        o_label = self.get_readable_label(o)

        if s_label == str(s) or p_label == str(p):
            return None
        if isinstance(o, URIRef) and o_label == str(o):
            return None
        if isinstance(o, Literal) and not o_label and o.value is None:
            return None

        return f"{s_label} {p_label} {o_label}"

    def _convert_quantity_triple(
        self,
        s: Node,
        p: Node,
        o: BNode,
        processed_bnodes: Set[BNode],
        sparql_bnode_details: Optional[Dict[str, Dict[str, str]]] = None,
    ) -> Optional[str]:
        if o in processed_bnodes:
            return None

        value_str = self.ont.missing_value_string
        unit_label_str = ""

        bnode_key = str(o)
        used_sparql = False

        if sparql_bnode_details and bnode_key in sparql_bnode_details:
            details = sparql_bnode_details[bnode_key]
            value_str = details.get("value", self.ont.missing_value_string)
            unit_label_str = details.get("unit_label", "")
            unit_uri = details.get("unit_uri", "")
            used_sparql = True

            if not unit_label_str and unit_uri:
                readable = self.get_readable_label(URIRef(unit_uri))
                if readable != unit_uri and "#" not in readable and "/" not in readable:
                    unit_label_str = readable

            if value_str != self.ont.missing_value_string:
                try:
                    val_f = float(value_str)
                    value_str = str(int(val_f)) if val_f.is_integer() else str(val_f)
                except ValueError:
                    pass

        if not used_sparql:
            value_lit = self.graph.value(subject=o, predicate=self._value_uri) if self._value_uri else None
            unit_node = self.graph.value(subject=o, predicate=self._unit_uri) if self._unit_uri else None
            value_str = self.get_readable_label(value_lit) if value_lit else self.ont.missing_value_string
            unit_label_str = self.get_readable_label(unit_node) if unit_node else ""

        combined = f"{value_str} {unit_label_str}".strip()

        if value_str == self.ont.missing_value_string or combined == self.ont.missing_value_string:
            # A quantity with no resolvable value is worse than no fact at
            # all — never verbalize the placeholder itself (e.g. "Х има
            # население [липсва стойност] души"), unit or no unit.
            return None

        processed_bnodes.add(o)
        s_label = self.get_readable_label(s)
        p_label = self.get_readable_label(p)
        if s_label == str(s) or p_label == str(p):
            return None
        return f"{s_label} {p_label} {combined}"

    def _convert_type_triple(self, s: Node, o: Node) -> Optional[str]:
        if not isinstance(o, URIRef) or o in self._schema_ignore:
            return None
        s_label = self.get_readable_label(s)
        o_label = self.get_readable_label(o)
        if s_label == str(s) or o_label == str(o):
            return None
        return f"{s_label} е тип {o_label}"

    def process_triples(self) -> Tuple[List[str], List[Dict[str, Any]]]:
        """Process all graph triples into semantic strings with metadata."""
        semantic_strings: List[str] = []
        metadata_list: List[Dict[str, Any]] = []
        processed_bnodes: Set[BNode] = set()
        seen_texts: Set[str] = set()
        duplicates_skipped = 0

        logger.info("Processing triples into semantic strings...")
        for s, p, o in tqdm(self.graph, desc="Processing triples"):
            if isinstance(s, BNode):
                continue

            text = self.convert_triple_to_semantic(s, p, o, processed_bnodes)
            if not text:
                continue

            if text in seen_texts:
                duplicates_skipped += 1
                continue
            seen_texts.add(text)

            meta = self._build_metadata(s, p, o)
            semantic_strings.append(text)
            metadata_list.append(meta)

        logger.info(
            "Processed %d semantic fact strings (%d duplicate-text triples skipped).",
            len(semantic_strings), duplicates_skipped,
        )
        return semantic_strings, metadata_list

    def _build_metadata(self, s: Node, p: Node, o: Node) -> Dict[str, Any]:
        meta: Dict[str, Any] = {
            "subject_uri": str(s) if isinstance(s, URIRef) else "",
            "predicate_uri": str(p) if isinstance(p, URIRef) else "",
            "predicate_importance": get_predicate_importance(p, self.ont) if isinstance(p, URIRef) else 99,
        }

        if p in self._quantity_uris and isinstance(o, BNode):
            value_lit = self.graph.value(subject=o, predicate=self._value_uri) if self._value_uri else None
            unit_ref = self.graph.value(subject=o, predicate=self._unit_uri) if self._unit_uri else None

            meta["object_bnode_id"] = str(o)
            meta["object_uri"] = ""
            meta["object_value"] = str(value_lit.value) if value_lit and hasattr(value_lit, "value") else ""
            meta["object_datatype"] = str(value_lit.datatype) if value_lit and value_lit.datatype else str(XSD.decimal)
            meta["object_unit_uri"] = str(unit_ref) if unit_ref else ""
            meta["object_unit_label"] = self.get_readable_label(unit_ref) if unit_ref else ""
        else:
            meta["object_uri"] = str(o) if isinstance(o, URIRef) else ""
            meta["object_value"] = str(o.value) if isinstance(o, Literal) else ""
            meta["object_datatype"] = str(o.datatype) if isinstance(o, Literal) and o.datatype else ""
            if isinstance(o, BNode):
                meta["object_bnode_id"] = str(o)

        if isinstance(s, URIRef):
            s_types = sorted({
                self.get_readable_label(t)
                for t in self.graph.objects(s, RDF.type)
                if isinstance(t, URIRef) and t not in _IGNORED_SUBJECT_TYPES
            })
            if s_types:
                meta["subject_types"] = ", ".join(s_types)

        if isinstance(o, URIRef):
            o_types = sorted({
                self.get_readable_label(t)
                for t in self.graph.objects(o, RDF.type)
                if isinstance(t, URIRef) and t not in _IGNORED_OBJECT_TYPES
            })
            if o_types:
                meta["object_types"] = ", ".join(o_types)

        return meta


def get_predicate_importance(p_uri: URIRef, ont: OntologyConfig) -> int:
    ns = ont.namespace

    if p_uri in ont.quantity_property_uris:
        return 1
    if p_uri == RDF.type:
        return 2

    core_relations = {
        ns.находящСеВ, ns.извираОт, ns.вливаСеВ,
        ns.притокНа, ns.имаПриток, ns.найВисокВръхНа,
        ns.столица, ns.еСтолицаНа, ns.областенЦентър,
        ns.еОбластенЦентърНа, ns.общинскиЦентър, ns.еОбщинскиЦентърНа,
        ns.мястоНаВливане, ns.находящСеВПодножиетоНа,
        ns.находящСеВПоречиетоНа, ns.имаЯзовир,
        ns.еСъставящаЧастНа, ns.имаСъставящаЧаст,
    }
    if p_uri in core_relations:
        return 3

    secondary_relations = {
        ns.намираСеНаРека, ns.частОтПрироденПарк, ns.частОтКаскада,
        ns.странаНаПритока, ns.свързанС, ns.близоДо, ns.захранваСеОт,
    }
    if p_uri in secondary_relations:
        return 4

    descriptive_attrs = {
        ns.имаАлтернативноИме, ns.основанПрезГодина, ns.еЗавършенПрез,
        ns.имаТипСтена, ns.имаСтатутНаЗащитенаМестност,
        ns.обявенаСъсЗаповед, ns.официаленЕзик, ns.датаНаОбявяване,
        ns.датаНаЗавършване, ns.датаНаПровеждане,
    }
    if p_uri in descriptive_attrs:
        return 5

    contextual_relations = {
        ns.намираСеВЗемлищетоНа, ns.разположенВМестност,
        ns.проведеноЗа, ns.обхващаПериодОтДо, ns.строител,
        ns.частОтИнфраструктуренОбект,
    }
    if p_uri in contextual_relations:
        return 6

    general_relations = {
        ns.граничиС, ns.преминаваПрез, ns.протичаПрез,
        ns.свързва, ns.стопанисваСеОт, ns.образува, ns.използваСеЗа,
    }
    if p_uri in general_relations:
        return 7

    long_text_props = {
        ns.имаОписание, ns.имаИстория, ns.имаЕтимология,
        ns.имаКлимат, ns.имаГеология, ns.имаФлора,
        ns.имаФауна, ns.частОтИсториятаНа,
    }
    if p_uri in long_text_props:
        return 8

    classification_props = {
        ns.имаПолезноИзкопаемо, ns.имаТипПочва,
        ns.имаХарактеренРастителенВид, ns.имаХарактеренЖивотинскиВид,
        ns.принадлежиКъмКлиматичнаЗона,
    }
    if p_uri in classification_props:
        return 9

    if p_uri == ns.държава:
        return 10

    if p_uri in {ns.имаСтойност, ns.имаМернаЕдиница}:
        return 98

    if str(p_uri).startswith(str(RDFS)) or str(p_uri).startswith(str(OWL)):
        return 99

    if str(p_uri).startswith(str(ns)):
        return 15

    return 20