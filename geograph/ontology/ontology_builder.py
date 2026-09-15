import logging
import os
import re
import shutil
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from openai import OpenAI
from rdflib import OWL, RDF, RDFS, Graph, Namespace, URIRef

from geograph.config import PipelineConfig

logger = logging.getLogger(__name__)

PREFIXES = """\
@base <http://example.org/geographbg#> .
@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .
@prefix geographbg: <http://example.org/geographbg#> .
"""

SYSTEM_PROMPT = (
    "You are an expert RDF/OWL ontology engineer. Your task is to extract structured "
    "knowledge from Wikipedia content and output valid RDF triples in Turtle format. "
    "You may propose new classes and properties when the existing schema is insufficient, "
    "but you must clearly separate schema extensions from instance data."
)

EXTRACTION_PROMPT = """\
You are given the current state of an OWL ontology and a Wikipedia page. Your job is to:

1. Extract factual information from the page as RDF triples.
2. If the existing schema lacks classes or properties needed to represent the content,
   you MAY propose new ones — but keep them minimal and consistent with the existing
   hierarchy and naming conventions.

Output format — TWO clearly labeled sections of valid Turtle (no explanations outside):

```turtle
# === SCHEMA EXTENSIONS ===
# New classes (with rdfs:subClassOf linking to existing hierarchy) and/or
# new properties (with rdfs:domain, rdfs:range, rdfs:label).
# If no extensions are needed, leave this section empty (just the comment header).

# === INSTANCE DATA ===
# New individuals and their properties, using existing + any newly proposed schema.
# Reuse known instance IRIs from the list below when referring to existing entities.
# For new instances: use geographbg: prefix, CamelCase local names, rdfs:label with @bg.
# For quantitative values: use the КоличественаСтойност blank-node pattern shown below.
```

Constraints:
- Output ONLY valid Turtle. No markdown, no prose outside the turtle blocks.
- Do NOT redefine existing classes, properties, or instances.
- Every new class MUST have rdfs:subClassOf linking it to an existing class.
- Every new property MUST have rdfs:label, and ideally rdfs:domain + rdfs:range.
- Every new instance MUST have `a <Type>` and `rdfs:label "..."@bg`.
- Use the КоличественаСтойност pattern for ALL numeric measurements:
  ```
  geographbg:Entity geographbg:someProperty [
      a geographbg:КоличественаСтойност ;
      geographbg:имаСтойност "123.0"^^xsd:decimal ;
      geographbg:имаМернаЕдиница geographbg:UnitName
  ] .
  ```

{schema_section}

Known Instance Local Names (reuse these IRIs, e.g., geographbg:Марица):
{instance_names}

Wikipedia Page Title: {page_title}

Wikipedia Page Content:
{page_content}

Output (valid Turtle only):
{prefixes}
"""


@dataclass
class OntologySnapshot:
    schema_turtle: str = ""
    instance_names: List[str] = field(default_factory=list)
    class_count: int = 0
    property_count: int = 0
    instance_count: int = 0


@dataclass
class ExtractionResult:
    page_title: str
    raw_output: str
    triples_added: int = 0
    new_classes: List[str] = field(default_factory=list)
    new_properties: List[str] = field(default_factory=list)
    new_instances: List[str] = field(default_factory=list)
    validation_errors: List[str] = field(default_factory=list)
    success: bool = False


class OntologyBuilder:
    """Incrementally builds an OWL ontology by processing text through an LLM."""

    def __init__(
        self,
        ontology_path: str,
        api_key: str = "",
        namespace: str = "http://example.org/geographbg#",
        model: str = "gpt-4",
        max_tokens: int = 4000,
    ):
        self.ontology_path = ontology_path
        self.namespace = Namespace(namespace)
        self.model = model
        self.max_tokens = max_tokens
        self.client = OpenAI(api_key=api_key) if api_key else None
        self.graph = Graph()

        if os.path.exists(ontology_path):
            self.graph.parse(ontology_path, format="turtle", publicID=namespace)
            logger.info(
                "Loaded ontology: %d triples from %s", len(self.graph), ontology_path
            )
        else:
            logger.warning("Ontology file not found at %s. Starting with empty graph.", ontology_path)

    def snapshot(self, relevant_text: str = "", max_instances: int = 300) -> OntologySnapshot:
        """Extract current schema and instance names from the live graph."""
        snap = OntologySnapshot()

        schema_graph = Graph()
        schema_graph.bind("geographbg", self.namespace)
        schema_types = {
            OWL.Class, OWL.ObjectProperty, OWL.DatatypeProperty,
            OWL.SymmetricProperty, OWL.TransitiveProperty,
            OWL.FunctionalProperty, OWL.InverseFunctionalProperty,
            OWL.AnnotationProperty, RDFS.Class, RDF.Property,
        }
        unit_class = self.namespace.МернаЕдиница

        for s, p, o in self.graph:
            s_str = str(s)
            if not s_str.startswith(str(self.namespace)):
                continue

            is_schema = False
            for s_type in self.graph.objects(s, RDF.type):
                if s_type in schema_types or s_type == unit_class:
                    is_schema = True
                    break

            # Also include subClassOf, inverseOf, domain, range triples
            if p in {RDFS.subClassOf, OWL.inverseOf, RDFS.domain, RDFS.range}:
                is_schema = True

            if is_schema:
                schema_graph.add((s, p, o))

        snap.schema_turtle = schema_graph.serialize(format="turtle")
        snap.class_count = len(list(schema_graph.subjects(RDF.type, OWL.Class)))
        snap.property_count = len(list(schema_graph.subjects(RDF.type, OWL.ObjectProperty))) + \
                              len(list(schema_graph.subjects(RDF.type, OWL.DatatypeProperty)))

        instance_names = set()
        for s in self.graph.subjects(RDF.type, None):
            if not isinstance(s, URIRef) or not str(s).startswith(str(self.namespace)):
                continue
            s_types = set(self.graph.objects(s, RDF.type))
            if s_types & schema_types:
                continue
            local = str(s).replace(str(self.namespace), "")
            if local:
                instance_names.add(local)

        snap.instance_names = self._relevant_instance_names(instance_names, relevant_text, max_instances)
        snap.instance_count = len(snap.instance_names)

        logger.info(
            "Snapshot: %d classes, %d properties, %d instances.",
            snap.class_count, snap.property_count, snap.instance_count,
        )
        return snap

    @staticmethod
    def _relevant_instance_names(instance_names, relevant_text: str, max_instances: int) -> List[str]:
        all_names = sorted(instance_names)
        if not relevant_text:
            return all_names[:max_instances]

        text_lower = relevant_text.lower()
        word_re = re.compile(r"[А-ЯA-Z][а-яa-z]*")

        def is_relevant(name: str) -> bool:
            return any(len(word) >= 4 and word.lower() in text_lower for word in word_re.findall(name))

        relevant = [n for n in all_names if is_relevant(n)]
        return relevant[:max_instances]

    def process_batch(
        self,
        pages: List[Dict[str, str]],
        backup: bool = True,
    ) -> List[ExtractionResult]:
        """Process a batch of Wikipedia pages, extracting and merging triples."""
        if backup and os.path.exists(self.ontology_path):
            backup_path = self.ontology_path + ".backup"
            shutil.copy2(self.ontology_path, backup_path)
            logger.info("Backup saved to %s", backup_path)

        results = []
        for i, page in enumerate(pages):
            title = page.get("title", f"Untitled-{i}")
            content = page.get("content", "")

            if not content.strip():
                logger.warning("Skipping empty page: %s", title)
                results.append(ExtractionResult(page_title=title, raw_output="", validation_errors=["Empty content"]))
                continue

            logger.info("Processing page %d/%d: %s", i + 1, len(pages), title)
            result = self._process_single_page(title, content)
            results.append(result)

            if result.success:
                logger.info(
                    "  Added %d triples. New classes: %s, New properties: %s, New instances: %s",
                    result.triples_added, result.new_classes, result.new_properties, result.new_instances,
                )
            else:
                logger.warning("  Failed: %s", result.validation_errors)

        total_added = sum(r.triples_added for r in results)
        successful = sum(1 for r in results if r.success)
        logger.info(
            "Batch complete: %d/%d pages succeeded, %d total triples added.",
            successful, len(pages), total_added,
        )
        return results

    def build_extraction_prompt(self, title: str, content: str) -> str:
        snap = self.snapshot(relevant_text=f"{title} {content}")
        schema_section = f"Current Ontology Schema:\n{snap.schema_turtle}"
        instance_names_str = "\n".join(f"- {name}" for name in snap.instance_names)
        if not instance_names_str:
            instance_names_str = "(No instances yet)"

        return EXTRACTION_PROMPT.format(
            schema_section=schema_section,
            instance_names=instance_names_str,
            page_title=title,
            page_content=content,
            prefixes=PREFIXES,
        )

    def process_manual(self, title: str, raw_turtle: str) -> ExtractionResult:
        return self._validate_and_merge(title, raw_turtle)

    def _process_single_page(self, title: str, content: str) -> ExtractionResult:
        """Process one page: prompt LLM → validate → merge."""
        prompt = self.build_extraction_prompt(title, content)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=self.max_tokens,
                temperature=0.0,
            )
            raw = response.choices[0].message.content
        except Exception as e:
            result = ExtractionResult(page_title=title, raw_output="")
            result.validation_errors.append(f"LLM API error: {e}")
            return result

        return self._validate_and_merge(title, raw)

    def _validate_and_merge(self, title: str, raw: str) -> ExtractionResult:
        """Steps shared by both the API and manual extraction paths:
        strip markdown fences, validate as Turtle (with a prefix-repair
        fallback), classify new schema/instances, and merge into the graph."""
        result = ExtractionResult(page_title=title, raw_output=raw)

        # Step 3: Extract Turtle from response (strip markdown fences if present)
        turtle_text = self._extract_turtle(raw)

        # Step 4: Validate by parsing with rdflib
        triples_before = len(self.graph)
        new_graph = Graph()
        try:
            new_graph.parse(data=turtle_text, format="turtle", publicID=str(self.namespace))
        except Exception as e:
            result.validation_errors.append(f"Invalid Turtle: {e}")
            # Try a best-effort repair: add prefixes if missing
            turtle_with_prefixes = PREFIXES + "\n" + turtle_text
            try:
                new_graph = Graph()
                new_graph.parse(data=turtle_with_prefixes, format="turtle", publicID=str(self.namespace))
                logger.info("  Recovered by prepending prefixes.")
            except Exception as e2:
                result.validation_errors.append(f"Still invalid after prefix repair: {e2}")
                return result

        # Step 5: Classify and merge new triples
        schema_types = {OWL.Class, OWL.ObjectProperty, OWL.DatatypeProperty}
        new_classes = []
        new_properties = []
        new_instances = []

        for s, p, o in new_graph:
            # Skip prefix/base declarations (blank subjects, etc.)
            if not isinstance(s, URIRef):
                # Still add the triple (e.g., blank node for QuantitativeValue)
                self.graph.add((s, p, o))
                continue

            # Check if this subject is a new schema element
            if p == RDF.type and o in schema_types:
                local = str(s).replace(str(self.namespace), "")
                # Only count as new if not already in graph
                existing_types = set(self.graph.objects(s, RDF.type))
                if o not in existing_types:
                    if o == OWL.Class:
                        new_classes.append(local)
                    else:
                        new_properties.append(local)

            # Check for new instances
            if p == RDF.type and o not in schema_types and isinstance(o, URIRef):
                local = str(s).replace(str(self.namespace), "")
                existing_types = set(self.graph.objects(s, RDF.type))
                if o not in existing_types and local not in new_instances:
                    new_instances.append(local)

            self.graph.add((s, p, o))

        result.triples_added = len(self.graph) - triples_before
        result.new_classes = new_classes
        result.new_properties = new_properties
        result.new_instances = new_instances
        result.success = True
        return result

    def save(self, output_path: Optional[str] = None) -> str:
        """Serialize the current graph back to a Turtle file."""
        path = output_path or self.ontology_path
        self.graph.serialize(destination=path, format="turtle")
        logger.info("Ontology saved: %d triples → %s", len(self.graph), path)
        return path

    @staticmethod
    def _extract_turtle(raw_output: str) -> str:
        """Strip markdown code fences from LLM output to get raw Turtle."""
        text = raw_output.strip()

        if "```turtle" in text:
            parts = text.split("```turtle")
            turtle_parts = []
            for part in parts[1:]:
                if "```" in part:
                    turtle_parts.append(part[:part.index("```")])
                else:
                    turtle_parts.append(part)
            return "\n".join(turtle_parts).strip()

        if text.startswith("```"):
            text = text[3:]
            if text.endswith("```"):
                text = text[:-3]
            return text.strip()

        return text


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")

    ontology_file = sys.argv[1] if len(sys.argv) > 1 else "ontologies/huge_ontology_v3.ttl"
    config = PipelineConfig.from_constants()

    builder = OntologyBuilder(ontology_file, api_key=config.reasoning.api_key)
    snap = builder.snapshot()

    print(f"\nOntology: {ontology_file}")
    print(f"  Classes:    {snap.class_count}")
    print(f"  Properties: {snap.property_count}")
    print(f"  Instances:  {snap.instance_count}")
    print(f"\n  Sample instances: {snap.instance_names[:15]}")