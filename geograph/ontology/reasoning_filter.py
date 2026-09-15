import ast
import logging
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

from openai import OpenAI

from geograph.config import OntologyConfig, ReasoningConfig

logger = logging.getLogger(__name__)


@dataclass
class FilterResult:
    triples: List[str]
    n_prefiltered: int
    llm_used: bool


class ReasoningFilter:
    def __init__(self, config: ReasoningConfig, ont_config: OntologyConfig):
        self.config = config
        self.ont = ont_config
        self.client = OpenAI(api_key=config.api_key)

    def filter(self, semantic_triples: List[str], user_query: str) -> FilterResult:
        logger.info("Reasoning filter input: %d triples.", len(semantic_triples))
        if not semantic_triples:
            return FilterResult(triples=[], n_prefiltered=0, llm_used=False)

        pre_filtered = [t for t in semantic_triples if self._is_valid(t)]
        logger.info("After pre-filtering: %d triples.", len(pre_filtered))
        if not pre_filtered:
            return FilterResult(triples=[], n_prefiltered=0, llm_used=False)

        if not self.config.enabled:
            return FilterResult(triples=pre_filtered, n_prefiltered=len(pre_filtered), llm_used=False)

        triples, llm_used = self._apply_reasoning(pre_filtered, user_query)
        return FilterResult(triples=triples, n_prefiltered=len(pre_filtered), llm_used=llm_used)

    def _is_valid(self, triple: str) -> bool:
        if not triple:
            return False
        if self.ont.missing_value_string in triple:
            return False
        for ending in self.ont.unwanted_triple_endings:
            if triple.endswith(ending):
                return False
        if triple.endswith("държава България"):
            return False
        return True

    def _apply_reasoning(self, triples: List[str], user_query: str) -> Tuple[List[str], bool]:
        formatted = "\n".join(
            f'{i + 1}. "{triple}"' for i, triple in enumerate(triples)
        )
        prompt_content = self.config.filter_prompt_template.format(
            kg_triples=formatted,
            user_query=user_query,
        )

        try:
            response = self.client.responses.create(
                model=self.config.model,
                reasoning={"effort": self.config.effort},
                input=[
                    {"role": "system", "content": self.config.system_prompt},
                    {"role": "user", "content": prompt_content},
                ],
            )
            response_text = response.output_text
            logger.info("Reasoning model raw output:\n%s", response_text)
        except Exception as e:
            logger.error("Reasoning API error: %s. Returning pre-filtered triples.", e)
            return triples, False

        parsed = self._parse_response(response_text, triples)
        if parsed is None:
            return triples, False

        if parsed and len(parsed) < self.config.min_keep:
            parsed = self._top_up(parsed, triples)

        logger.info("Reasoning filter output: %d triples.", len(parsed))
        return parsed, True

    def _top_up(self, kept: List[str], pre_filtered: List[str]) -> List[str]:
        kept_set = set(kept)
        topped_up = list(kept)
        for triple in pre_filtered:
            if len(topped_up) >= self.config.min_keep:
                break
            if triple not in kept_set:
                topped_up.append(triple)
        logger.info(
            "Topped up filter output from %d to %d triples (min_keep=%d).",
            len(kept), len(topped_up), self.config.min_keep,
        )
        return topped_up

    def _parse_response(self, response_text: str, original_triples: List[str]) -> Optional[List[str]]:
        text = response_text.strip()

        # Strip code block markers
        if text.startswith("```python"):
            text = text[len("```python"):].strip()
        elif text.startswith("```"):
            text = text[3:].strip()
        if text.endswith("```"):
            text = text[:-3].strip()

        if not text.startswith("[") or not text.endswith("]"):
            logger.warning("Reasoning output not a list.")
            return None

        try:
            parsed = ast.literal_eval(text)
            if not isinstance(parsed, list):
                raise ValueError("Parsed result is not a list.")
        except (SyntaxError, ValueError, TypeError) as e:
            logger.error("Failed to parse reasoning output: %s.", e)
            return None

        filtered = []
        original_set = set(original_triples)
        for item in parsed:
            if not isinstance(item, str):
                continue
            cleaned = re.sub(r"^\d+\.\s*\"", '"', item).strip()
            if cleaned.startswith('"') and cleaned.endswith('"'):
                cleaned = cleaned[1:-1]
            if cleaned in original_set:
                filtered.append(cleaned)
            else:
                logger.warning("Reasoning model returned unknown triple: '%s'. Skipping.", cleaned)

        return filtered
