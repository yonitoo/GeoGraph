import json
import logging
import re
from typing import Optional

import requests

from geograph.config import LLMConfig

logger = logging.getLogger(__name__)

_CYRILLIC_MAP = str.maketrans("АВСД", "ABCD")
_MCQ_RE = re.compile(r"\b([ABCD])\b|\b([АВСД])\b")


def extract_mcq_letter(text: str) -> Optional[str]:
    for m in _MCQ_RE.finditer(text):
        char = (m.group(1) or m.group(2)).translate(_CYRILLIC_MAP)
        if char in "ABCD":
            return char
    return None


class BgGPTClient:
    """Client for the BgGPT (Gemma-2 based) completion API."""

    def __init__(self, config: LLMConfig):
        self.config = config

    def generate(self, prompt: str, max_tokens: Optional[int] = None) -> str:
        """Send a prompt and return the accumulated response text."""
        max_tokens = max_tokens or self.config.max_tokens
        inst_prompt = f"<bos><start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"

        payload = {
            "max_tokens": max_tokens,
            "model": self.config.model,
            "stop": ["<end_of_turn>", "<eos>"],
            "temperature": self.config.temperature,
            "top_k": self.config.top_k,
            "repetition_penalty": self.config.repetition_penalty,
            "stream": True,
            "prompt": inst_prompt,
        }
        headers = {
            "apikey": self.config.api_key,
            "accept": "application/json",
            "content-type": "application/json",
        }

        response = requests.post(
            self.config.api_url,
            headers=headers,
            json=payload,
            stream=True,
            timeout=120,
        )

        if response.status_code != 200:
            raise RuntimeError(
                f"BgGPT request failed ({response.status_code}): {response.text}"
            )

        accumulated = []
        for line in response.iter_lines():
            if not line:
                continue
            decoded = line.decode("utf-8").strip()
            if not decoded.startswith("data: "):
                continue
            data = decoded[len("data: "):]
            if data == "[DONE]":
                break
            try:
                chunk = json.loads(data)
                content = chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
                if content:
                    accumulated.append(content)
            except json.JSONDecodeError:
                continue

        return "".join(accumulated).strip()

    def generate_answer_letter(self, prompt: str) -> Optional[str]:
        """Generate a single-letter MCQ answer (A/B/C/D)."""
        inst_prompt = f"<bos><start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"

        payload = {
            "max_tokens": 2,
            "model": self.config.model,
            "stop": ["<end_of_turn>", "<eos>"],
            "temperature": self.config.temperature,
            "top_k": self.config.top_k,
            "repetition_penalty": self.config.repetition_penalty,
            "stream": True,
            "prompt": inst_prompt,
        }
        headers = {
            "apikey": self.config.api_key,
            "accept": "application/json",
            "content-type": "application/json",
        }

        response = requests.post(
            self.config.api_url,
            headers=headers,
            json=payload,
            stream=True,
            timeout=60,
        )

        if response.status_code != 200:
            raise RuntimeError(
                f"BgGPT request failed ({response.status_code}): {response.text}"
            )

        accumulated = []
        for line in response.iter_lines():
            if not line:
                continue
            decoded = line.decode("utf-8").strip()
            if not decoded.startswith("data: "):
                continue
            data = decoded[len("data: "):]
            if data == "[DONE]":
                break
            try:
                chunk = json.loads(data)
                content = chunk.get("choices", [{}])[0].get("text", "") or \
                          chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
                if content:
                    accumulated.append(content)
                    letter = extract_mcq_letter(content)
                    if letter:
                        return letter
            except json.JSONDecodeError:
                continue

        return extract_mcq_letter("".join(accumulated))


class BgGPTChatClient:
    """Client for the BgGPT v3 (Gemma-3) chat-completions REST API."""

    def __init__(self, config: LLMConfig):
        self.config = config

    def _post(self, prompt: str, max_tokens: int) -> str:
        payload = {
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.config.temperature,
            "max_tokens": max_tokens,
            "top_k": self.config.top_k,
            "top_p": 0.95,
            "repetition_penalty": self.config.repetition_penalty,
            "stream": False,
        }
        headers = {
            "apikey": self.config.api_key,
            "accept": "application/json",
            "content-type": "application/json",
        }
        response = requests.post(
            self.config.api_url,
            headers=headers,
            json=payload,
            timeout=120,
        )
        if response.status_code != 200:
            raise RuntimeError(
                f"BgGPT v3 request failed ({response.status_code}): {response.text}"
            )
        data = response.json()
        return data["choices"][0]["message"]["content"].strip()

    def generate(self, prompt: str, max_tokens: Optional[int] = None) -> str:
        return self._post(prompt, max_tokens or self.config.max_tokens)

    def generate_answer_letter(self, prompt: str) -> Optional[str]:
        text = self._post(prompt, max_tokens=10)
        return extract_mcq_letter(text)
