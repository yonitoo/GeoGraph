import re
from typing import Dict, List

_WORD_RE = re.compile(r"\w+", re.UNICODE)


def recall_proxy(context_texts: List[str], options: List[Dict], correct_answer: str) -> bool:
    """True if the correct option's text appears (case-insensitive) in the retrieved context."""
    correct_text = next(
        (o["text"] for o in options if o.get("label") == corrSect_answer), ""
    )
    if not correct_text:
        return False
    joined = " ".join(context_texts).lower()
    words = [w for w in _WORD_RE.findall(correct_text.lower()) if len(w) > 3]
    if not words:
        return correct_text.lower() in joined
    return sum(1 for w in words if w in joined) / len(words) >= 0.5
