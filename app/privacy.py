"""Anonymiser. Used ONLY before text is sent to a cloud model."""
import re

_PATTERNS = [
    (re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"), "[EMAIL]"),
    (re.compile(r"https?://\S+"), "[LINK]"),
    (re.compile(r"\b(?:\+?\d[\s-]?){10,16}\b"), "[NUMBER]"),
]


def scrub(text: str, names):
    for rx, tag in _PATTERNS:
        text = rx.sub(tag, text)
    for i, n in enumerate(sorted(names, key=len, reverse=True), 1):
        text = re.sub(re.escape(n), f"[PERSON{i}]", text, flags=re.I)
    return text
