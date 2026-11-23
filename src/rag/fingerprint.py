"""Fingerprints of each collection, used by the output check to catch leaks.

At ingest time we pull "sensitive-looking" terms out of every document:
  - e-mail addresses                       hr@brightnest.co.id
  - phone numbers                          +62 811-0000-9999   (stored as digits, 62 -> 0)
  - extensions                             ext. 2101 / extension 2101 / ekstensi 2101
  - large numbers and money amounts        Rp16,000,000 / 16.000.000   (stored as digits, 5+ digits)
  - names written in capitals              Sari Wulandari, Asuransi Sehat Bersama, BrightHub

Each term is hashed (SHA-256), and the hashes are saved per collection in storage/security/fingerprints.json.
At answer time we extract the same kinds of terms from the answer, hash them, and check them against the
collections the user may NOT read (minus anything that also appears in a collection they may read, so public
facts never trigger it). Storing only hashes means the public assistant can recognise an internal e-mail
address without holding a readable list of internal data.

It is a safety net with simple rules: it catches copied contact details, figures and names, not paraphrases.
"""

import hashlib
import json
import re
from pathlib import Path

_EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
_PHONE = re.compile(r"(?:\+62[\s-]?|\b0)\d{2,3}[\s-]?\d{3,4}[\s-]?\d{3,4}\b")
_EXTENSION = re.compile(r"\b(?:ext\.?|extension|ekstensi)\s*(\d{3,5})\b", re.IGNORECASE)
_NUMBER = re.compile(r"\b\d{1,3}(?:[.,]\d{3})+\b|\b\d{5,}\b")
_NAME = re.compile(r"\b[A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,}){1,2}\b")
_CAMEL = re.compile(r"\b[A-Z][a-z]+[A-Z][A-Za-z]+\b")  # BrightHub, ServiceDesk, GlobalConnect


def extract_terms(text: str) -> set[str]:
    terms: set[str] = set()
    for email in _EMAIL.findall(text):
        terms.add("email:" + email.lower().rstrip("."))
    for phone in _PHONE.findall(text):
        digits = re.sub(r"\D", "", phone)
        terms.add("phone:" + ("0" + digits[2:] if digits.startswith("62") else digits))
    without_phones = _PHONE.sub(" ", text)
    for ext in _EXTENSION.findall(without_phones):
        terms.add("ext:" + ext)
    for number in _NUMBER.findall(without_phones):
        digits = re.sub(r"\D", "", number)
        if len(digits) >= 5:
            terms.add("num:" + digits)
    for name in _NAME.findall(text):
        words = name.lower().split()
        terms.add("name:" + " ".join(words))
        # "Call Sari Wulandari" is matched as one 3-word name, so also keep every 2-word piece of it.
        terms.update("name:" + " ".join(words[i:i + 2]) for i in range(len(words) - 1))
    for word in _CAMEL.findall(text):
        terms.add("name:" + word.lower())
    return terms


def hash_term(term: str) -> str:
    return hashlib.sha256(term.encode("utf-8")).hexdigest()[:20]


def build_fingerprints(texts_by_collection: dict[str, list[str]]) -> dict[str, list[str]]:
    return {
        collection: sorted({hash_term(t) for text in texts for t in extract_terms(text)})
        for collection, texts in texts_by_collection.items()
    }


def save_fingerprints(fingerprints: dict[str, list[str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(fingerprints), encoding="utf-8")


class LeakDetector:
    """Checks answers for terms from collections the user may not read."""

    def __init__(self, forbidden_hashes: set[str]):
        self.forbidden = forbidden_hashes

    @classmethod
    def for_collections(cls, path: Path, allowed: tuple[str, ...]) -> "LeakDetector":
        if not path.exists():
            raise RuntimeError(f"no fingerprints at {path}. Run `python -m rag.ingest` first.")
        data: dict[str, list[str]] = json.loads(path.read_text(encoding="utf-8"))
        allowed_hashes = {h for c in allowed for h in data.get(c, [])}
        forbidden = {h for c, hashes in data.items() if c not in allowed for h in hashes}
        return cls(forbidden - allowed_hashes)

    def leaked_terms(self, answer: str) -> list[str]:
        """The terms in `answer` that come from forbidden collections (empty list = clean)."""
        return sorted(t for t in extract_terms(answer) if hash_term(t) in self.forbidden)
