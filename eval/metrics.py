"""Scoring functions. Plain code, no LLM, so these numbers are exact and repeatable.

Retrieval
  hit@k        did ANY expected document show up in the top k chunks?
  all@k        did ALL expected documents show up? (matters for multi-document questions)
  MRR          1 / rank of the first expected document (1.0 = it was the very first chunk)
  distinct@k   how many different documents are in the top k (EN and ID copies of one doc count once)

Answers
  facts        all key facts (must_include) appear in the answer
  stale        the answer cites an outdated (superseded) document, or states a fact from it (must_not_include)
  leak         a forbidden collection reached the LLM, or a forbidden string appears in the answer
"""

import re
from collections.abc import Iterable

_THOUSANDS = re.compile(r"(?<=\d)[.,](?=\d{3}(?!\d))")   # 1.200.000 / 1,200,000 -> 1200000
_DOT_TIME = re.compile(r"(?<!\d)(\d{1,2})\.(\d{2})(?!\d)")  # 08.00 (Indonesian style) -> 08:00
_RP_SPACE = re.compile(r"\brp\s+(?=\d)")            # "Rp 250.000" -> "rp250.000"
_DASHES = str.maketrans({"–": "-", "—": "-", " ": " "})


def normalize(text: str) -> str:
    """Lower-case text with number formats made uniform, so 'Rp1.200.000' and 'Rp 1,200,000' compare equal."""
    t = text.lower().translate(_DASHES)
    t = _RP_SPACE.sub("rp", t)
    t = _THOUSANDS.sub("", t)
    t = _DOT_TIME.sub(r"\1:\2", t)
    return re.sub(r"\s+", " ", t)


def contains(text: str, phrase: str) -> bool:
    """Does `phrase` occur in `text`? Numbers must match whole: '12' does not match inside '120'."""
    needle = normalize(phrase)
    pattern = re.escape(needle)
    if needle[:1].isdigit():
        pattern = r"(?<!\d)" + pattern
    if needle[-1:].isdigit():
        pattern = pattern + r"(?!\d)"
    return re.search(pattern, normalize(text)) is not None


def facts_ok(answer: str, groups: Iterable[Iterable[str]]) -> bool:
    return all(any(contains(answer, alt) for alt in group) for group in groups)


def found_strings(answer: str, phrases: Iterable[str]) -> list[str]:
    return [p for p in phrases if contains(answer, p)]


def first_rank(retrieved_doc_ids: list[str], expected: Iterable[str]) -> int | None:
    expected = set(expected)
    for rank, doc_id in enumerate(retrieved_doc_ids, start=1):
        if doc_id in expected:
            return rank
    return None


def retrieval_scores(retrieved_doc_ids: list[str], expected: Iterable[str]) -> dict:
    expected = set(expected)
    rank = first_rank(retrieved_doc_ids, expected)
    return {
        "hit": rank is not None,
        "all": expected <= set(retrieved_doc_ids),
        "rr": 1.0 / rank if rank else 0.0,
        "rank": rank,
    }


REFUSAL_MARKERS = (
    "don't know", "do not know", "don't have", "do not have", "not able to", "unable to", "can't help",
    "cannot help", "can't share", "cannot share", "not in the sources", "no information", "not mentioned",
    "only help with", "can only help", "tidak tahu", "tidak memiliki informasi", "tidak ada informasi",
    "tidak dapat", "tidak bisa membantu", "tidak bisa memberikan", "belum memiliki informasi", "hanya dapat membantu",
    "hanya bisa membantu",
)


def looks_like_refusal(answer: str) -> bool:
    """Rough keyword check, used only when the LLM judge is turned off."""
    a = normalize(answer)
    return any(marker in a for marker in REFUSAL_MARKERS)


def mean(values: Iterable[float | bool | None]) -> float | None:
    vals = [float(v) for v in values if v is not None]
    return sum(vals) / len(vals) if vals else None
