"""Keyword search with BM25, the classic search-engine scoring formula.

Vector search understands meaning ("time off" ~ "leave") but can miss exact terms such as "Care+", "COD",
"G5" or an order number. BM25 is the opposite: it only matches words, and scores a chunk higher when
  - it contains the query's words, more often (with diminishing returns: parameter k1),
  - those words are rare across all chunks (inverse document frequency: "warranty" counts more than "the"),
  - the chunk is short (a match in a short chunk means more; parameter b).
Retrieval combines both rankings (see retrieval.py).

The corpus is small (a few hundred chunks), so the index is built in memory when the assistant starts.
"""

import math
import re
from collections import Counter

_WORD = re.compile(r"\w+")
_THOUSANDS = re.compile(r"(?<=\d)[.,](?=\d{3}(?!\d))")  # "250.000" and "250,000" both become "250000"


def tokenize(text: str) -> list[str]:
    return _WORD.findall(_THOUSANDS.sub("", text.lower()))


class BM25:
    def __init__(self, texts: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.term_counts = [Counter(tokenize(t)) for t in texts]
        self.lengths = [sum(c.values()) for c in self.term_counts]
        self.avg_length = (sum(self.lengths) / len(self.lengths)) if self.lengths else 0.0
        doc_freq = Counter(term for counts in self.term_counts for term in counts)
        n = len(texts)
        self.idf = {term: math.log(1 + (n - df + 0.5) / (df + 0.5)) for term, df in doc_freq.items()}

    def scores(self, query: str) -> list[float]:
        terms = [t for t in set(tokenize(query)) if t in self.idf]
        result = []
        for counts, length in zip(self.term_counts, self.lengths):
            score = 0.0
            for term in terms:
                tf = counts.get(term, 0)
                if tf:
                    norm = self.k1 * (1 - self.b + self.b * length / self.avg_length)
                    score += self.idf[term] * tf * (self.k1 + 1) / (tf + norm)
            result.append(score)
        return result

    def top(self, query: str, n: int) -> list[tuple[int, float]]:
        """Indexes and scores of the best `n` texts with a score above zero, best first."""
        ranked = sorted(enumerate(self.scores(query)), key=lambda pair: pair[1], reverse=True)
        return [(i, s) for i, s in ranked[:n] if s > 0]
