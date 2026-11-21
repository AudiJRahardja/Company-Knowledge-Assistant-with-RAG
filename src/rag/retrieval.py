"""Find the chunks most relevant to a question.

    question
      -> vector search (meaning) in every allowed collection   ┐ each gives a ranked list of candidates
      -> keyword search (BM25) in every allowed collection     ┘
      -> combine the two lists with Reciprocal Rank Fusion
      -> merge translation twins (keep one copy of each section)
      -> reranker re-scores the best ~20 candidates (see reranker.py)
      -> top_k chunks
    A question that asks two things ("How much leave do I get, and how do I add the baby to my insurance?")
    is also searched part by part; see `retrieve`.

Reciprocal Rank Fusion (RRF): a chunk gets 1 / (60 + rank) points from every list it appears in, and the
points are added up. It only uses ranks, so it doesn't matter that vector scores (0..1) and BM25 scores
(0..20+) are on different scales. A chunk that is near the top of both lists wins; a chunk that only one
method finds can still make it.

Merging twins: every section exists in English and Indonesian, and both copies score high for the same
question. Without this step half of the top 5 would be the same text twice. We keep the best-ranked copy
(usually the one in the question's language).

Each step can be switched off (RetrievalOptions) so the eval can measure what it adds.
"""

import re
from dataclasses import dataclass, replace

from rag.embeddings import Embedder
from rag.reranker import Reranker
from rag.store import Hit, KnowledgeStore

CANDIDATES_PER_RESULT = 4  # each method fetches 4x top_k candidates
RRF_K = 60                 # the standard constant; larger = ranks further down still count


@dataclass(frozen=True)
class RetrievalOptions:
    hybrid: bool = True        # add BM25 keyword search to vector search
    merge_twins: bool = True   # keep only one language copy of each section
    reranker: Reranker | None = None  # None = no reranking
    rerank_candidates: int = 20       # how many candidates the reranker re-scores
    rewrite: bool = True              # rewrite follow-up questions before searching (pipeline, needs the LLM)
    split_questions: bool = True      # also search each part of a multi-part question


def merge_twins(hits: list[Hit]) -> list[Hit]:
    """Keep the first (best) hit of each twin group. `hits` must already be sorted best first."""
    seen: set[str] = set()
    kept = []
    for h in hits:
        key = h.twin_key or h.chunk_key  # older indexes have no twin key: then nothing is merged
        if key not in seen:
            seen.add(key)
            kept.append(h)
    return kept


def rrf(*ranked_lists: list[Hit]) -> list[Hit]:
    """Reciprocal Rank Fusion. Returns the chunks sorted by fused score; Hit.score becomes the fused score."""
    points: dict[str, float] = {}
    first_seen: dict[str, Hit] = {}
    for hits in ranked_lists:
        for rank, h in enumerate(hits, start=1):
            points[h.chunk_key] = points.get(h.chunk_key, 0.0) + 1.0 / (RRF_K + rank)
            first_seen.setdefault(h.chunk_key, h)
    fused = [replace(first_seen[key], score=round(score, 5)) for key, score in points.items()]
    return sorted(fused, key=lambda h: h.score, reverse=True)


_ASK_EN = r"(?:how|what|when|where|which|who|can|do|does|is|are|will|should)"
_ASK_ID = r"(?:bagaimana|berapa|kapan|apa|apakah|di mana|siapa|bisakah|bolehkah)"
# Split after a "?" or before "and/dan/or/atau" when a new question starts right after it.
_SPLIT = re.compile(rf"\?\s+|[,;]?\s+(?:and|dan|or|atau)\s+(?={_ASK_EN}\b|{_ASK_ID}\b)", re.IGNORECASE)
MAX_PARTS = 3


def split_question(question: str) -> list[str]:
    """The separate questions inside a multi-part question, or [] if it asks one thing.

    "How long is probation, and when can I use annual leave?" -> ["How long is probation",
    "when can I use annual leave"]. Pieces shorter than 3 words are not real questions and are ignored.
    """
    parts = [p.strip(" ?,.;") for p in _SPLIT.split(question)]
    parts = [p for p in parts if len(p.split()) >= 3]
    return parts[:MAX_PARTS] if len(parts) >= 2 else []


def retrieve(store: KnowledgeStore, embedder: Embedder, question: str, top_k: int,
             options: RetrievalOptions = RetrievalOptions()) -> list[Hit]:
    """Search for the question. A multi-part question also gets each part searched on its own.

    Why: the reranker scores every chunk against the WHOLE question. A chunk that answers only one half
    ("paternity leave", or "adding the baby to health insurance") then scores low, and one half of the
    answer can drop out of the top 5. So each part's 2 best chunks are guaranteed a place, and the answer
    gets one extra source per extra part (top_k=5 -> 6 for a two-part question) so the full-question
    results still fit.
    """
    parts = split_question(question) if options.split_questions else []
    if not parts:
        return _search(store, embedder, question, top_k, options)
    size = top_k + len(parts) - 1
    full = _search(store, embedder, question, size, options)
    per_part = [h for part in parts for h in _search(store, embedder, part, 2, options)]
    return merge_twins(full[:1] + per_part + full[1:])[:size]


def _search(store: KnowledgeStore, embedder: Embedder, question: str, top_k: int,
            options: RetrievalOptions) -> list[Hit]:
    n = max(top_k, 5) * CANDIDATES_PER_RESULT
    vector = embedder.embed_query(question)
    by_meaning = [h for c in store.allowed for h in store.search(c, vector, limit=n)]
    by_meaning.sort(key=lambda h: h.score, reverse=True)
    hits = by_meaning[:n]
    if options.hybrid:
        by_keyword = [h for c in store.allowed for h in store.keyword_search(c, question, limit=n)]
        by_keyword.sort(key=lambda h: h.score, reverse=True)
        hits = rrf(hits, by_keyword[:n])
    if options.merge_twins:
        hits = merge_twins(hits)
    if options.reranker is not None:
        hits = options.reranker.rerank(question, hits[:options.rerank_candidates])
    return hits[:top_k]
