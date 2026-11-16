"""Find the chunks most relevant to a question.

Phase 1: plain vector search. We search every collection the profile may use, put all results together,
and keep the best `top_k` by similarity score. (Scores are comparable across collections because they all
use the same embedding model and cosine similarity.)

Phase 3 will add keyword search (BM25), rank fusion, a reranker and query rewriting on top of this.
"""

from rag.embeddings import Embedder
from rag.store import Hit, KnowledgeStore


def retrieve(store: KnowledgeStore, embedder: Embedder, question: str, top_k: int) -> list[Hit]:
    vector = embedder.embed_query(question)
    hits: list[Hit] = []
    for collection in store.allowed:
        hits.extend(store.search(collection, vector, limit=top_k))
    hits.sort(key=lambda h: h.score, reverse=True)
    return hits[:top_k]
