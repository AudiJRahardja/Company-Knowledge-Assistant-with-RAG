"""Build the search indexes from the Markdown documents.

Usage:  python -m rag.ingest

Steps: load + validate docs -> split into chunks -> embed -> write each chunk into the collection that matches
its visibility (public / internal / restricted). Each run rebuilds everything from scratch; with ~60 documents
this takes well under a minute. (Incremental re-indexing comes in Phase 6.)

Note: Qdrant's local mode locks its folders, so stop the Streamlit app before running this.
"""

import time

from rag.chunking import chunk_documents
from rag.config import get_settings
from rag.documents import load_documents
from rag.embeddings import Embedder
from rag.fingerprint import build_fingerprints, save_fingerprints
from rag.store import COLLECTIONS, IndexWriter, close_all


def main() -> None:
    settings = get_settings()
    started = time.perf_counter()

    docs = load_documents(settings.docs_dir)
    print(f"Loaded {len(docs)} documents from {settings.docs_dir}")

    embedder = Embedder(settings.embedding_model, settings.embedding_device)
    writer = IndexWriter(settings.storage_dir)

    print(f"\n{'collection':<12}{'docs':>6}{'chunks':>8}")
    for collection in COLLECTIONS:
        col_docs = [d for d in docs if d.visibility == collection]
        chunks = chunk_documents(col_docs)
        vectors = embedder.embed_documents([c.embed_text for c in chunks]) if chunks else []
        writer.rebuild(collection, chunks, vectors, embedder.dimension)
        print(f"{collection:<12}{len(col_docs):>6}{len(chunks):>8}")

    fingerprints = build_fingerprints({c: [d.body for d in docs if d.visibility == c] for c in COLLECTIONS})
    save_fingerprints(fingerprints, settings.security_dir / "fingerprints.json")
    print("\nLeak fingerprints (hashed): " + ", ".join(f"{c} {len(h)}" for c, h in fingerprints.items()))

    close_all()
    print(f"\nDone in {time.perf_counter() - started:.1f}s. Indexes are in {settings.storage_dir / 'qdrant'}")


if __name__ == "__main__":
    main()
