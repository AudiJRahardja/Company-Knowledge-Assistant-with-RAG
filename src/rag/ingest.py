"""Build the search indexes from the Markdown documents.

Usage:  python -m rag.ingest                      only re-index documents that changed (fast)
        python -m rag.ingest --full               rebuild everything
        python -m rag.ingest --collections public build only these collections (a customer-only server)

Steps: load + validate docs -> split into chunks -> embed -> write each chunk into the collection that matches
its visibility (public / internal / restricted).

Incremental re-indexing (Phase 6)
---------------------------------
A manifest (storage/qdrant/manifest.json) remembers a hash of every document file that is in the index. On
the next run only new or edited files are chunked and embedded again; the chunks of edited and deleted files
are removed first. A file that moved folder (e.g. internal -> public) counts as deleted in one collection
and new in the other. Everything is rebuilt anyway when the embedding model or the chunking method changed
(they are part of the manifest), when a collection has no index yet, or with --full.

The leak fingerprints (hashed names, emails, phone numbers per collection) are rebuilt on every run: that
needs no model and takes a moment.

Note: Qdrant's local mode locks its folders, so stop the app / API before running this.
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

from rag.chunking import CHUNKING_VERSION, chunk_documents
from rag.config import get_settings
from rag.documents import Document, load_documents
from rag.embeddings import Embedder
from rag.fingerprint import build_fingerprints, save_fingerprints
from rag.store import COLLECTIONS, IndexWriter, close_all


def file_hash(docs_dir: Path, doc: Document) -> str:
    return hashlib.sha256((docs_dir / doc.source).read_bytes()).hexdigest()


def load_manifest(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def plan(collection: str, docs: list[Document], hashes: dict[str, str], manifest: dict,
         index_key: str, exists: bool, full: bool) -> tuple[str, list[Document], list[str]]:
    """Decide what to do with one collection.

    Returns (mode, docs_to_embed, sources_to_delete), mode = "rebuild" | "update" | "unchanged".
    """
    if full or not exists or manifest.get("index_key") != index_key:
        return "rebuild", docs, []
    old = {src: h for src, h in manifest.get("docs", {}).get(collection, {}).items()}
    changed = [d for d in docs if old.get(d.source) != hashes[d.source]]
    current = {d.source for d in docs}
    removed = [src for src in old if src not in current]
    if not changed and not removed:
        return "unchanged", [], []
    # Edited files: delete their old chunks first (a shorter file has fewer chunks), then add the new ones.
    return "update", changed, removed + [d.source for d in changed if d.source in old]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the search indexes.")
    parser.add_argument("--full", action="store_true", help="rebuild every collection from scratch")
    parser.add_argument("--collections", default=",".join(COLLECTIONS),
                        help="comma-separated: which collections to build (default: all)")
    args = parser.parse_args()
    wanted = [c.strip() for c in args.collections.split(",") if c.strip()]
    unknown = set(wanted) - set(COLLECTIONS)
    if unknown:
        raise SystemExit(f"unknown collections: {', '.join(sorted(unknown))}")

    settings = get_settings()
    started = time.perf_counter()
    docs = load_documents(settings.docs_dir)
    print(f"Loaded {len(docs)} documents from {settings.docs_dir}")

    manifest_path = settings.storage_dir / "qdrant" / "manifest.json"
    manifest = load_manifest(manifest_path)
    index_key = f"{settings.embedding_model}|{CHUNKING_VERSION}"
    hashes = {d.source: file_hash(settings.docs_dir, d) for d in docs}

    writer = IndexWriter(settings.storage_dir)
    embedder = None
    new_docs: dict[str, dict[str, str]] = {}
    print(f"\n{'collection':<12}{'docs':>6}{'action':>12}{'embedded':>10}")
    for collection in wanted:
        col_docs = [d for d in docs if d.visibility == collection]
        mode, to_embed, to_delete = plan(collection, col_docs, hashes, manifest, index_key,
                                         writer.exists(collection), args.full)
        chunks = chunk_documents(to_embed)
        if (chunks or mode == "rebuild") and embedder is None:
            embedder = Embedder(settings.embedding_model, settings.embedding_device)
        vectors = embedder.embed_documents([c.embed_text for c in chunks]) if chunks else []
        if mode == "rebuild":
            writer.rebuild(collection, chunks, vectors, embedder.dimension)
        elif mode == "update":
            writer.delete_sources(collection, to_delete)
            writer.add(collection, chunks, vectors)
        new_docs[collection] = {d.source: hashes[d.source] for d in col_docs}
        detail = f"{len(to_embed)} docs / {len(chunks)} chunks" if mode != "unchanged" else "-"
        print(f"{collection:<12}{len(col_docs):>6}{mode:>12}   {detail}")

    # Collections not built in this run keep their old manifest entries (and must use the same index key).
    keep = manifest.get("docs", {}) if manifest.get("index_key") == index_key else {}
    manifest = {"index_key": index_key, "docs": {**{c: v for c, v in keep.items() if c not in wanted}, **new_docs}}
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=1, sort_keys=True), encoding="utf-8")

    # Fingerprints come from every document we have, also on a public-only server: they are only hashes.
    fingerprints = build_fingerprints({c: [d.body for d in docs if d.visibility == c] for c in COLLECTIONS})
    save_fingerprints(fingerprints, settings.security_dir / "fingerprints.json")
    print("\nLeak fingerprints (hashed): " + ", ".join(f"{c} {len(h)}" for c, h in fingerprints.items()))

    close_all()
    print(f"\nDone in {time.perf_counter() - started:.1f}s. Indexes are in {settings.storage_dir / 'qdrant'}")


if __name__ == "__main__":
    main()
