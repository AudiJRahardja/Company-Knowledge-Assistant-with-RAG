"""Vector storage, and the most important security boundary in the project.

Design
------
Each visibility level gets its OWN Qdrant database folder:

    storage/qdrant/public/       only public documents
    storage/qdrant/internal/     only internal documents
    storage/qdrant/restricted/   only restricted documents (HR / manager)

This is stronger than putting everything in one database with a "visibility" field and filtering at search
time: a filter is one line of code that can be forgotten or bypassed by a bug, but a database that was never
opened cannot return anything. When we deploy (Phase 6), the external assistant's container will only get the
public folder mounted, so internal data is physically not there.

Inside one Python process (e.g. the Streamlit demo that hosts both assistants), the boundary is the
`KnowledgeStore` class: it is created with a fixed list of allowed collections, only opens those folders,
and raises `AccessDenied` for anything else. The assistant pipeline only ever receives a `KnowledgeStore`.
Writing is done by a separate `IndexWriter`, used only by the ingest script.
"""

import atexit
import uuid
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from rag.chunking import Chunk
from rag.documents import VISIBILITIES
from rag.keyword import BM25

COLLECTIONS = VISIBILITIES          # one collection per visibility level
_QDRANT_COLLECTION = "chunks"       # name of the table inside each database folder

# Qdrant's local mode allows only one client per folder per process, so we share them.
_clients: dict[Path, QdrantClient] = {}


class AccessDenied(PermissionError):
    """Raised when code tries to read a collection it was not given access to."""


@dataclass(frozen=True)
class Hit:
    """One search result."""
    collection: str
    score: float
    chunk_key: str
    doc_id: str
    title: str
    language: str
    version: int
    status: str
    last_updated: str
    source: str
    text: str
    section: str = ""    # section heading the chunk comes from
    twin_key: str = ""   # same for a chunk and its translation (see chunking.Chunk.twin_key)


def _db_path(storage_dir: Path, collection: str) -> Path:
    if collection not in COLLECTIONS:
        raise ValueError(f"unknown collection {collection!r}; expected one of {COLLECTIONS}")
    return storage_dir / "qdrant" / collection


def _client_for(storage_dir: Path, collection: str) -> QdrantClient:
    path = _db_path(storage_dir, collection).resolve()
    if path not in _clients:
        path.mkdir(parents=True, exist_ok=True)
        _clients[path] = QdrantClient(path=str(path))
    return _clients[path]


def close_all() -> None:
    """Close every open database (needed before re-ingesting in the same process, and in tests)."""
    for client in _clients.values():
        client.close()
    _clients.clear()


atexit.register(close_all)  # close cleanly at exit instead of during interpreter shutdown


class KnowledgeStore:
    """Read-only access to a fixed set of collections. Nothing else can be reached through it."""

    def __init__(self, allowed: Iterable[str], storage_dir: Path):
        allowed = tuple(allowed)
        if not allowed:
            raise ValueError("a KnowledgeStore needs at least one collection")
        self._clients: dict[str, QdrantClient] = {}
        for name in allowed:
            client = _client_for(storage_dir, name)
            if not client.collection_exists(_QDRANT_COLLECTION):
                raise RuntimeError(
                    f"collection {name!r} has no index yet. Run `python -m rag.ingest` first."
                )
            self._clients[name] = client
        self._keyword: dict[str, tuple[BM25, list[Hit]]] = {}  # built on first keyword search

    def _check(self, collection: str) -> None:
        if collection not in self._clients:
            raise AccessDenied(f"this store has no access to the {collection!r} collection")

    @property
    def allowed(self) -> tuple[str, ...]:
        return tuple(self._clients)

    def search(self, collection: str, vector: list[float], limit: int) -> list[Hit]:
        """Vector (meaning) search."""
        self._check(collection)
        result = self._clients[collection].query_points(
            collection_name=_QDRANT_COLLECTION, query=vector, limit=limit, with_payload=True
        )
        return [
            Hit(collection=collection, score=p.score,
                **{k: p.payload[k] for k in _PAYLOAD_FIELDS},
                **{k: p.payload[k] for k in _OPTIONAL_FIELDS if k in p.payload})
            for p in result.points
        ]

    def keyword_search(self, collection: str, query: str, limit: int) -> list[Hit]:
        """BM25 keyword search. The score is the BM25 score (not comparable with vector scores)."""
        self._check(collection)
        if collection not in self._keyword:
            chunks = self._all_chunks(collection)
            self._keyword[collection] = (BM25([f"{h.title}\n{h.text}" for h in chunks]), chunks)
        index, chunks = self._keyword[collection]
        return [replace(chunks[i], score=score) for i, score in index.top(query, limit)]

    def _all_chunks(self, collection: str) -> list[Hit]:
        points, offset = [], None
        while True:
            batch, offset = self._clients[collection].scroll(
                _QDRANT_COLLECTION, limit=256, offset=offset, with_payload=True, with_vectors=False
            )
            points.extend(batch)
            if offset is None:
                break
        return [
            Hit(collection=collection, score=0.0,
                **{k: p.payload[k] for k in _PAYLOAD_FIELDS},
                **{k: p.payload[k] for k in _OPTIONAL_FIELDS if k in p.payload})
            for p in points
        ]


_PAYLOAD_FIELDS = (
    "chunk_key", "doc_id", "title", "language", "version", "status", "last_updated", "source", "text",
)
_OPTIONAL_FIELDS = ("section", "twin_key")  # added in Phase 3; older indexes simply lack them


class IndexWriter:
    """Builds the indexes. Only used by the ingest script, never by an assistant."""

    def __init__(self, storage_dir: Path):
        self.storage_dir = storage_dir

    def rebuild(self, collection: str, chunks: list[Chunk], vectors: list[list[float]], dim: int) -> None:
        """Delete the collection and fill it again from scratch."""
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length")
        for chunk in chunks:
            # Last line of defence: a chunk can only be written to the collection matching its label.
            if chunk.doc.visibility != collection:
                raise ValueError(
                    f"refusing to write {chunk.doc.visibility} chunk {chunk.chunk_key} into {collection!r}"
                )

        client = _client_for(self.storage_dir, collection)
        if client.collection_exists(_QDRANT_COLLECTION):
            client.delete_collection(_QDRANT_COLLECTION)
        client.create_collection(
            _QDRANT_COLLECTION, vectors_config=VectorParams(size=dim, distance=Distance.COSINE)
        )
        if not chunks:
            return
        points = [
            PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_URL, c.chunk_key)),
                vector=v,
                payload={
                    "chunk_key": c.chunk_key,
                    "doc_id": c.doc.doc_id,
                    "title": c.doc.title,
                    "language": c.doc.language,
                    "version": c.doc.version,
                    "status": c.doc.status,
                    "last_updated": c.doc.last_updated,
                    "source": c.doc.source,
                    "visibility": c.doc.visibility,
                    "allowed_roles": list(c.doc.allowed_roles),
                    "chunk_index": c.index,
                    "section": c.section,
                    "section_index": c.section_index,
                    "twin_key": c.twin_key,
                    "text": c.text,
                },
            )
            for c, v in zip(chunks, vectors)
        ]
        client.upsert(_QDRANT_COLLECTION, points=points)
