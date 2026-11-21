"""Security tests for the storage layer. These use tiny fake vectors, so no model is needed."""

import pytest

from rag.chunking import Chunk
from rag.documents import Document
from rag.store import AccessDenied, IndexWriter, KnowledgeStore, close_all

DIM = 4


def chunk(visibility: str, doc_id: str) -> Chunk:
    doc = Document(
        doc_id=doc_id, title=doc_id, visibility=visibility,
        allowed_roles=("hr",) if visibility == "restricted" else (), language="en", audience="x",
        version=1, status="current", last_updated="2026-01-01", body="text", source=f"{visibility}/en/x.md",
    )
    return Chunk(doc=doc, index=0, text=f"{visibility} text")


@pytest.fixture
def storage(tmp_path):
    writer = IndexWriter(tmp_path)
    for i, vis in enumerate(("public", "internal", "restricted")):
        vector = [0.0] * DIM
        vector[i] = 1.0
        writer.rebuild(vis, [chunk(vis, f"{vis}-doc")], [vector], DIM)
    yield tmp_path
    close_all()


def test_public_store_cannot_search_internal(storage):
    store = KnowledgeStore(["public"], storage)
    with pytest.raises(AccessDenied):
        store.search("internal", [0, 1, 0, 0], limit=5)
    with pytest.raises(AccessDenied):
        store.search("restricted", [0, 0, 1, 0], limit=5)


def test_public_store_only_returns_public_chunks(storage):
    store = KnowledgeStore(["public"], storage)
    # Even a query vector that points straight at the internal chunk only finds public data.
    hits = store.search("public", [0, 1, 0, 0], limit=5)
    assert [h.doc_id for h in hits] == ["public-doc"]


def test_unknown_collection_is_rejected(storage):
    with pytest.raises(ValueError):
        KnowledgeStore(["public", "everything"], storage)


def test_missing_index_gives_clear_error(tmp_path):
    with pytest.raises(RuntimeError, match="ingest"):
        KnowledgeStore(["public"], tmp_path)
    close_all()


def test_writer_refuses_chunk_with_wrong_visibility(tmp_path):
    writer = IndexWriter(tmp_path)
    with pytest.raises(ValueError, match="refusing"):
        writer.rebuild("public", [chunk("internal", "secret")], [[1.0, 0, 0, 0]], DIM)
    close_all()


def test_keyword_search_has_the_same_access_rules(storage):
    store = KnowledgeStore(["public"], storage)
    assert [h.collection for h in store.keyword_search("public", "public text", limit=5)] == ["public"]
    with pytest.raises(AccessDenied):
        store.keyword_search("internal", "internal text", limit=5)
