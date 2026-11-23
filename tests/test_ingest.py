"""Incremental re-indexing: what gets re-embedded, and removing a changed file's old chunks."""

from rag.chunking import Chunk
from rag.documents import Document
from rag.ingest import plan
from rag.store import IndexWriter, KnowledgeStore, close_all

KEY = "model|sections-v1"


def doc(source: str, doc_id: str = "faq") -> Document:
    return Document(doc_id=doc_id, title=doc_id, visibility="public", allowed_roles=(), language="en",
                    audience="x", version=1, status="current", last_updated="2026-01-01", body="text",
                    source=source)


def manifest(entries: dict[str, str], key: str = KEY) -> dict:
    return {"index_key": key, "docs": {"public": entries}}


def test_nothing_changed():
    docs = [doc("public/en/a.md"), doc("public/en/b.md")]
    hashes = {"public/en/a.md": "1", "public/en/b.md": "2"}
    assert plan("public", docs, hashes, manifest(hashes), KEY, exists=True, full=False) == ("unchanged", [], [])


def test_only_the_edited_and_new_files_are_embedded():
    a, b, c = doc("public/en/a.md"), doc("public/en/b.md"), doc("public/en/c.md")
    hashes = {a.source: "1", b.source: "2-edited", c.source: "3"}
    mode, embed, delete = plan("public", [a, b, c], hashes, manifest({a.source: "1", b.source: "2"}),
                               KEY, exists=True, full=False)
    assert mode == "update"
    assert [d.source for d in embed] == [b.source, c.source]
    assert delete == [b.source]  # the edited file's old chunks go first; the new file had none


def test_deleted_files_are_removed():
    a = doc("public/en/a.md")
    mode, embed, delete = plan("public", [a], {a.source: "1"},
                               manifest({a.source: "1", "public/en/gone.md": "9"}), KEY, exists=True, full=False)
    assert (mode, embed, delete) == ("update", [], ["public/en/gone.md"])


def test_rebuild_when_model_or_chunking_changed_or_asked():
    a = doc("public/en/a.md")
    same = manifest({a.source: "1"})
    assert plan("public", [a], {a.source: "1"}, same, "other-model|sections-v1", True, False)[0] == "rebuild"
    assert plan("public", [a], {a.source: "1"}, same, KEY, exists=False, full=False)[0] == "rebuild"
    assert plan("public", [a], {a.source: "1"}, same, KEY, exists=True, full=True)[0] == "rebuild"
    assert plan("public", [a], {a.source: "1"}, {}, KEY, exists=True, full=False)[0] == "rebuild"


def test_delete_sources_removes_only_that_file(tmp_path):
    a, b = doc("public/en/a.md", "a"), doc("public/en/b.md", "b")
    writer = IndexWriter(tmp_path)
    chunks = [Chunk(doc=a, index=0, text="a0"), Chunk(doc=a, index=1, text="a1"), Chunk(doc=b, index=0, text="b0")]
    writer.rebuild("public", chunks, [[1, 0], [1, 0], [0, 1]], 2)
    writer.delete_sources("public", [a.source])
    writer.add("public", [Chunk(doc=a, index=0, text="a0 new")], [[1, 0]])
    try:
        store = KnowledgeStore(["public"], tmp_path)
        texts = sorted(h.text for h in store._all_chunks("public"))
        assert texts == ["a0 new", "b0"]  # the old second chunk of a.md is gone, b.md untouched
    finally:
        close_all()
