from rag.chunking import chunk_document
from rag.documents import Document


def make_doc(n_words: int) -> Document:
    return Document(
        doc_id="d", title="Test Doc", visibility="public", allowed_roles=(), language="en",
        audience="customers", version=1, status="current", last_updated="2026-01-01",
        body=" ".join(f"w{i}" for i in range(n_words)), source="public/en/d.md",
    )


def test_short_document_is_one_chunk():
    chunks = chunk_document(make_doc(50), size=100, overlap=20)
    assert len(chunks) == 1
    assert chunks[0].text.split()[-1] == "w49"


def test_chunks_overlap_and_cover_every_word():
    chunks = chunk_document(make_doc(250), size=100, overlap=20)
    assert len(chunks) == 3
    assert chunks[0].text.split()[-20:] == chunks[1].text.split()[:20]
    covered = {w for c in chunks for w in c.text.split()}
    assert covered == {f"w{i}" for i in range(250)}


def test_embed_text_starts_with_title_and_keys_are_unique():
    chunks = chunk_document(make_doc(250), size=100, overlap=20)
    assert all(c.embed_text.startswith("Test Doc") for c in chunks)
    assert len({c.chunk_key for c in chunks}) == len(chunks)
