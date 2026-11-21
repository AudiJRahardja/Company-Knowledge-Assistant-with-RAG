from collections import defaultdict

from rag.chunking import chunk_document, chunk_documents, split_sections, split_words
from rag.config import get_settings
from rag.documents import Document, load_documents


def make_doc(body: str, language: str = "en") -> Document:
    return Document(
        doc_id="d", title="Test Doc", visibility="public", allowed_roles=(), language=language,
        audience="customers", version=1, status="current", last_updated="2026-01-01",
        body=body, source=f"public/{language}/d.md",
    )


BODY = """# Test Doc

Short intro line.

## Returns
You can return items within 30 days.

## Shipping
We ship to all of Indonesia.
"""


def test_one_chunk_per_section_with_intro_joined_to_first():
    chunks = chunk_document(make_doc(BODY))
    assert [c.section for c in chunks] == ["Returns", "Shipping"]
    assert chunks[0].text.startswith("## Returns\nShort intro line.")
    assert "30 days" in chunks[0].text and "30 days" not in chunks[1].text
    assert "Test Doc" not in chunks[0].text  # the # title line is not repeated in every chunk


def test_embed_text_has_title_and_section():
    chunk = chunk_document(make_doc(BODY))[1]
    assert chunk.embed_text.startswith("Test Doc > Shipping\n\n")


def test_document_without_headings_is_one_chunk():
    assert split_sections("# T\n\nJust some text.") == [("", "Just some text.")]
    chunks = chunk_document(make_doc("# T\n\nJust some text."))
    assert len(chunks) == 1 and chunks[0].embed_text.startswith("Test Doc\n\n")


def test_long_section_is_split_into_overlapping_windows():
    words = " ".join(f"w{i}" for i in range(250))
    windows = split_words(words, size=100, overlap=20)
    assert len(windows) == 3
    assert windows[0].split()[-20:] == windows[1].split()[:20]
    assert {w for win in windows for w in win.split()} == {f"w{i}" for i in range(250)}
    chunks = chunk_document(make_doc(f"## Long\n{words}"), size=100, overlap=20)
    assert [c.part for c in chunks] == [0, 1, 2] and len({c.chunk_key for c in chunks}) == 3


def test_translations_share_twin_keys():
    en = chunk_document(make_doc(BODY, "en"))
    id_ = chunk_document(make_doc(BODY.replace("Returns", "Pengembalian"), "id"))
    assert [c.twin_key for c in en] == [c.twin_key for c in id_]
    assert en[0].chunk_key != id_[0].chunk_key


def test_real_documents_line_up_across_languages():
    """Every English chunk has an Indonesian twin and the other way round."""
    keys = defaultdict(set)
    for chunk in chunk_documents(load_documents(get_settings().docs_dir)):
        keys[chunk.doc.language].add(chunk.twin_key)
    assert keys["en"] == keys["id"]
