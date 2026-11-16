"""Split documents into chunks small enough to search precisely.

Phase 1 uses a deliberately simple method: fixed-size windows of words with some overlap, ignoring the
document's headings. This gives us a baseline to measure against; Phase 3 replaces it with heading-aware
chunking and compares the eval scores.

Every chunk's search text starts with the document title, so a chunk from the middle of "Sick Leave Policy"
still "knows" what it is about.
"""

from dataclasses import dataclass

from rag.documents import Document

CHUNK_WORDS = 220      # roughly 300-400 tokens for English, a bit more for Indonesian
OVERLAP_WORDS = 40     # words repeated between neighbouring chunks so a sentence is not cut off from context


@dataclass(frozen=True)
class Chunk:
    doc: Document
    index: int     # position of the chunk inside its document (0, 1, 2, ...)
    text: str      # the chunk's own text, shown to the LLM and the user

    @property
    def chunk_key(self) -> str:
        """Unique, stable name for this chunk, e.g. 'faq/en/v1/#2'."""
        return f"{self.doc.doc_id}/{self.doc.language}/v{self.doc.version}/#{self.index}"

    @property
    def embed_text(self) -> str:
        """What we actually turn into a vector: title + chunk text."""
        return f"{self.doc.title}\n\n{self.text}"


def chunk_document(doc: Document, size: int = CHUNK_WORDS, overlap: int = OVERLAP_WORDS) -> list[Chunk]:
    if overlap >= size:
        raise ValueError("overlap must be smaller than size")
    words = doc.body.split()
    if not words:
        return []

    chunks = []
    step = size - overlap
    start = 0
    while True:
        window = words[start:start + size]
        chunks.append(Chunk(doc=doc, index=len(chunks), text=" ".join(window)))
        if start + size >= len(words):
            break
        start += step
    return chunks


def chunk_documents(docs: list[Document]) -> list[Chunk]:
    return [chunk for doc in docs for chunk in chunk_document(doc)]
