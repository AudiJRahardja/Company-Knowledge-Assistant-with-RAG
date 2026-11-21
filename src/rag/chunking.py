"""Split documents into chunks small enough to search precisely.

Heading-aware chunking (Phase 3)
--------------------------------
Our documents are Markdown with `##` section headings, and each section is about one topic ("Carrying over
leave", "Peak periods for store staff", ...). So we cut at those headings: one section = one chunk. A chunk
then never mixes two topics, and never cuts a rule in half.

- Text before the first `##` (a short intro such as "This version replaces the 2024 policy") is joined to the
  first section, so it is not left as a tiny chunk of its own.
- A section longer than CHUNK_WORDS is split further into overlapping word windows (none are today; the
  longest is ~140 words), so the method still works if a long document is added later.
- Every chunk keeps its heading, and its search text starts with "Document title > Section heading", so
  a chunk from the middle of "Sick Leave Policy" still "knows" what it is about.

English and Indonesian versions of a document have the same sections in the same order, so section N of
the English file and section N of the Indonesian file are translations of each other. Retrieval uses this
to avoid showing both copies (see `twin_key`).

Phase 1 used fixed 220-word windows that ignored headings; the eval results for both are in eval/results.csv.
"""

import re
from dataclasses import dataclass

from rag.documents import Document

CHUNK_WORDS = 220      # sections longer than this are split into windows
OVERLAP_WORDS = 40     # words repeated between neighbouring windows so a sentence is not cut off from context

_H2 = re.compile(r"^##\s+(.+?)\s*$")
_H1 = re.compile(r"^#\s+.+$")


@dataclass(frozen=True)
class Chunk:
    doc: Document
    index: int              # position of the chunk inside its document (0, 1, 2, ...)
    text: str               # the chunk's own text (with its heading), shown to the LLM and the user
    section: str = ""       # the section heading, e.g. "Carrying over leave"
    section_index: int = 0  # which section of the document (same number in the EN and ID versions)
    part: int = 0           # 0 unless a long section was split into several windows

    @property
    def chunk_key(self) -> str:
        """Unique, stable name for this chunk, e.g. 'faq/en/v1/#2'."""
        return f"{self.doc.doc_id}/{self.doc.language}/v{self.doc.version}/#{self.index}"

    @property
    def twin_key(self) -> str:
        """Same value for a chunk and its translation, e.g. 'faq/v1/s2.0' for both EN and ID."""
        return f"{self.doc.doc_id}/v{self.doc.version}/s{self.section_index}.{self.part}"

    @property
    def embed_text(self) -> str:
        """What we actually turn into a vector: 'title > section' + chunk text."""
        header = f"{self.doc.title} > {self.section}" if self.section else self.doc.title
        return f"{header}\n\n{self.text}"


def split_sections(body: str) -> list[tuple[str, str]]:
    """Return [(heading, text), ...] for the `##` sections of a Markdown body. The `#` title line is dropped."""
    sections: list[tuple[str, list[str]]] = [("", [])]
    for line in body.splitlines():
        match = _H2.match(line)
        if match:
            sections.append((match.group(1), []))
        elif not _H1.match(line):
            sections[-1][1].append(line)
    result = [(heading, "\n".join(lines).strip()) for heading, lines in sections]
    intro_heading, intro = result[0]
    result = result[1:] if not intro else result
    if intro and len(result) > 1:  # join the intro to the first real section
        heading, text = result[1]
        result = [(heading, f"{intro}\n\n{text}")] + result[2:]
    return [(h, t) for h, t in result if t]


def split_words(text: str, size: int, overlap: int) -> list[str]:
    """Split text into windows of `size` words, each overlapping the previous one by `overlap` words."""
    if overlap >= size:
        raise ValueError("overlap must be smaller than size")
    words = text.split()
    if len(words) <= size:
        return [text]
    windows, start = [], 0
    while True:
        windows.append(" ".join(words[start:start + size]))
        if start + size >= len(words):
            return windows
        start += size - overlap


def chunk_document(doc: Document, size: int = CHUNK_WORDS, overlap: int = OVERLAP_WORDS) -> list[Chunk]:
    chunks: list[Chunk] = []
    for section_index, (heading, text) in enumerate(split_sections(doc.body)):
        for part, piece in enumerate(split_words(text, size, overlap)):
            shown = f"## {heading}\n{piece}" if heading else piece
            chunks.append(Chunk(doc=doc, index=len(chunks), text=shown, section=heading,
                                section_index=section_index, part=part))
    return chunks


def chunk_documents(docs: list[Document]) -> list[Chunk]:
    return [chunk for doc in docs for chunk in chunk_document(doc)]
