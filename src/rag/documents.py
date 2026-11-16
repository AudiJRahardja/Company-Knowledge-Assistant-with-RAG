"""Load the Markdown knowledge base.

Each file starts with a YAML "front-matter" block between two `---` lines that describes the document
(id, visibility, language, version, ...). The rest of the file is the document text.

Files live in data/docs/<visibility>/<language>/<name>.md. We check that the visibility written inside the
file matches the folder it sits in, so an internal document accidentally copied into the public folder is
caught at load time instead of leaking into the public index.
"""

from dataclasses import dataclass
from pathlib import Path

import yaml

VISIBILITIES = ("public", "internal", "restricted")
LANGUAGES = ("en", "id")
STATUSES = ("current", "superseded")
REQUIRED_FIELDS = (
    "doc_id", "title", "visibility", "allowed_roles", "language",
    "audience", "version", "status", "last_updated",
)


@dataclass(frozen=True)
class Document:
    doc_id: str
    title: str
    visibility: str
    allowed_roles: tuple[str, ...]
    language: str
    audience: str
    version: int
    status: str
    last_updated: str  # ISO date, e.g. "2026-03-01"
    body: str
    source: str  # path relative to the docs folder, e.g. "public/en/faq.md"


class DocumentError(ValueError):
    """A document file is missing metadata or has invalid values."""


def split_front_matter(text: str) -> tuple[dict, str]:
    """Return (metadata, body) for a Markdown file that starts with a --- front-matter block."""
    if not text.startswith("---"):
        raise DocumentError("file does not start with a '---' front-matter block")
    try:
        _, header, body = text.split("---", 2)
    except ValueError:
        raise DocumentError("front-matter block is not closed with '---'") from None
    meta = yaml.safe_load(header) or {}
    return meta, body.strip()


def parse_document(path: Path, docs_dir: Path) -> Document:
    meta, body = split_front_matter(path.read_text(encoding="utf-8"))
    source = path.relative_to(docs_dir).as_posix()

    missing = [f for f in REQUIRED_FIELDS if f not in meta]
    if missing:
        raise DocumentError(f"{source}: missing fields {missing}")
    if meta["visibility"] not in VISIBILITIES:
        raise DocumentError(f"{source}: unknown visibility {meta['visibility']!r}")
    if meta["language"] not in LANGUAGES:
        raise DocumentError(f"{source}: unknown language {meta['language']!r}")
    if meta["status"] not in STATUSES:
        raise DocumentError(f"{source}: unknown status {meta['status']!r}")

    # Safety check: the folder decides which index a document goes into, so it must agree with the label.
    folder = source.split("/")[0]
    if folder != meta["visibility"]:
        raise DocumentError(
            f"{source}: visibility is {meta['visibility']!r} but the file is in the {folder!r} folder"
        )
    roles = tuple(meta["allowed_roles"] or ())
    if meta["visibility"] == "restricted" and not roles:
        raise DocumentError(f"{source}: restricted documents must list allowed_roles")
    if meta["visibility"] != "restricted" and roles:
        raise DocumentError(f"{source}: only restricted documents can have allowed_roles")

    return Document(
        doc_id=str(meta["doc_id"]),
        title=str(meta["title"]),
        visibility=meta["visibility"],
        allowed_roles=roles,
        language=meta["language"],
        audience=str(meta["audience"]),
        version=int(meta["version"]),
        status=meta["status"],
        last_updated=str(meta["last_updated"]),
        body=body,
        source=source,
    )


def load_documents(docs_dir: Path) -> list[Document]:
    """Load and validate every .md file under docs_dir."""
    paths = sorted(docs_dir.rglob("*.md"))
    if not paths:
        raise DocumentError(f"no Markdown files found in {docs_dir}")
    return [parse_document(p, docs_dir) for p in paths]
