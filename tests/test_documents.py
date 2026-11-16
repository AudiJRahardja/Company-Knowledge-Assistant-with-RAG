from collections import defaultdict

import pytest

from rag.config import get_settings
from rag.documents import DocumentError, load_documents, parse_document

DOCS_DIR = get_settings().docs_dir


@pytest.fixture(scope="module")
def docs():
    return load_documents(DOCS_DIR)


def test_all_documents_load(docs):
    assert len(docs) >= 40


def test_every_document_has_english_and_indonesian_version(docs):
    languages = defaultdict(set)
    for d in docs:
        languages[(d.doc_id, d.version)].add(d.language)
    missing = {key: langs for key, langs in languages.items() if langs != {"en", "id"}}
    assert not missing, f"documents without both languages: {missing}"


def test_language_matches_folder(docs):
    for d in docs:
        assert d.source.split("/")[1] == d.language, d.source


def test_restricted_docs_have_roles_and_others_do_not(docs):
    for d in docs:
        if d.visibility == "restricted":
            assert d.allowed_roles, d.source
        else:
            assert not d.allowed_roles, d.source


def test_superseded_versions_have_a_newer_current_version(docs):
    current = {(d.doc_id, d.language): d.version for d in docs if d.status == "current"}
    for d in docs:
        if d.status == "superseded":
            assert current.get((d.doc_id, d.language), 0) > d.version, d.source


def test_internal_doc_in_public_folder_is_rejected(tmp_path):
    folder = tmp_path / "public" / "en"
    folder.mkdir(parents=True)
    path = folder / "leaked.md"
    path.write_text(
        "---\ndoc_id: x\ntitle: X\nvisibility: internal\nallowed_roles: []\nlanguage: en\n"
        "audience: employees\nversion: 1\nstatus: current\nlast_updated: 2026-01-01\n---\nSecret.",
        encoding="utf-8",
    )
    with pytest.raises(DocumentError, match="folder"):
        parse_document(path, tmp_path)
