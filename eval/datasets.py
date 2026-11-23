"""Load and check the eval question files.

Three files, one JSON object per line:
  eval/internal.jsonl, eval/external.jsonl   quality tests for each assistant
  eval/leak_tests.jsonl                      security tests (must never leak)

Every question exists twice, in English (`-en`) and Indonesian (`-id`), with the same id stem.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

from rag.llm import Message

EVAL_DIR = Path(__file__).resolve().parent
FILES = {
    "internal": EVAL_DIR / "internal.jsonl",
    "external": EVAL_DIR / "external.jsonl",
    "leak": EVAL_DIR / "leak_tests.jsonl",
}
CATEGORIES = ("lookup", "multi_doc", "followup", "confusable", "outdated", "old_version", "unanswerable",
              "role_access")
# outdated     = a newer version exists; the answer must use the current one
# old_version  = the user asks about the old rules on purpose; using the outdated document is correct
# role_access  = only answerable for the role in the case's "role" field (manager / hr)
ATTACKS = ("internal_info", "restricted_info", "internal_contacts", "prompt_injection", "role_escalation",
           "order_wrong_email")
LANGUAGES = ("en", "id")


class DatasetError(ValueError):
    pass


@dataclass(frozen=True)
class QualityCase:
    id: str
    language: str
    category: str
    question: str
    expected_answer: str
    expected_sources: tuple[str, ...]          # doc_ids; any language version counts
    must_include: tuple[tuple[str, ...], ...]  # every group must match; any string in a group is enough
    must_not_include: tuple[str, ...] = ()
    should_refuse: bool = False
    history: tuple[Message, ...] = ()
    role: str | None = None                    # who is asking (internal only); None = a regular employee


@dataclass(frozen=True)
class LeakCase:
    id: str
    language: str
    profile: str
    attack: str
    question: str
    forbidden_collections: tuple[str, ...]
    forbidden_strings: tuple[str, ...]
    role: str | None = None
    requires: str | None = None                # e.g. "tools": skipped until that feature exists
    history: tuple[Message, ...] = field(default_factory=tuple)
    forbidden_docs: tuple[str, ...] = ()       # doc_ids that must not reach the LLM (when a role may open the
                                               # collection but not every document in it)


def stem(case_id: str) -> str:
    return case_id.rsplit("-", 1)[0]


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise DatasetError(f"{path.name} line {n}: {exc}") from exc
    return rows


def _history(row: dict) -> tuple[Message, ...]:
    return tuple(Message(m["role"], m["content"]) for m in row.get("history", []))


def _check_common(row: dict, where: str) -> None:
    for key in ("id", "language", "question"):
        if not row.get(key):
            raise DatasetError(f"{where}: missing '{key}'")
    if row["language"] not in LANGUAGES:
        raise DatasetError(f"{where}: language must be en or id")
    if not row["id"].endswith("-" + row["language"]):
        raise DatasetError(f"{where}: id must end with -{row['language']}")


def _check_pairs(ids: list[str], name: str) -> None:
    if len(ids) != len(set(ids)):
        raise DatasetError(f"{name}: duplicate ids")
    stems = {stem(i) for i in ids}
    missing = [f"{s}-{lang}" for s in sorted(stems) for lang in LANGUAGES if f"{s}-{lang}" not in ids]
    if missing:
        raise DatasetError(f"{name}: every question needs an English and an Indonesian version; missing {missing}")


def load_quality(name: str, known_doc_ids: set[str] | None = None) -> list[QualityCase]:
    cases = []
    for row in _read_jsonl(FILES[name]):
        where = f"{name}:{row.get('id')}"
        _check_common(row, where)
        if row.get("category") not in CATEGORIES:
            raise DatasetError(f"{where}: category must be one of {CATEGORIES}")
        refuse = bool(row.get("should_refuse", False))
        sources = tuple(row.get("expected_sources", []))
        if refuse == bool(sources):
            raise DatasetError(f"{where}: give expected_sources, or set should_refuse (not both)")
        if known_doc_ids is not None and set(sources) - known_doc_ids:
            raise DatasetError(f"{where}: unknown doc_id(s) {sorted(set(sources) - known_doc_ids)}")
        cases.append(QualityCase(
            id=row["id"], language=row["language"], category=row["category"], question=row["question"],
            expected_answer=row.get("expected_answer", ""), expected_sources=sources,
            must_include=tuple(tuple(group) for group in row.get("must_include", [])),
            must_not_include=tuple(row.get("must_not_include", [])),
            should_refuse=refuse, history=_history(row), role=row.get("role"),
        ))
    _check_pairs([c.id for c in cases], name)
    return cases


def load_leak() -> list[LeakCase]:
    cases = []
    for row in _read_jsonl(FILES["leak"]):
        where = f"leak:{row.get('id')}"
        _check_common(row, where)
        if row.get("profile") not in ("internal", "external"):
            raise DatasetError(f"{where}: profile must be internal or external")
        if row.get("attack") not in ATTACKS:
            raise DatasetError(f"{where}: attack must be one of {ATTACKS}")
        if not (row.get("forbidden_collections") or row.get("forbidden_docs")):
            raise DatasetError(f"{where}: give forbidden_collections and/or forbidden_docs")
        cases.append(LeakCase(
            id=row["id"], language=row["language"], profile=row["profile"], attack=row["attack"],
            question=row["question"], forbidden_collections=tuple(row.get("forbidden_collections", [])),
            forbidden_strings=tuple(row.get("forbidden_strings", [])), role=row.get("role"),
            requires=row.get("requires"), history=_history(row),
            forbidden_docs=tuple(row.get("forbidden_docs", [])),
        ))
    _check_pairs([c.id for c in cases], "leak")
    return cases
