"""Chat log, feedback and analytics (Phase 6).

Every answer is saved in a small SQLite database (storage/logs/chats.sqlite) with a random id. The UI sends
thumbs up / down for that id, and the analytics page reads both tables per assistant:

    - top questions                        what people ask most
    - unanswered questions                 internal: gaps in the HR / IT docs; external: FAQ / website gaps
    - hand-off rate                        how often a person had to take over
    - feedback                             thumbs up / down, and the answers people disliked

Privacy
-------
- Email addresses and phone numbers are blanked before anything is saved ("[email]", "[phone]"). Customers
  type their email for order lookups; the log does not need it.
- The internal log keeps the role (employee / manager / hr), not the username: the analytics are about
  the documents, not about watching what a colleague asked.

Usage:  python -m rag.chatlog [--profile internal|external] [--days 30]
"""

import argparse
import json
import re
import sqlite3
import threading
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS chats (
    id          TEXT PRIMARY KEY,
    time        TEXT NOT NULL,        -- UTC, ISO format
    profile     TEXT NOT NULL,        -- internal | external
    role        TEXT,                 -- employee | manager | hr (internal only)
    language    TEXT,
    question    TEXT NOT NULL,        -- emails and phone numbers blanked
    search_query TEXT,
    status      TEXT NOT NULL,        -- answered | not_found | needs_info | blocked_input | blocked_output
    tools_used  TEXT NOT NULL,        -- JSON list
    cited_docs  TEXT NOT NULL,        -- JSON list of "doc_id > section" that the answer cited
    flags       TEXT NOT NULL,        -- JSON list: why something was blocked
    answer      TEXT NOT NULL,        -- blanked like the question
    model       TEXT,
    latency_ms  INTEGER
);
CREATE INDEX IF NOT EXISTS chats_profile_time ON chats (profile, time);
CREATE TABLE IF NOT EXISTS feedback (
    answer_id   TEXT PRIMARY KEY REFERENCES chats (id),
    time        TEXT NOT NULL,
    rating      INTEGER NOT NULL CHECK (rating IN (-1, 1)),
    comment     TEXT NOT NULL DEFAULT ''
);
"""

# Tools that pass the conversation to a person (a support agent, HR / IT, the sales team).
HANDOFF_TOOLS = ("support_handoff", "create_ticket", "sales_lead")

_EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
_PHONE = re.compile(r"(?:\+62|\b0)[\s-]?\d{2,4}(?:[\s-]?\d{3,4}){1,3}\b")
_SPACES = re.compile(r"\s+")
_PUNCT = re.compile(r"[^\w\s]")


def redact(text: str, keep: tuple[str, ...] = ()) -> str:
    """Blank email addresses and phone numbers, except the company's own contacts in `keep`."""
    def email(m: re.Match) -> str:
        return m.group(0) if m.group(0).lower() in keep else "[email]"

    def phone(m: re.Match) -> str:
        return m.group(0) if _digits(m.group(0)) in {_digits(k) for k in keep} else "[phone]"

    return _PHONE.sub(phone, _EMAIL.sub(email, text))


def _digits(text: str) -> str:
    return re.sub(r"\D", "", text)


def normalize_question(question: str) -> str:
    """For grouping: "Where is my ORDER?" and "where is my order" count as the same question."""
    return _SPACES.sub(" ", _PUNCT.sub(" ", question.lower())).strip()


class ChatLog:
    def __init__(self, path: Path, keep: tuple[str, ...] = ()):
        """keep: contact details that may stay in the log (the company's own hotline and emails)."""
        self.path = path
        self.keep = tuple(k.lower() for k in keep)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        with self._connect() as db:
            db.executescript(SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.path)
        db.row_factory = sqlite3.Row
        return db

    # ------------------------------------------------------------ writing

    def record(self, answer_id: str, question: str, answer) -> None:
        """Save one answer (a rag.pipeline.Answer)."""
        cited = [f"{h.doc_id} > {h.section}" if h.section else h.doc_id
                 for i, h in enumerate(answer.sources, start=1)
                 if i in answer.cited and h.collection != "tool"]
        row = (answer_id, _now(), answer.profile, answer.role, answer.language,
               redact(question, self.keep), redact(answer.search_query, self.keep), answer.status,
               json.dumps(answer.tools_used), json.dumps(cited), json.dumps(answer.flags),
               redact(answer.text, self.keep), answer.model, answer.latency_ms)
        with self._lock, self._connect() as db:
            db.execute("INSERT INTO chats VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)", row)

    def add_feedback(self, answer_id: str, rating: int, comment: str = "") -> bool:
        """Thumbs up (1) or down (-1) for an answer. A second vote replaces the first.
        Returns False if there is no such answer."""
        if rating not in (-1, 1):
            raise ValueError("rating must be 1 or -1")
        with self._lock, self._connect() as db:
            if db.execute("SELECT 1 FROM chats WHERE id = ?", (answer_id,)).fetchone() is None:
                return False
            db.execute("INSERT OR REPLACE INTO feedback VALUES (?,?,?,?)",
                       (answer_id, _now(), rating, redact(comment.strip(), self.keep)[:500]))
        return True

    # ------------------------------------------------------------ reading

    def analytics(self, profile: str, days: int | None = None, top: int = 10) -> dict:
        """Numbers for the analytics page of one assistant."""
        where, args = "c.profile = ?", [profile]
        if days:
            where += " AND c.time >= ?"
            args.append((datetime.now(timezone.utc) - timedelta(days=days)).isoformat(timespec="seconds"))
        with self._connect() as db:
            rows = db.execute(f"SELECT c.*, f.rating, f.comment FROM chats c "
                              f"LEFT JOIN feedback f ON f.answer_id = c.id WHERE {where} "
                              f"ORDER BY c.time DESC", args).fetchall()

        total = len(rows)
        status = _count(r["status"] for r in rows)
        tools = _count(t for r in rows for t in json.loads(r["tools_used"]))
        handed_over = sum(1 for r in rows
                          if r["status"] == "not_found" or set(json.loads(r["tools_used"])) & set(HANDOFF_TOOLS))
        up = sum(1 for r in rows if r["rating"] == 1)
        down = sum(1 for r in rows if r["rating"] == -1)

        return {
            "profile": profile,
            "questions": total,
            "status": status,
            "languages": _count(r["language"] for r in rows),
            "roles": _count(r["role"] for r in rows if r["role"]),
            "tools": tools,
            "answered_rate": _share(status.get("answered", 0), total),
            "unanswered_rate": _share(status.get("not_found", 0), total),
            "blocked_rate": _share(status.get("blocked_input", 0) + status.get("blocked_output", 0), total),
            "handoff_rate": _share(handed_over, total),
            "feedback": {"up": up, "down": down, "rated_share": _share(up + down, total),
                         "satisfaction": _share(up, up + down)},
            "avg_latency_ms": round(sum(r["latency_ms"] or 0 for r in rows) / total) if total else None,
            "top_questions": _grouped(rows, top),
            "unanswered": _grouped([r for r in rows if r["status"] == "not_found"], top),
            "top_docs": sorted(_count(d for r in rows for d in json.loads(r["cited_docs"])).items(),
                               key=lambda kv: -kv[1])[:top],
            "disliked": [{"time": r["time"], "question": r["question"], "answer": r["answer"],
                          "comment": r["comment"]} for r in rows if r["rating"] == -1][:top],
        }


def _grouped(rows, top: int) -> list[dict]:
    """Group the same question (case and punctuation ignored); most asked first."""
    groups: dict[str, dict] = {}
    for r in rows:  # newest first, so the example is the latest wording
        g = groups.setdefault(normalize_question(r["question"]),
                              {"question": r["question"], "count": 0, "language": r["language"]})
        g["count"] += 1
    return sorted(groups.values(), key=lambda g: -g["count"])[:top]


def _count(values) -> dict[str, int]:
    out: dict[str, int] = {}
    for v in values:
        out[v] = out.get(v, 0) + 1
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


def _share(part: int, whole: int) -> float | None:
    return round(part / whole, 3) if whole else None


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def new_answer_id() -> str:
    return uuid.uuid4().hex


def main() -> None:
    from rag.config import get_settings

    parser = argparse.ArgumentParser(description="Print the chat analytics of one assistant.")
    parser.add_argument("--profile", default="external", choices=["internal", "external"])
    parser.add_argument("--days", type=int, default=None)
    args = parser.parse_args()
    s = get_settings()
    if not s.chat_db_path.exists():
        raise SystemExit(f"No chats logged yet ({s.chat_db_path}).")
    report = ChatLog(s.chat_db_path).analytics(args.profile, args.days)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
