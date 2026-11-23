"""Chat log, feedback and analytics (no models needed)."""

import sqlite3

import pytest

from rag.chatlog import ChatLog, new_answer_id, normalize_question, redact
from rag.pipeline import NOT_FOUND, Answer, company_contacts
from rag.profiles import load_profile
from rag.config import get_settings
from rag.store import Hit

WHATSAPP = "+62 811-0000-1234"


def hit(doc_id: str, section: str = "", collection: str = "public") -> Hit:
    return Hit(collection=collection, score=1.0, chunk_key=doc_id, doc_id=doc_id, title=doc_id, language="en",
               version=1, status="current", last_updated="2026-01-01", source=f"{doc_id}.md", text="x",
               section=section)


def answer(text="ok [1]", status="answered", profile="external", tools=(), sources=None, cited=(1,)):
    return Answer(text=text, sources=sources if sources is not None else [hit("faq", "Returns")],
                  cited=list(cited), latency_ms=900, profile=profile, model="m", status=status,
                  tools_used=list(tools))


@pytest.fixture
def log(tmp_path):
    return ChatLog(tmp_path / "chats.sqlite", keep=("help@brightnest.co.id", WHATSAPP))


def test_redact_blanks_customer_contacts_but_keeps_the_company_ones():
    text = (f"My email is rina.setiawati@example.com, call 0812-3456-7890 or +62 812 3456 7890. "
            f"Support: help@brightnest.co.id, WhatsApp {WHATSAPP}.")
    out = redact(text, keep=("help@brightnest.co.id", WHATSAPP))
    assert "rina.setiawati" not in out and "3456" not in out
    assert out.count("[email]") == 1 and out.count("[phone]") == 2
    assert "help@brightnest.co.id" in out and WHATSAPP in out


@pytest.mark.parametrize("text", ["Order BN-2026-104233", "Rp200.000 and Rp2.500.000", "ticket IT-2026-0001",
                                  "on 2026-11-25", "tracking JNE7783211045", "ext. 2100"])
def test_redact_leaves_order_numbers_prices_and_dates_alone(text):
    assert redact(text) == text


def test_company_contacts_come_from_the_profile():
    contacts = company_contacts(load_profile("external", get_settings().profiles_dir))
    assert "help@brightnest.co.id" in contacts
    assert any("811-0000-1234" in c for c in contacts)


def test_record_stores_redacted_text_and_no_username(log):
    a = answer(text="Your order is shipped [1]. We wrote to rina.setiawati@example.com.")
    log.record("a" * 32, "Where is BN-2026-104233? rina.setiawati@example.com", a)
    row = sqlite3.connect(log.path).execute("SELECT question, answer, cited_docs FROM chats").fetchone()
    assert "rina" not in row[0] and "rina" not in row[1]
    assert "BN-2026-104233" in row[0]
    assert row[2] == '["faq > Returns"]'
    columns = [c[1] for c in sqlite3.connect(log.path).execute("PRAGMA table_info(chats)")]
    assert "username" not in columns


def test_feedback_needs_a_known_answer_and_a_second_vote_replaces_the_first(log):
    answer_id = new_answer_id()
    log.record(answer_id, "q", answer())
    assert log.add_feedback("f" * 32, 1) is False
    assert log.add_feedback(answer_id, 1)
    assert log.add_feedback(answer_id, -1, "wrong store")
    report = log.analytics("external")
    assert report["feedback"]["up"] == 0 and report["feedback"]["down"] == 1
    assert report["disliked"][0]["comment"] == "wrong store"
    with pytest.raises(ValueError):
        log.add_feedback(answer_id, 5)


def test_analytics_counts_per_assistant(log):
    log.record(new_answer_id(), "Where is my order?", answer(tools=["order_status"]))
    log.record(new_answer_id(), "where is my ORDER", answer(tools=["order_status"]))
    log.record(new_answer_id(), "Do you sell cars?", answer(text="I don't know.", status=NOT_FOUND, sources=[],
                                                            cited=()))
    log.record(new_answer_id(), "I want to talk to a person", answer(tools=["support_handoff"]))
    log.record(new_answer_id(), "How many leave days?", answer(profile="internal"))

    r = log.analytics("external")
    assert r["questions"] == 4
    assert r["top_questions"][0] == {"question": "where is my ORDER", "count": 2, "language": "en"}
    assert r["unanswered"] == [{"question": "Do you sell cars?", "count": 1, "language": "en"}]
    assert r["unanswered_rate"] == 0.25
    assert r["handoff_rate"] == 0.5          # the unanswered one + the support hand-off
    assert r["tools"] == {"order_status": 2, "support_handoff": 1}
    assert r["top_docs"][0] == ("faq > Returns", 3)
    assert log.analytics("internal")["questions"] == 1


def test_tool_results_are_not_counted_as_documents(log):
    a = answer(sources=[hit("tool:order_status", collection="tool")], tools=["order_status"])
    log.record(new_answer_id(), "q", a)
    assert log.analytics("external")["top_docs"] == []


def test_normalize_question():
    assert normalize_question("  Where is my ORDER?? ") == "where is my order"
