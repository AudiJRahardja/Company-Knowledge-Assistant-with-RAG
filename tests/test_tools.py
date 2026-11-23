"""Phase 5 tools, on a fresh temporary database. No models or LLM needed."""

from datetime import date

import pytest

from rag.chunking import Chunk
from rag.documents import Document
from rag.mockdb import create
from rag.pipeline import ANSWERED, NEEDS_INFO, Assistant
from rag.profiles import Profile, ProfileError, _validate
from rag.retrieval import RetrievalOptions
from rag.store import IndexWriter, KnowledgeStore, close_all
from rag.tools import ToolContext, Toolbox, parse_tool_call

TODAY = date(2026, 11, 23)
RIZKY = ToolContext(username="rizky", client_id="rizky", today=TODAY)
VISITOR = ToolContext(client_id="ip-1", today=TODAY)
INTERNAL = ("leave_balance", "my_manager", "create_ticket")
EXTERNAL = ("order_status", "store_hours", "support_handoff", "sales_lead")


@pytest.fixture
def box(tmp_path):
    return Toolbox(create(tmp_path / "mock.sqlite"), order_lookup_limit=3)


def run(box, tool, ctx=VISITOR, **args):
    allowed = INTERNAL if tool in INTERNAL else EXTERNAL
    return box.run(tool, args, ctx, allowed)


# ---------------------------------------------------------------- internal tools

def test_leave_balance_uses_the_signed_in_employee(box):
    r = run(box, "leave_balance", RIZKY)
    assert r.ok and "Rizky Pratama" in r.text and "Remaining annual leave: 8 days" in r.text
    assert "expired on 31 March 2026" in r.text


def test_carried_over_days_count_until_31_march(box):
    early = ToolContext(username="rizky", today=date(2026, 2, 1))
    assert "Remaining annual leave: 10 days" in run(box, "leave_balance", early).text


def test_employee_tools_need_a_signed_in_user(box):
    r = run(box, "leave_balance", VISITOR)
    assert not r.ok and "signed-in employee" in r.text


def test_the_llm_cannot_ask_for_someone_elses_balance(box):
    # Extra arguments are ignored: leave_balance takes none, so "username" never reaches the database.
    r = box.run("leave_balance", {"username": "maya"}, RIZKY, INTERNAL)
    assert "Rizky Pratama" in r.text and "Maya" not in r.text


def test_my_manager(box):
    assert "Andi Wijaya, Store Manager, Kemang" in run(box, "my_manager", RIZKY).text


def test_tickets_get_numbered_per_team(box):
    first = run(box, "create_ticket", RIZKY, team="it", summary="My laptop screen is broken.")
    second = run(box, "create_ticket", RIZKY, team="IT", summary="Printer jam at Kemang.")
    assert "IT-2026-0001" in first.text and "IT-2026-0002" in second.text
    assert "HR-2026-0001" in run(box, "create_ticket", RIZKY, team="hr", summary="Update bank account").text
    assert not run(box, "create_ticket", RIZKY, team="finance", summary="x").ok
    assert not run(box, "create_ticket", RIZKY, team="it", summary="").ok


# ---------------------------------------------------------------- order status

def test_order_status_with_the_right_email(box):
    r = run(box, "order_status", order_id="bn-2026-104233", email="Rina.Setiawati@example.com")
    assert r.ok and "shipped" in r.text and "JNE7783211045" in r.text and "Bandung" in r.text
    assert "Cempaka" not in r.text  # never the street address


def test_wrong_email_and_unknown_order_look_the_same(box):
    wrong_email = run(box, "order_status", order_id="BN-2026-104233", email="someone@example.com").text
    unknown = run(box, "order_status", order_id="BN-2026-999999", email="rina.setiawati@example.com").text
    garbage = run(box, "order_status", order_id="DROP TABLE orders", email="x@example.com").text
    assert wrong_email == unknown == garbage
    assert "Rina" not in wrong_email and "shipped" not in wrong_email


def test_failed_lookups_are_limited(box):
    for _ in range(3):
        run(box, "order_status", order_id="BN-2026-000001", email="guess@example.com")
    right = run(box, "order_status", order_id="BN-2026-104233", email="rina.setiawati@example.com")
    assert "Too many failed order lookups" in right.text and "shipped" not in right.text
    other_visitor = ToolContext(client_id="ip-2", today=TODAY)
    assert "shipped" in box.run("order_status", {"order_id": "BN-2026-104233",
                                                 "email": "rina.setiawati@example.com"}, other_visitor, EXTERNAL).text


def test_order_status_needs_both_fields(box):
    assert not run(box, "order_status", order_id="BN-2026-104233").ok


# ---------------------------------------------------------------- store hours

@pytest.mark.parametrize("store, day, expected", [
    ("Dago", "2026-06-07", "Sunday 07 June 2026: open 10:00-20:00"),
    ("dago store", "2026-06-01", "Monday 01 June 2026: open 09:00-21:00"),   # Pancasila Day: normal hours
    ("Pasteur Outlet", "2026-06-01", "closed (the store is closed on Mondays)"),
    ("Kemang", "2026-03-20", "closed (first day of Idul Fitri)"),
    ("Kemang", "2026-03-21", "open 13:00-21:00 (second day of Idul Fitri)"),
    ("Darmo", "2026-12-24", "open 10:00-18:00"),
    ("BSD", "today", "Monday 23 November 2026: open 10:00-21:00"),
    ("BSD", "tomorrow", "Tuesday 24 November 2026"),
])
def test_store_hours(box, store, day, expected):
    r = run(box, "store_hours", store=store, date=day)
    assert r.ok and expected in r.text


def test_store_hours_for_a_city_lists_every_store_there(box):
    r = run(box, "store_hours", store="Bandung", date="2026-06-07")
    assert "Dago store" in r.text and "Pasteur Outlet store" in r.text


def test_store_hours_bad_input(box):
    assert not run(box, "store_hours", store="Bali", date="today").ok
    assert not run(box, "store_hours", store="Dago", date="next Friday").ok


# ---------------------------------------------------------------- hand-offs and leads

def test_support_handoff_and_sales_lead(box):
    r = run(box, "support_handoff", name="Budi", contact="budi@example.com", message="Question about a refund")
    assert r.ok and "CS-2026-0001" in r.text
    lead = run(box, "sales_lead", name="Ayu", company="Hotel Melati", contact="+62 812 3456 7890",
               message="200 bath towels")
    assert lead.ok and "LEAD-2026-0001" in lead.text
    assert not run(box, "support_handoff", name="Budi", contact="call me", message="hi").ok
    assert not run(box, "sales_lead", name="Ayu", company="", contact="ayu@example.com", message="x").ok


def test_unknown_or_unlisted_tools_are_refused(box):
    assert not box.run("drop_tables", {}, VISITOR, EXTERNAL).ok
    assert not box.run("leave_balance", {}, RIZKY, EXTERNAL).ok  # not on this assistant's list


# ---------------------------------------------------------------- parsing and profiles

def test_parse_tool_call():
    assert parse_tool_call('{"tool": "my_manager", "args": {}}') == ("my_manager", {})
    assert parse_tool_call('```json\n{"tool": "store_hours", "args": {"store": "Dago"}}\n```') == (
        "store_hours", {"store": "Dago"})
    assert parse_tool_call("Your manager is Andi [1].") is None
    assert parse_tool_call('Sure! {"tool": "my_manager"}') is None  # only a reply that is ONLY the JSON
    assert parse_tool_call('{"answer": "hi"}') is None


def test_customer_assistant_cannot_get_employee_tools():
    p = Profile(name="p", display_name="P", description="", auth="anonymous", collections=("public",), top_k=5,
                system_prompt="x", fallback_message={"en": "y"}, tools=("store_hours", "leave_balance"))
    with pytest.raises(ProfileError):
        _validate(p)


# ---------------------------------------------------------------- the flow inside the assistant

class SequenceLLM:
    name = "fake"
    last_model = "fake-1"

    def __init__(self, *replies):
        self.replies = list(replies)
        self.prompts = []

    def generate(self, system, messages):
        self.prompts.append(messages[-1].content)
        return self.replies.pop(0)


class FakeEmbedder:
    def embed_query(self, text):
        return [1.0, 0.0, 0.0, 0.0]


@pytest.fixture
def public_store(tmp_path):
    doc = Document(doc_id="faq", title="FAQ", visibility="public", allowed_roles=(), language="en",
                   audience="x", version=1, status="current", last_updated="2026-01-01", body="x",
                   source="public/en/faq.md")
    IndexWriter(tmp_path).rebuild("public", [Chunk(doc, 0, "Track orders in the app.")], [[1, 0, 0, 0]], 4)
    yield KnowledgeStore(["public"], tmp_path)
    close_all()


def customer_assistant(store, box, llm):
    p = Profile(name="external", display_name="C", description="", auth="anonymous", collections=("public",),
                top_k=5, system_prompt="Help customers.", fallback_message={"en": "Contact us."}, tools=EXTERNAL)
    return Assistant(p, store, FakeEmbedder(), llm, options=RetrievalOptions(hybrid=False, split_questions=False),
                     toolbox=box)


def test_tool_result_becomes_a_cited_source(public_store, box):
    llm = SequenceLLM('{"tool": "order_status", "args": {"order_id": "BN-2026-104233", '
                      '"email": "rina.setiawati@example.com"}}',
                      "Your order has been shipped with JNE [2].")
    answer = customer_assistant(public_store, box, llm).ask("Where is order BN-2026-104233? rina.setiawati@example.com",
                                                            context=VISITOR)
    assert answer.status == ANSWERED and answer.tools_used == ["order_status"] and answer.cited == [2]
    assert answer.sources[-1].doc_id == "tool:order_status" and "JNE" in answer.sources[-1].text
    assert "Source [2] is the result of the order_status tool" in llm.prompts[1]


def test_asking_the_user_for_missing_details(public_store, box):
    llm = SequenceLLM('{"tool": "ask_user", "args": {"question": "What e-mail address did you use?"}}')
    answer = customer_assistant(public_store, box, llm).ask("Where is my order BN-2026-104233?", context=VISITOR)
    assert answer.status == NEEDS_INFO and answer.text == "What e-mail address did you use?"


def test_only_one_tool_per_question(public_store, box):
    llm = SequenceLLM('{"tool": "store_hours", "args": {"store": "Dago", "date": "today"}}',
                      '{"tool": "order_status", "args": {}}')
    answer = customer_assistant(public_store, box, llm).ask("Is Dago open?", context=VISITOR)
    assert answer.tools_used == ["store_hours"] and "Contact us." in answer.text


def test_tool_questions_in_the_eval_match_the_mock_data(box):
    """The expected facts of the eval's tool questions really come out of the tools."""
    from eval.datasets import load_quality

    cases = {c.id: c for name in ("internal", "external") for c in load_quality(name) if c.category == "tool"}
    assert "Remaining annual leave: 8 days" in run(box, "leave_balance", RIZKY).text          # int-41
    assert "Andi Wijaya" in run(box, "my_manager", RIZKY).text                               # int-42
    order = run(box, "order_status", order_id="BN-2026-104233", email="rina.setiawati@example.com").text
    assert "JNE" in order and "2026-11-25" in order                                           # ext-37
    assert "closed" in run(box, "store_hours", store="Kemang", date="2026-03-20").text        # ext-38
    assert {c.expected_tool for c in cases.values()} == {
        "leave_balance", "my_manager", "create_ticket", "order_status", "store_hours", "ask_user",
        "support_handoff", "sales_lead"}


def test_tool_missing_details_means_needs_info(public_store, box):
    llm = SequenceLLM('{"tool": "order_status", "args": {"order_id": "BN-2026-104501"}}',
                      "Which e-mail address did you use for this order?")
    answer = customer_assistant(public_store, box, llm).ask("Where is my order BN-2026-104501?", context=VISITOR)
    assert answer.status == NEEDS_INFO and answer.tools_used == ["order_status"]
    assert llm.prompts[0].endswith("Answer in English.")


def test_asking_for_details_after_the_tool_said_something_is_missing(public_store, box):
    llm = SequenceLLM('{"tool": "order_status", "args": {"order_id": "BN-2026-104501"}}',
                      '{"tool": "ask_user", "args": {"question": "What e-mail did you use for the order?"}}')
    answer = customer_assistant(public_store, box, llm).ask("Where is my order BN-2026-104501?", context=VISITOR)
    assert answer.status == NEEDS_INFO and answer.text == "What e-mail did you use for the order?"
