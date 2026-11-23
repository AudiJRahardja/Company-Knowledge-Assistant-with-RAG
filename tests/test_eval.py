"""Tests for the eval tooling: scoring functions, judge parsing, caching, and the test sets themselves."""

import pytest

from eval.datasets import load_leak, load_quality
from eval.judge import parse_verdict
from eval.metrics import contains, facts_ok, found_strings, looks_like_refusal, retrieval_scores
from eval.run import CachedLLM
from rag.config import get_settings
from rag.documents import load_documents
from rag.llm import Message


@pytest.fixture(scope="module")
def docs():
    return load_documents(get_settings().docs_dir)


# ---------- string matching

@pytest.mark.parametrize("text, phrase", [
    ("Rp1.200.000 per malam", "1200000"),
    ("Rp 1,200,000 per night", "Rp1,200,000"),
    ("buka pukul 08.00-18.00", "08:00"),
    ("Core hours are 10:00–15:00", "15:00"),
    ("You get 12 working days", "12"),
    ("Tutup pada hari Senin", "tutup"),
])
def test_contains_matches_across_formats(text, phrase):
    assert contains(text, phrase)


@pytest.mark.parametrize("text, phrase", [
    ("about 120 stores", "12"),           # numbers must match whole
    ("Rp2,500,000", "250000"),
    ("Rp2.500.000", "500000"),
])
def test_contains_does_not_match_inside_other_numbers(text, phrase):
    assert not contains(text, phrase)


def test_facts_need_every_group():
    answer = "Up to 5 days, used by 31 Maret."
    assert facts_ok(answer, [["5"], ["31 March", "31 Maret"]])
    assert not facts_ok(answer, [["5"], ["30 June"]])
    assert facts_ok(answer, [])


def test_found_strings_lists_leaked_phrases():
    assert found_strings("The G5 band is Rp16.000.000 to Rp26.000.000", ["16000000", "65000000"]) == ["16000000"]


def test_retrieval_scores():
    s = retrieval_scores(["faq", "warranty", "faq"], ["warranty"])
    assert s == {"hit": True, "all": True, "rr": 0.5, "rank": 2}
    s = retrieval_scores(["faq", "warranty"], ["warranty", "products"])
    assert s["hit"] and not s["all"]
    assert retrieval_scores(["faq"], ["warranty"])["rr"] == 0.0


def test_refusal_keywords():
    assert looks_like_refusal("Sorry, I don't know that. Please contact HR.")
    assert looks_like_refusal("Maaf, saya tidak memiliki informasi tentang itu.")
    assert not looks_like_refusal("Standard delivery outside Java (di luar Jawa) takes 4-8 days [1].")


# ---------- judge parsing

def test_parse_verdict_reads_json_with_extra_text():
    v = parse_verdict('Sure:\n```json\n{"correct": "partial", "faithful": "yes", "relevant": "no", '
                      '"refused": false, "reason": "x"}\n```')
    assert (v.correct, v.faithful, v.relevant, v.refused) == (0.5, 1.0, 0.0, False)


def test_parse_verdict_handles_garbage():
    v = parse_verdict("I cannot grade this")
    assert v.correct is None and "not JSON" in v.reason


# ---------- cache

class CountingLLM:
    name = "fake"
    last_model = "fake-1"

    def __init__(self):
        self.calls = 0

    def generate(self, system, messages):
        self.calls += 1
        return f"reply {self.calls}"


def test_cached_llm_reuses_replies_across_runs(tmp_path):
    path = tmp_path / "cache.jsonl"
    inner = CountingLLM()
    first = CachedLLM(inner, delay_s=0, path=path)
    msgs = [Message("user", "hi")]
    assert first.generate("sys", msgs) == "reply 1"
    assert first.generate("sys", msgs) == "reply 1"
    assert first.generate("other system prompt", msgs) == "reply 2"
    second = CachedLLM(inner, delay_s=0, path=path)  # a new run reads the cache file
    assert second.generate("sys", msgs) == "reply 1"
    assert inner.calls == 2 and second.cache_hits == 1


# ---------- the test sets themselves

@pytest.mark.parametrize("name", ["internal", "external"])
def test_quality_sets_are_valid_and_bilingual(name, docs):
    cases = load_quality(name, {d.doc_id for d in docs})  # raises if invalid or missing a language pair
    assert len(cases) >= 50
    common = {"lookup", "multi_doc", "followup", "confusable", "unanswerable"}
    internal_only = {"outdated", "old_version", "role_access"}
    assert {c.category for c in cases} == common | (internal_only if name == "internal" else set())


@pytest.mark.parametrize("name", ["internal", "external"])
def test_expected_facts_really_are_in_the_expected_sources(name, docs):
    """Catches typos in the test set: every key fact must appear in one of the expected documents."""
    by_id: dict[str, str] = {}
    for d in docs:  # every version: "old_version" questions are answered from the outdated document
        by_id[d.doc_id] = by_id.get(d.doc_id, "") + "\n" + d.body
    for case in load_quality(name):
        text = "\n".join(by_id[s] for s in case.expected_sources)
        for group in case.must_include:
            assert any(contains(text, alt) for alt in group), f"{case.id}: none of {group} is in {case.expected_sources}"


def test_leak_canaries_are_not_in_allowed_documents(docs):
    """A forbidden string must not appear in documents the assistant may legitimately use, or an honest
    answer would be flagged as a leak."""
    allowed = {"external": {"public"}, "internal": {"public", "internal"}}
    for case in load_leak():
        text = "\n".join(d.body for d in docs if d.visibility in allowed[case.profile])
        assert not found_strings(text, case.forbidden_strings), case.id
        if case.forbidden_docs:  # a forbidden document must exist, or the test checks nothing
            assert set(case.forbidden_docs) <= {d.doc_id for d in docs}, case.id


def test_judge_sees_the_earlier_conversation():
    from eval.judge import judge

    class Recorder:
        name = "fake"
        last_model = "fake"
        prompt = ""

        def generate(self, system, messages):
            Recorder.prompt = messages[0].content
            return '{"correct": "no", "faithful": "yes", "relevant": "no", "refused": false, "reason": "x"}'

    history = (Message("user", "How do I get IT help?"), Message("assistant", "Use ServiceDesk [1]."))
    judge(Recorder(), "What are their opening hours?", "08:00-18:00", [], "Stores open 10:00", history)
    assert "User: How do I get IT help?" in Recorder.prompt
    assert "Chatbot: Use ServiceDesk [1]." in Recorder.prompt
