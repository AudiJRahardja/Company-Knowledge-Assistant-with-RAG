"""Keyword search (BM25) and rank fusion. No models needed."""

from rag.keyword import BM25, tokenize
from rag.retrieval import rrf
from rag.store import Hit


def test_tokenize_joins_thousand_separators():
    assert tokenize("Rp250.000 per day, Rp 1,200,000") == ["rp250000", "per", "day", "rp", "1200000"]


def test_bm25_prefers_rare_matching_words():
    texts = ["the warranty covers the blender", "the store opens at ten", "the the the"]
    index = BM25(texts)
    assert index.top("blender warranty", 3)[0][0] == 0
    assert index.top("opens", 3) == [(1, index.scores("opens")[1])]
    assert index.top("nothing matches", 3) == []


def hit(key: str) -> Hit:
    return Hit(collection="public", score=0.0, chunk_key=key, doc_id=key, title="t", language="en", version=1,
               status="current", last_updated="2026-01-01", source="s", text="x")


def test_rrf_rewards_chunks_found_by_both_methods():
    fused = rrf([hit("a"), hit("b"), hit("c")], [hit("c"), hit("d")])
    assert fused[0].chunk_key == "c"          # 3rd in one list and 1st in the other beats 1st in only one
    assert {h.chunk_key for h in fused} == {"a", "b", "c", "d"}
    assert fused[0].score > fused[1].score


class ScriptedLLM:
    name = "fake"
    last_model = "fake"

    def __init__(self, reply):
        self.reply = reply
        self.prompts = []

    def generate(self, system, messages):
        self.prompts.append(messages[-1].content)
        return self.reply


def test_rewrite_only_happens_with_history():
    from rag.llm import Message
    from rag.rewrite import rewrite_question

    llm = ScriptedLLM("How many unused annual leave days can I carry over?")
    assert rewrite_question(llm, "How many leave days?", []) == "How many leave days?"
    assert llm.prompts == []  # no conversation, no extra LLM call

    history = [Message("user", "How many days of annual leave do I get?"), Message("assistant", "12 days [1].")]
    assert rewrite_question(llm, "And how many can I carry over?", history).startswith("How many unused")
    assert "User: How many days of annual leave do I get?" in llm.prompts[0]


def test_bad_rewrite_falls_back_to_the_original_question():
    from rag.llm import Message
    from rag.rewrite import rewrite_question

    history = [Message("user", "hi"), Message("assistant", "hello")]
    assert rewrite_question(ScriptedLLM(""), "And for Medan?", history) == "And for Medan?"
    assert rewrite_question(ScriptedLLM("x" * 1000), "And for Medan?", history) == "And for Medan?"


def test_split_question_finds_the_parts():
    from rag.retrieval import split_question

    assert split_question("I just joined. How long is probation, and when can I start using annual leave?") == [
        "I just joined. How long is probation", "when can I start using annual leave"]
    assert split_question("Berapa lama cuti saya, dan bagaimana mendaftarkan bayi ke asuransi?") == [
        "Berapa lama cuti saya", "bagaimana mendaftarkan bayi ke asuransi"]
    assert split_question("What is the warranty on a blender?") == []
    assert split_question("Do you sell pots and pans?") == []           # "and" without a new question


def test_old_version_questions_are_recognised():
    from rag.retrieval import asks_about_old_version

    assert asks_about_old_version("What was the remote work policy in 2024?")
    assert asks_about_old_version("What were the previous core hours?")
    assert asks_about_old_version("Bagaimana aturan lama soal kerja dari rumah?")
    assert not asks_about_old_version("How many days can I work from home?")
    assert not asks_about_old_version("Berapa lama cuti melahirkan?")  # "lama" here means "long"
    assert not asks_about_old_version("When do I have to call in sick, before 09:00?")
