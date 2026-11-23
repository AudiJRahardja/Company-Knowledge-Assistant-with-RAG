"""The full question -> answer flow for one assistant and one user role.

    question
      -> input checks: length, prompt-injection screening (also of earlier messages)   -> blocked
      -> rewrite follow-ups, retrieve chunks (only from collections this role may read)
      -> nothing found                                                                  -> not found + hand-off
      -> LLM writes the answer from numbered sources, or asks for a tool (see tools.py): then the tool runs,
         its result becomes one more numbered source, and the LLM answers from that
      -> output checks: no system prompt, no fingerprints of unreadable documents       -> blocked
      -> "I don't know" answers are marked as not found and get the hand-off contact
      -> log -> Answer
"""

import re
import time
from dataclasses import dataclass, field
from pathlib import Path

from rag.chatlog import ChatLog, new_answer_id
from rag.answer import GROUNDING_RULES, build_system_prompt, build_user_message, extract_citations
from rag.config import Settings, get_settings
from rag.embeddings import Embedder
from rag.fingerprint import LeakDetector, extract_terms, hash_term
from rag.guard import (BLOCKED_REPLY, check_question, detect_language, looks_like_injection, repeats_prompt,
                       shingles)
from rag.llm import LLMClient, Message, get_llm
from rag.profiles import Profile, load_profile
from rag.reranker import get_reranker
from rag.retrieval import RetrievalOptions, retrieve
from rag.rewrite import rewrite_question
from rag.store import Hit, KnowledgeStore
from rag.tools import ASK_USER, ToolContext, Toolbox, describe_tools, parse_tool_call, today_for, tool_hit

MAX_HISTORY_MESSAGES = 6  # last 3 question/answer pairs are sent along for follow-up questions

# Answer.status values
ANSWERED = "answered"
NOT_FOUND = "not_found"            # the documents don't cover it: hand off to a person
BLOCKED_INPUT = "blocked_input"    # the question tried to override the assistant's rules
BLOCKED_OUTPUT = "blocked_output"  # the answer failed the output check and was replaced
NEEDS_INFO = "needs_info"          # the assistant asks the user for something a tool needs (order number...)


@dataclass
class Answer:
    text: str
    sources: list[Hit]   # everything retrieved, in the order numbered for the LLM
    cited: list[int]     # 1-based numbers of the sources the answer actually cites
    latency_ms: int
    profile: str
    model: str           # the LLM model that actually answered (may be a fallback model); "" if none was called
    search_query: str = ""  # what was searched for (a follow-up question is rewritten to stand alone)
    status: str = ANSWERED
    language: str = "en"
    handoff: str = ""    # who to contact, when the assistant could not answer
    role: str | None = None
    flags: list[str] = field(default_factory=list)  # why something was blocked (for the log, not the user)
    tools_used: list[str] = field(default_factory=list)
    id: str = ""         # random id, used to attach thumbs up / down feedback to this answer


class Assistant:
    def __init__(self, profile: Profile, store: KnowledgeStore, embedder: Embedder, llm: LLMClient | None,
                 chat_log: ChatLog | None = None, options: RetrievalOptions = RetrievalOptions(),
                 role: str | None = None, leak_detector: LeakDetector | None = None, max_question_chars: int = 1000,
                 toolbox: Toolbox | None = None):
        if set(store.allowed) != set(profile.collections_for(role)):
            raise ValueError("the store's collections must match what the profile allows for this role")
        if store.role != role:
            raise ValueError("the store must be opened for the same role as the assistant")
        self.profile = profile
        self.store = store
        self.embedder = embedder
        self.llm = llm
        self.chat_log = chat_log
        self.options = options
        self.role = role
        self.leak_detector = leak_detector
        self.max_question_chars = max_question_chars
        self.toolbox = toolbox if profile.tools else None
        self.system_prompt = build_system_prompt(profile)
        if self.toolbox:
            self.system_prompt += "\n" + describe_tools(profile.tools) + "\n"
        # 8-word pieces of the instructions, to notice if an answer starts quoting them. The fallback texts
        # are left out: the assistant is supposed to repeat those.
        self._prompt_shingles = shingles(f"{profile.system_prompt}\n{GROUNDING_RULES}")
        self._own_terms = {hash_term(t) for text in profile.fallback_message.values() for t in extract_terms(text)}

    # ---------------------------------------------------------------- main flow

    def ask(self, question: str, history: list[Message] | None = None,
            context: ToolContext | None = None) -> Answer:
        """Answer one question. `context` says who is asking (signed-in user, visitor id) for the tools;
        it must come from the login token / request, never from anything the user typed."""
        if self.llm is None:
            raise ValueError("this assistant was built for search only (no LLM)")
        started = time.perf_counter()
        question = check_question(question, self.max_question_chars)  # raises InputRejected
        history = list(history or [])[-MAX_HISTORY_MESSAGES:]
        language = detect_language(question)

        def finish(answer: Answer) -> Answer:
            answer.latency_ms = int((time.perf_counter() - started) * 1000)
            answer.id = new_answer_id()
            if self.chat_log is not None:
                self.chat_log.record(answer.id, question, answer)
            return answer

        def blank(status: str, text: str, flags: list[str] | None = None, query: str = "") -> Answer:
            return Answer(text=text, sources=[], cited=[], latency_ms=0, profile=self.profile.name, model="",
                          search_query=query or question, status=status, language=language,
                          handoff=self.profile.fallback(language), role=self.role, flags=flags or [])

        # 1. Input screening: the question and every earlier user message.
        if looks_like_injection(question) or any(m.role == "user" and looks_like_injection(m.content)
                                                 for m in history):
            return finish(blank(BLOCKED_INPUT, BLOCKED_REPLY[language], ["prompt_injection"]))

        # 2. Search.
        query = self.search_query(question, history)
        hits = retrieve(self.store, self.embedder, query, self.profile.top_k, self.options)
        if not hits and not self.toolbox:
            return finish(blank(NOT_FOUND, self._not_found_text(language), query=query))

        # 3. Answer. With tools, the first reply may be a tool request instead of an answer.
        first_message = build_user_message(question, hits, query, language)
        text = self.llm.generate(self.system_prompt, history + [Message("user", first_message)])
        tools_used: list[str] = []
        needs_info = False
        call = parse_tool_call(text) if self.toolbox else None
        if call and call[0] == ASK_USER:
            # The LLM needs more information from the user first (e.g. the e-mail for an order).
            ask = str(call[1].get("question") or "").strip() or text
            return finish(Answer(text=ask, sources=hits, cited=[], latency_ms=0, profile=self.profile.name,
                                 model=self.llm.last_model, search_query=query, status=NEEDS_INFO,
                                 language=language, role=self.role, tools_used=[ASK_USER]))
        if call:
            ctx = context or ToolContext()
            result = self.toolbox.run(call[0], call[1], ctx, self.profile.tools)
            tools_used.append(result.tool)
            needs_info = result.needs_info  # e.g. order_status called without the e-mail: we'll be asking for it
            hits = hits + [tool_hit(result, today_for(ctx))]
            n = len(hits)
            follow_up = (build_user_message(question, hits, query, language) +
                         f"\n\nSource [{n}] is the result of the {result.tool} tool you asked for. Answer the user "
                         f"now from the sources, citing [{n}]. Do not request another tool.")
            text = self.llm.generate(self.system_prompt, history + [Message("user", follow_up)])
            second = parse_tool_call(text)
            if second and second[0] == ASK_USER:  # e.g. the tool said the e-mail is missing: ask for it
                text = str(second[1].get("question") or "").strip() or self._not_found_text(language)
                needs_info = True
            elif second:  # only one tool per question; a second request is not followed
                text = self._not_found_text(language)
        answer = Answer(text=text, sources=hits, cited=extract_citations(text, len(hits)), latency_ms=0,
                        profile=self.profile.name, model=self.llm.last_model, search_query=query,
                        language=language, role=self.role, tools_used=tools_used)

        # 4. Output checks.
        flags = self.output_problems(text)
        if flags:
            blocked = blank(BLOCKED_OUTPUT, BLOCKED_REPLY[language], flags, query)
            blocked.model = answer.model
            return finish(blocked)

        # 5. A tool still needs details from the user: the answer is the question for those details.
        if needs_info:
            answer.status = NEEDS_INFO
        # 6. No citation means the answer is not based on any document ("I don't know", "please ask HR",
        #    or an off-topic reply): mark it as not found and attach the hand-off contact.
        elif not answer.cited:
            answer.status = NOT_FOUND
            answer.handoff = self.profile.fallback(language)
        return finish(answer)

    def output_problems(self, text: str) -> list[str]:
        """Reasons this answer must not be shown (empty list = fine)."""
        flags = []
        if repeats_prompt(text, self._prompt_shingles):
            flags.append("repeats_system_prompt")
        if self.leak_detector is not None:
            leaked = [t for t in self.leak_detector.leaked_terms(text) if hash_term(t) not in self._own_terms]
            if leaked:
                flags.append("leak:" + ",".join(t.split(":", 1)[0] for t in leaked))
        return flags

    def _not_found_text(self, language: str) -> str:
        sorry = ("Maaf, saya tidak menemukan informasi tentang itu di dokumen kami." if language == "id"
                 else "Sorry, I couldn't find information about that in our documents.")
        return f"{sorry} {self.profile.fallback(language)}"

    # ---------------------------------------------------------------- search

    def search_query(self, question: str, history: list[Message] | None = None) -> str:
        """The question to search for: follow-ups are rewritten to stand on their own (needs the LLM)."""
        if history and self.options.rewrite and self.llm is not None:
            return rewrite_question(self.llm, question, list(history))
        return question

    def search(self, question: str, history: list[Message] | None = None) -> list[Hit]:
        """Only the retrieval part of `ask`: the chunks the LLM would see."""
        query = self.search_query(question, history)
        return retrieve(self.store, self.embedder, query, self.profile.top_k, self.options)


def retrieval_options(s: Settings) -> RetrievalOptions:
    reranker = None
    if s.reranker_model and s.reranker_model.lower() != "none":
        reranker = get_reranker(s.reranker_model, s.embedding_device)
    return RetrievalOptions(hybrid=s.retrieval_hybrid, merge_twins=s.retrieval_merge_twins, reranker=reranker,
                            rewrite=s.query_rewrite, split_questions=s.split_questions,
                            hide_superseded=s.hide_superseded)


def company_contacts(profile: Profile) -> tuple[str, ...]:
    """The company's own emails and phone numbers (in the hand-off text): these stay readable in the log."""
    text = " ".join(profile.fallback_message.values())
    return tuple(re.findall(r"[\w.+-]+@[\w.-]+\w", text) + re.findall(r"\+?\d[\d\s-]{7,}\d", text))


def default_role(profile: Profile) -> str | None:
    return "employee" if profile.auth == "login" else None


_toolboxes: dict[Path, Toolbox] = {}


def get_toolbox(s: Settings) -> Toolbox:
    """One toolbox per database, shared by all assistants (so the order-lookup limits are shared too)."""
    if s.mock_db_path not in _toolboxes:
        _toolboxes[s.mock_db_path] = Toolbox(s.mock_db_path, s.order_lookup_limit)
    return _toolboxes[s.mock_db_path]


def build_assistant(profile_name: str, embedder: Embedder | None = None, llm: LLMClient | None = None,
                    settings: Settings | None = None, search_only: bool = False,
                    role: str | None = None, toolbox: Toolbox | None = None) -> Assistant:
    """Create an assistant for one profile and one user role.

    The store only opens the collections that this profile allows for this role, and restricted chunks are
    filtered by role inside the store. Anonymous profiles (the customer assistant) never have a role.
    search_only=True skips creating the LLM client (used by the eval's --retrieval-only mode).
    """
    s = settings or get_settings()
    profile = load_profile(profile_name, s.profiles_dir)
    if profile.auth == "anonymous" and role is not None:
        raise ValueError(f"the {profile.name} assistant is anonymous and takes no role")
    role = role or default_role(profile)
    collections = profile.collections_for(role)
    store = KnowledgeStore(collections, s.storage_dir, role=role)
    return Assistant(
        profile=profile,
        store=store,
        embedder=embedder or Embedder(s.embedding_model, s.embedding_device),
        llm=None if search_only else (llm or get_llm(s)),
        chat_log=ChatLog(s.chat_db_path, keep=company_contacts(profile)),
        options=retrieval_options(s),
        role=role,
        leak_detector=LeakDetector.for_collections(s.security_dir / "fingerprints.json", collections),
        max_question_chars=s.max_question_chars,
        toolbox=toolbox or get_toolbox(s),
    )
