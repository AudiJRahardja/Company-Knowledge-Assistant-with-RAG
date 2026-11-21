"""The full question -> answer flow for one assistant.

    question -> retrieve chunks (only from the profile's collections) -> build prompt -> LLM
             -> parse citations -> log -> Answer
"""

import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from rag.answer import build_system_prompt, build_user_message, extract_citations
from rag.config import Settings, get_settings
from rag.embeddings import Embedder
from rag.llm import LLMClient, Message, get_llm
from rag.profiles import Profile, load_profile
from rag.reranker import get_reranker
from rag.retrieval import RetrievalOptions, retrieve
from rag.rewrite import rewrite_question
from rag.store import Hit, KnowledgeStore

MAX_HISTORY_MESSAGES = 6  # last 3 question/answer pairs are sent along for follow-up questions


@dataclass
class Answer:
    text: str
    sources: list[Hit]   # everything retrieved, in the order numbered for the LLM
    cited: list[int]     # 1-based numbers of the sources the answer actually cites
    latency_ms: int
    profile: str
    model: str           # the LLM model that actually answered (may be a fallback model)
    search_query: str = ""  # what was searched for (a follow-up question is rewritten to stand alone)


class Assistant:
    def __init__(self, profile: Profile, store: KnowledgeStore, embedder: Embedder, llm: LLMClient | None,
                 log_path: Path | None = None, options: RetrievalOptions = RetrievalOptions()):
        if set(store.allowed) != set(profile.collections):
            raise ValueError("the store's collections must match the profile's collections exactly")
        self.profile = profile
        self.store = store
        self.embedder = embedder
        self.llm = llm
        self.log_path = log_path
        self.options = options
        self.system_prompt = build_system_prompt(profile)

    def ask(self, question: str, history: list[Message] | None = None) -> Answer:
        if self.llm is None:
            raise ValueError("this assistant was built for search only (no LLM)")
        started = time.perf_counter()
        query = self.search_query(question, history)
        hits = retrieve(self.store, self.embedder, query, self.profile.top_k, self.options)
        messages = list(history or [])[-MAX_HISTORY_MESSAGES:]
        messages.append(Message("user", build_user_message(question, hits)))
        text = self.llm.generate(self.system_prompt, messages)
        answer = Answer(
            text=text,
            sources=hits,
            cited=extract_citations(text, len(hits)),
            latency_ms=int((time.perf_counter() - started) * 1000),
            profile=self.profile.name,
            model=self.llm.last_model,
            search_query=query,
        )
        self._log(question, answer)
        return answer

    def search_query(self, question: str, history: list[Message] | None = None) -> str:
        """The question to search for: follow-ups are rewritten to stand on their own (needs the LLM)."""
        if history and self.options.rewrite and self.llm is not None:
            return rewrite_question(self.llm, question, list(history))
        return question

    def search(self, question: str, history: list[Message] | None = None) -> list[Hit]:
        """Only the retrieval part of `ask`: the chunks the LLM would see."""
        query = self.search_query(question, history)
        return retrieve(self.store, self.embedder, query, self.profile.top_k, self.options)

    def _log(self, question: str, answer: Answer) -> None:
        if self.log_path is None:
            return
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "profile": answer.profile,
            "llm": answer.model,
            "question": question,
            "search_query": answer.search_query,
            "answer": answer.text,
            "cited": answer.cited,
            "sources": [
                {k: v for k, v in asdict(h).items() if k != "text"} for h in answer.sources
            ],
            "latency_ms": answer.latency_ms,
        }
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def retrieval_options(s: Settings) -> RetrievalOptions:
    reranker = None
    if s.reranker_model and s.reranker_model.lower() != "none":
        reranker = get_reranker(s.reranker_model, s.embedding_device)
    return RetrievalOptions(hybrid=s.retrieval_hybrid, merge_twins=s.retrieval_merge_twins, reranker=reranker,
                            rewrite=s.query_rewrite, split_questions=s.split_questions)


def build_assistant(profile_name: str, embedder: Embedder | None = None, llm: LLMClient | None = None,
                    settings: Settings | None = None, search_only: bool = False) -> Assistant:
    """Create an assistant whose store can only open the collections listed in its profile.

    search_only=True skips creating the LLM client (used by the eval's --retrieval-only mode).
    """
    s = settings or get_settings()
    profile = load_profile(profile_name, s.profiles_dir)
    store = KnowledgeStore(profile.collections, s.storage_dir)
    return Assistant(
        profile=profile,
        store=store,
        embedder=embedder or Embedder(s.embedding_model, s.embedding_device),
        llm=None if search_only else (llm or get_llm(s)),
        log_path=s.log_path,
        options=retrieval_options(s),
    )
