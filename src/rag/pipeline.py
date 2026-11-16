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
from rag.retrieval import retrieve
from rag.store import Hit, KnowledgeStore

MAX_HISTORY_MESSAGES = 6  # last 3 question/answer pairs are sent along for follow-up questions


@dataclass
class Answer:
    text: str
    sources: list[Hit]   # everything retrieved, in the order numbered for the LLM
    cited: list[int]     # 1-based numbers of the sources the answer actually cites
    latency_ms: int
    profile: str


class Assistant:
    def __init__(self, profile: Profile, store: KnowledgeStore, embedder: Embedder, llm: LLMClient,
                 log_path: Path | None = None):
        if set(store.allowed) != set(profile.collections):
            raise ValueError("the store's collections must match the profile's collections exactly")
        self.profile = profile
        self.store = store
        self.embedder = embedder
        self.llm = llm
        self.log_path = log_path
        self.system_prompt = build_system_prompt(profile)

    def ask(self, question: str, history: list[Message] | None = None) -> Answer:
        started = time.perf_counter()
        hits = retrieve(self.store, self.embedder, question, self.profile.top_k)
        messages = list(history or [])[-MAX_HISTORY_MESSAGES:]
        messages.append(Message("user", build_user_message(question, hits)))
        text = self.llm.generate(self.system_prompt, messages)
        answer = Answer(
            text=text,
            sources=hits,
            cited=extract_citations(text, len(hits)),
            latency_ms=int((time.perf_counter() - started) * 1000),
            profile=self.profile.name,
        )
        self._log(question, answer)
        return answer

    def _log(self, question: str, answer: Answer) -> None:
        if self.log_path is None:
            return
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "profile": answer.profile,
            "llm": self.llm.name,
            "question": question,
            "answer": answer.text,
            "cited": answer.cited,
            "sources": [
                {k: v for k, v in asdict(h).items() if k != "text"} for h in answer.sources
            ],
            "latency_ms": answer.latency_ms,
        }
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_assistant(profile_name: str, embedder: Embedder | None = None, llm: LLMClient | None = None,
                    settings: Settings | None = None) -> Assistant:
    """Create an assistant whose store can only open the collections listed in its profile."""
    s = settings or get_settings()
    profile = load_profile(profile_name, s.profiles_dir)
    store = KnowledgeStore(profile.collections, s.storage_dir)
    return Assistant(
        profile=profile,
        store=store,
        embedder=embedder or Embedder(s.embedding_model, s.embedding_device),
        llm=llm or get_llm(s),
        log_path=s.log_path,
    )
