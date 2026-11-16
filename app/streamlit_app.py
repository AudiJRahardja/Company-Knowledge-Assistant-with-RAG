"""Phase 1 demo UI: one page, a switch between the two assistants.

Run:  streamlit run app/streamlit_app.py
"""

import streamlit as st

from rag.config import get_settings
from rag.embeddings import Embedder
from rag.llm import LLMError, Message, get_llm
from rag.pipeline import Answer, Assistant, build_assistant

ASSISTANTS = {
    "internal": "Internal: employees",
    "external": "External: customers",
}
EXAMPLES = {
    "internal": [
        "How many days of annual leave do I get?",
        "Berapa uang harian untuk perjalanan dinas ke Surabaya?",
        "How many days can I work from home?",
    ],
    "external": [
        "What time does the Dago store open on Sunday?",
        "Bisakah saya mengembalikan blender yang sudah dibuka?",
        "How long does shipping to Medan take?",
    ],
}

st.set_page_config(page_title="Brightnest Assistants", page_icon="🏠", layout="centered")


@st.cache_resource(show_spinner="Loading the embedding model...")
def get_embedder() -> Embedder:
    s = get_settings()
    return Embedder(s.embedding_model, s.embedding_device)


@st.cache_resource
def get_assistant(name: str) -> Assistant:
    return build_assistant(name, embedder=get_embedder(), llm=get_llm())


def render_sources(answer_sources: list[dict], cited: list[int]) -> None:
    if not answer_sources:
        return
    with st.expander(f"Sources ({len(cited)} cited, {len(answer_sources)} retrieved)"):
        for i, s in enumerate(answer_sources, start=1):
            mark = "✅ cited" if i in cited else "not cited"
            st.markdown(
                f"**[{i}] {s['title']}** · `{s['collection']}` · {s['language'].upper()} · "
                f"updated {s['last_updated']} · score {s['score']:.3f} · {mark}  \n"
                f"<small>{s['source']}</small>",
                unsafe_allow_html=True,
            )
            st.caption(s["text"][:400] + ("..." if len(s["text"]) > 400 else ""))


def to_message_dict(answer: Answer) -> dict:
    return {
        "role": "assistant",
        "content": answer.text,
        "cited": answer.cited,
        "latency_ms": answer.latency_ms,
        "sources": [
            {"title": h.title, "collection": h.collection, "language": h.language,
             "last_updated": h.last_updated, "score": h.score, "source": h.source, "text": h.text}
            for h in answer.sources
        ],
    }


# ---------- sidebar ----------
with st.sidebar:
    st.header("Brightnest Assistants")
    choice = st.radio("Assistant", list(ASSISTANTS), format_func=ASSISTANTS.get)
    st.caption(
        "Internal searches internal + public docs. External can only open the public collection."
    )
    if choice == "internal":
        st.info("Signed in as: demo employee (real login and roles come in Phase 4)")
    if st.button("Clear this chat"):
        st.session_state.setdefault("chats", {})[choice] = []

    s = get_settings()
    model = s.gemini_model if s.llm_provider == "gemini" else s.ollama_model
    st.caption(f"LLM: {s.llm_provider} · {model}")

chats = st.session_state.setdefault("chats", {name: [] for name in ASSISTANTS})
history = chats.setdefault(choice, [])

try:
    assistant = get_assistant(choice)
except (RuntimeError, LLMError) as exc:
    st.error(str(exc))
    st.stop()

st.title(assistant.profile.display_name)
st.caption(assistant.profile.description)

# ---------- chat history ----------
for msg in history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            render_sources(msg["sources"], msg["cited"])
            st.caption(f"{msg['latency_ms'] / 1000:.1f}s")

if not history:
    st.write("Try asking:")
    for example in EXAMPLES[choice]:
        st.markdown(f"- {example}")

# ---------- new question ----------
if question := st.chat_input("Ask a question in English or Bahasa Indonesia"):
    with st.chat_message("user"):
        st.markdown(question)

    past = [Message(m["role"], m["content"]) for m in history]
    with st.chat_message("assistant"):
        with st.spinner("Searching documents..."):
            try:
                answer = assistant.ask(question, past)
            except LLMError as exc:
                st.error(str(exc))
                st.stop()
        st.markdown(answer.text)
        msg = to_message_dict(answer)
        render_sources(msg["sources"], msg["cited"])
        st.caption(f"{answer.latency_ms / 1000:.1f}s")

    history.append({"role": "user", "content": question})
    history.append(msg)
