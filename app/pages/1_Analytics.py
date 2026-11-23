"""Analytics per assistant, from the chat log (storage/logs/chats.sqlite).

What to look at:
- Unanswered questions: internal -> a gap in the HR / IT documents; external -> something the website or FAQ
  should explain.
- Hand-off rate: how often a person had to take over (no answer, or a ticket / support hand-off / sales lead).
- Thumbs down: answers to read and fix.

In a real deployment this page would be for the HR / support team only. Questions are stored with emails
and phone numbers blanked out, and without usernames.
"""

import streamlit as st

from rag.chatlog import ChatLog
from rag.config import get_settings

st.set_page_config(page_title="Analytics · Brightnest Assistants", page_icon="📊", layout="wide")
settings = get_settings()

st.title("Assistant analytics")
profile = st.radio("Assistant", ["internal", "external"], horizontal=True,
                   format_func={"internal": "Internal: employees", "external": "External: customers"}.get)
period = st.selectbox("Period", [7, 30, 90, None],
                      format_func=lambda d: f"Last {d} days" if d else "All time", index=1)

if not settings.chat_db_path.exists():
    st.info("No chats logged yet. Ask the assistants a few questions first.")
    st.stop()

report = ChatLog(settings.chat_db_path).analytics(profile, period, top=15)
if not report["questions"]:
    st.info("No questions in this period.")
    st.stop()


def pct(value: float | None) -> str:
    return "–" if value is None else f"{value:.0%}"


c = st.columns(5)
c[0].metric("Questions", report["questions"])
c[1].metric("Answered", pct(report["answered_rate"]))
c[2].metric("Unanswered", pct(report["unanswered_rate"]),
            help="No answer in the documents: the user got the contact details instead.")
c[3].metric("Handed to a person", pct(report["handoff_rate"]),
            help="Unanswered, or a ticket / support hand-off / sales lead was created.")
fb = report["feedback"]
c[4].metric("Thumbs up", pct(fb["satisfaction"]), help=f"{fb['up']} up, {fb['down']} down "
            f"({pct(fb['rated_share'])} of answers were rated).")

left, right = st.columns(2)
with left:
    st.subheader("Top questions")
    st.dataframe([{"question": g["question"], "times": g["count"], "lang": g["language"]}
                  for g in report["top_questions"]], hide_index=True, width="stretch")
    st.subheader("Unanswered questions")
    st.caption("Doc gaps (internal) or FAQ / website improvements (external).")
    if report["unanswered"]:
        st.dataframe([{"question": g["question"], "times": g["count"], "lang": g["language"]}
                      for g in report["unanswered"]], hide_index=True, width="stretch")
    else:
        st.write("None.")
with right:
    st.subheader("Most cited documents")
    st.dataframe([{"document > section": d, "cited": n} for d, n in report["top_docs"]],
                 hide_index=True, width="stretch")
    st.subheader("How questions ended")
    st.bar_chart(report["status"], horizontal=True)
    if report["tools"]:
        st.subheader("Tools used")
        st.bar_chart(report["tools"], horizontal=True)

st.subheader("Answers with a thumbs down")
if report["disliked"]:
    for d in report["disliked"]:
        with st.expander(f"{d['time'][:16].replace('T', ' ')} · {d['question']}"):
            st.write(d["answer"])
            if d["comment"]:
                st.caption(f"Comment: {d['comment']}")
else:
    st.write("None.")

left, right = st.columns(2)
left.caption("Languages: " + ", ".join(f"{k} {v}" for k, v in report["languages"].items()))
if report["roles"]:
    right.caption("Roles: " + ", ".join(f"{k} {v}" for k, v in report["roles"].items()))
st.caption(f"Average answer time: {report['avg_latency_ms'] / 1000:.1f}s · blocked: {pct(report['blocked_rate'])}")
