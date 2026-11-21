"""Turn a follow-up question into a question that makes sense on its own, before searching.

"And how many of those can I carry over?" contains no topic, so searching for it finds nothing useful.
With the conversation as context, the LLM rewrites it as "How many unused annual leave days can I carry
over to next year?", and search works again.

Only the *search* uses the rewritten question. The answering LLM still sees the user's real words and the
conversation. A question without earlier conversation is never rewritten (no extra LLM call).
"""

from rag.llm import LLMClient, Message

REWRITE_SYSTEM = """\
You rewrite the user's latest message into one standalone search question.
- Use the conversation to fill in what the latest message refers to ("it", "those", "there", "and what about...").
- Keep the same language as the latest message (English or Bahasa Indonesia).
- If the latest message is already clear on its own, return it unchanged.
- Never answer the question. Never add facts. Output only the rewritten question, nothing else.
"""

MAX_TURNS = 6  # only the recent conversation matters for what "it" refers to


def build_rewrite_prompt(question: str, history: list[Message]) -> str:
    lines = [f"{'User' if m.role == 'user' else 'Assistant'}: {m.content}" for m in history[-MAX_TURNS:]]
    return "Conversation so far:\n" + "\n".join(lines) + f"\n\nLatest message: {question}\n\nStandalone question:"


def rewrite_question(llm: LLMClient, question: str, history: list[Message]) -> str:
    if not history:
        return question
    text = llm.generate(REWRITE_SYSTEM, [Message("user", build_rewrite_prompt(question, history))])
    rewritten = text.strip().strip('"').strip()
    # Guard against a bad reply (empty, or an essay instead of a question): fall back to the original.
    if not rewritten or len(rewritten) > 4 * len(question) + 200:
        return question
    return rewritten
