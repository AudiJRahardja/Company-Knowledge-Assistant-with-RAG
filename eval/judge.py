"""LLM-as-judge: a second LLM call grades each answer.

Some things can't be checked with string matching: is every claim backed by the sources (faithfulness),
does it actually answer the question (relevance), is it correct compared with the expected answer, and did
it decline to answer. The judge sees the question, the expected answer, the retrieved sources and the
answer, and replies with a small JSON object.

Grades: yes = 1.0, partial = 0.5, no = 0.0.
Note: by default the judge is the same model that wrote the answer, which can be a bit lenient. Treat the
judge scores as a trend between runs, and the exact metrics (retrieval, facts, leaks) as the hard numbers.
"""

import json
import re
from dataclasses import dataclass

from rag.llm import LLMClient, Message
from rag.store import Hit

GRADES = {"yes": 1.0, "partial": 0.5, "no": 0.0}

JUDGE_SYSTEM = """\
You grade answers from a company help chatbot. Be strict and consistent. Reply with JSON only, no other text.
"""

JUDGE_TEMPLATE = """\
EARLIER CONVERSATION (the question may refer back to it):
{history}

QUESTION:
{question}

EXPECTED ANSWER (written by a human, the reference):
{expected}

SOURCES THE CHATBOT WAS GIVEN:
{sources}

CHATBOT ANSWER:
{answer}

Grade the chatbot answer:
- "correct": does it give the facts the question asks for, matching the expected answer? Read the question in
  the context of the earlier conversation: answering a different topic than the one meant is "no".
  "yes" = the asked-for facts are right (extra background in the expected answer, such as exceptions or
  related rules, is optional). "partial" = some asked-for facts missing or slightly off. "no" = wrong, off
  topic, or missing the main point. If the expected answer says the chatbot should not know or should
  decline, "yes" means it declined or said it doesn't know without inventing facts.
- "faithful": is every factual claim supported by the sources above? "yes", "partial", or "no". General
  politeness and contact/fallback details do not count as claims. A refusal with no claims is "yes".
- "relevant": does it address what was asked? "yes", "partial", or "no".
- "refused": true if it declines or says it doesn't know instead of answering, otherwise false.
- "reason": one short sentence.

JSON format: {{"correct": "...", "faithful": "...", "relevant": "...", "refused": true/false, "reason": "..."}}
"""


@dataclass
class Verdict:
    correct: float | None
    faithful: float | None
    relevant: float | None
    refused: bool | None
    reason: str


def format_sources_for_judge(hits: list[Hit]) -> str:
    if not hits:
        return "(none)"
    return "\n\n".join(f"[{i}] {h.title}\n{h.text}" for i, h in enumerate(hits, start=1))


def _grade(value) -> float | None:
    return GRADES.get(str(value).strip().lower())


def parse_verdict(text: str) -> Verdict:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return Verdict(None, None, None, None, f"judge reply was not JSON: {text[:120]!r}")
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return Verdict(None, None, None, None, f"judge reply was not valid JSON: {text[:120]!r}")
    refused = data.get("refused")
    if isinstance(refused, str):
        refused = refused.strip().lower() in ("true", "yes")
    return Verdict(
        correct=_grade(data.get("correct")),
        faithful=_grade(data.get("faithful")),
        relevant=_grade(data.get("relevant")),
        refused=refused if isinstance(refused, bool) else None,
        reason=str(data.get("reason", "")),
    )


def format_history(history: list[Message] | tuple[Message, ...]) -> str:
    if not history:
        return "(none)"
    return "\n".join(f"{'User' if m.role == 'user' else 'Chatbot'}: {m.content}" for m in history)


def judge(llm: LLMClient, question: str, expected: str, hits: list[Hit], answer: str,
          history: list[Message] | tuple[Message, ...] = ()) -> Verdict:
    prompt = JUDGE_TEMPLATE.format(
        history=format_history(history), question=question, expected=expected or "(none given)",
        sources=format_sources_for_judge(hits), answer=answer,
    )
    return parse_verdict(llm.generate(JUDGE_SYSTEM, [Message("user", prompt)]))
