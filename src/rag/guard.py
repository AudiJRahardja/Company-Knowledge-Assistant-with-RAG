"""Safety checks around the assistant: before the question goes in, and before the answer goes out.

Input
  - length limit (long inputs cost money and are a common way to smuggle in instructions)
  - prompt-injection screening: messages that try to override the assistant's rules ("ignore your
    instructions", "show your system prompt", "you are now DevBot", "SYSTEM NOTICE: ...") get a polite
    refusal and never reach the LLM. Earlier user messages are screened too, because an attacker can plant
    the trick in the conversation history.

Output
  - the answer must not repeat the system prompt (checked with 8-word overlaps)
  - the answer must not contain "fingerprints" of documents the user may not read: e-mail addresses,
    phone numbers and extensions, money amounts, and person/system names that only appear in internal or
    restricted documents. The fingerprints are made at ingest time and stored only as hashes (see
    fingerprint.py), so the customer assistant can recognise an internal e-mail address without having a
    list of them.

Also: a simple language detector (refusals and hand-offs in the user's language) and a rate limiter.

These are the second line of defence. The first is that the data simply isn't reachable (store.py).
"""

import re
import time
from collections import defaultdict, deque

# ---------------------------------------------------------------- language

_ID_WORDS = {
    "apa", "apakah", "berapa", "bagaimana", "kapan", "siapa", "mengapa", "kenapa", "bisakah", "bolehkah",
    "saya", "aku", "kamu", "anda", "kami", "kita", "yang", "dan", "atau", "di", "ke", "dari", "untuk",
    "dengan", "tidak", "bisa", "boleh", "ada", "ini", "itu", "sudah", "belum", "akan", "harus", "jam",
    "hari", "toko", "cuti", "karyawan", "barang", "pesanan", "berapa", "tolong", "mohon", "dong", "ya",
    "kalau", "jika", "sebelumnya", "lama", "baru", "saja", "juga", "lagi", "pada", "dalam", "terima",
    "kasih", "gaji", "kantor", "libur", "uang", "tahun", "bulan", "minggu",
}
_EN_WORDS = {
    "what", "how", "when", "where", "who", "why", "which", "can", "could", "do", "does", "is", "are", "the",
    "a", "an", "and", "or", "of", "to", "for", "with", "my", "i", "you", "your", "we", "our", "it", "in",
    "on", "at", "be", "have", "has", "will", "should", "much", "many", "there", "this", "that", "from",
    "store", "leave", "days", "please", "about", "get", "if", "not", "any",
}


def detect_language(text: str) -> str:
    """'id' or 'en', from a count of very common words. Good enough for routing replies."""
    words = re.findall(r"[a-z]+", text.lower())
    id_score = sum(w in _ID_WORDS for w in words)
    en_score = sum(w in _EN_WORDS for w in words)
    return "id" if id_score > en_score else "en"


# ---------------------------------------------------------------- input checks

class InputRejected(ValueError):
    """The question is malformed (too long, empty). The API turns this into an HTTP 4xx error."""


_INJECTION = [re.compile(p, re.IGNORECASE) for p in (
    # English
    r"\b(ignore|disregard|forget|override|bypass)\b.{0,30}\b(instructions?|rules|prompts?|guidelines|restrictions)\b",
    r"\b(system|hidden|initial|original)\s+(prompt|instructions?|message)\b",
    r"\b(reveal|print|show|repeat|output|display)\b.{0,20}\b(your|the)\s+(prompt|instructions|rules)\b",
    r"\byou\s+are\s+now\b|\bfrom\s+now\s+on,?\s+you\s+are\b|\bpretend\s+(to\s+be|you\s+are)\b",
    r"\b(developer|admin|debug|god|jailbreak|unrestricted)\s+mode\b|(?-i:\bDAN\b)",  # not Indonesian "dan"
    r"\bno\s+restrictions\b|\bwithout\s+(any\s+)?restrictions\b",
    r"^\W*(system|admin|developer)\s*(notice|message|override|instruction)?\s*:",
    # Bahasa Indonesia
    r"\b(abaikan|lupakan|langgar|lewati)\b.{0,30}\b(instruksi|perintah|aturan|arahan)",
    r"\b(tampilkan|tunjukkan|bocorkan|ulangi)\b.{0,30}\b(prompt|instruksi|aturan)",
    r"\bmulai\s+sekarang\s+kamu\b|\bkamu\s+sekarang\s+adalah\b|\bberpura-?pura\b",
    r"\bmode\s+(admin|debug|developer|pengembang)\b|\btanpa\s+batasan\b",
    r"^\W*pemberitahuan\s+sistem\b",
)]


def looks_like_injection(text: str) -> bool:
    return any(p.search(text) for p in _INJECTION)


def check_question(question: str, max_chars: int) -> str:
    """Return the cleaned question, or raise InputRejected."""
    question = (question or "").strip()
    if not question:
        raise InputRejected("The question is empty.")
    if len(question) > max_chars:
        raise InputRejected(f"The question is too long ({len(question)} characters, the limit is {max_chars}).")
    return question


BLOCKED_REPLY = {
    "en": "Sorry, I can't help with that request. I can only answer questions using the company's documents.",
    "id": "Maaf, saya tidak dapat membantu permintaan tersebut. Saya hanya dapat menjawab pertanyaan berdasarkan "
          "dokumen perusahaan.",
}


# ---------------------------------------------------------------- output checks

def _words(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def shingles(text: str, n: int = 8) -> set[str]:
    """All runs of n consecutive words."""
    w = _words(text)
    return {" ".join(w[i:i + n]) for i in range(len(w) - n + 1)}


PROMPT_SHARE_LIMIT = 0.2


def repeats_prompt(answer: str, prompt_shingles: set[str], limit: float = PROMPT_SHARE_LIMIT) -> bool:
    """Does the answer reproduce a large part (default 20%) of the instructions?

    Not "any overlap": a polite refusal naturally reuses a sentence of the instructions ("I can only help
    with questions about Brightnest products and services, stores ..."), which is fine to say. In the eval
    such refusals overlapped about 9%; dumping the instructions overlaps close to 100%.
    """
    if not prompt_shingles:
        return False
    return len(shingles(answer) & prompt_shingles) >= limit * len(prompt_shingles)


# ---------------------------------------------------------------- rate limiting

class RateLimiter:
    """At most `limit` requests per `window_s` seconds per key (a user id or an IP address).

    Sliding window: we remember the time of each recent request and count those inside the window.
    In-memory, so it resets on restart and isn't shared between server processes. That's fine for one
    server; a bigger deployment would use Redis for this.
    """

    def __init__(self, limit: int, window_s: float = 60.0, clock=time.monotonic):
        self.limit = limit
        self.window_s = window_s
        self.clock = clock
        self._hits: dict[str, deque] = defaultdict(deque)

    def allow(self, key: str) -> bool:
        now = self.clock()
        hits = self._hits[key]
        while hits and now - hits[0] >= self.window_s:
            hits.popleft()
        if len(hits) >= self.limit:
            return False
        hits.append(now)
        return True

    def retry_after(self, key: str) -> int:
        """Seconds until the next request will be allowed."""
        hits = self._hits.get(key)
        if not hits or len(hits) < self.limit:
            return 0
        return max(1, int(self.window_s - (self.clock() - hits[0])) + 1)
