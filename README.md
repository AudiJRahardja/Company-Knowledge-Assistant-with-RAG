# Brightnest Knowledge Assistants

Two chatbots for one (fictional) Indonesian home & kitchen retailer, **Brightnest Home & Living**, built on a
single retrieval-augmented generation (RAG) engine:

- **Employee Helper (internal)**: HR policies, leave, public holidays, remote work, reimbursements,
  benefits, IT help, onboarding.
- **Customer Help (external)**: products, store hours, contact, refunds and returns, shipping, warranty,
  loyalty program, careers.

Both answer in English or Bahasa Indonesia, using only the company documents, with numbered citations.

> Status: **Phase 1 (MVP)**. Evaluation, hybrid search, security hardening, tools and deployment are
> coming in later phases. See [docs/PLAN.md](docs/PLAN.md).

## How it works

```
                 ┌──────────── ingest (offline) ────────────┐
data/docs/*.md → validate → chunk → embed (BGE-M3) → storage/qdrant/public/
                                                    → storage/qdrant/internal/
                                                    → storage/qdrant/restricted/

question → profile (internal | external) → search ONLY the profile's databases
         → top 5 chunks as numbered sources → LLM (Gemini or Ollama) → answer with [1] [2] citations
```

**Key security idea:** public and internal documents live in physically separate databases. The external
assistant can only open the public one. This is enforced in the storage layer (`src/rag/store.py`) and the
profile loader, not just by the prompt. Tests in `tests/test_store.py` check it.

## Knowledge base

56 Markdown files: 28 topics, each in English (`en/`) and Indonesian (`id/`).

| Visibility | Topics | Examples |
|---|---|---|
| public | 12 | about us, store hours, refunds, shipping, warranty, FAQ |
| internal | 13 + 1 old version | leave, holidays, remote work (2024 and 2026), reimbursements, IT |
| restricted | 2 | salary bands (HR only), manager guidelines (managers + HR) |

Built-in tricky cases: an outdated and a current policy version, and look-alike topics (customer refunds vs
employee reimbursements, store hours vs employee working hours, customer service holidays vs employee
holidays).

## Setup (Windows, Python 3.12)

```bash
python -m venv .venv
.venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cu126   # GPU build; omit the URL for CPU
pip install -r requirements.txt
pip install -e .
copy .env.example .env      # then add your GEMINI_API_KEY
```

```bash
python -m rag.ingest                  # build the indexes (first run downloads BGE-M3, ~2.2 GB)
python -m rag.llm                     # check the LLM connection
streamlit run app/streamlit_app.py    # open the chat UI
pytest                                # run the tests
```

Stop the Streamlit app before re-running `rag.ingest`: Qdrant's local mode locks its database folders.

### Using a local model instead of Gemini
Install [Ollama](https://ollama.com), run `ollama pull qwen2.5:7b`, and set `LLM_PROVIDER=ollama` in `.env`.
On a 6 GB GPU, also set `EMBEDDING_DEVICE=cpu` so the embedding model and the LLM don't compete for GPU
memory.

## Evaluation

Every change is measured, not eyeballed. There are three test sets, and every question in them is written
in both English and Indonesian:

| File | Questions | What it checks |
|---|---|---|
| `eval/internal.jsonl` | 74 | employee questions: lookups, multi-document, follow-ups, outdated policy, confusable topics, unanswerable |
| `eval/external.jsonl` | 72 | customer questions, same categories, plus off-topic requests |
| `eval/leak_tests.jsonl` | 32 | attempts to get internal or HR-only information, prompt injection, role escalation |

Each question lists the document(s) that should be found and the key facts the answer must contain.

```bash
python -m eval.run --profile internal --retrieval-only   # search quality only, no LLM calls (seconds)
python -m eval.run --profile internal --label baseline   # full run: answers + LLM judge
python -m eval.run --profile leak                        # security tests
```

What gets measured:
- **Search:** hit@5, meaning the right document is in the top 5 chunks. Also MRR (how high it ranks) and
  how many different documents fill those 5 slots.
- **Answers:**
  - whether the key facts are present (string check);
  - whether the answer is correct, faithful to its sources and relevant (graded by an LLM judge);
  - whether it cites its sources;
  - whether it refuses when it should, and only then;
  - whether it uses an outdated policy.
- **Leaks:** a test counts as leaked if a forbidden document reaches the LLM, or a forbidden fact
  (salary figures, internal names, extensions) appears in the answer. The target is 0%.

Search and leak checks are plain code, so those numbers are exact. Judge scores come from a second LLM call
and are best read as a trend between runs. Every run is saved under `eval/runs/` (not committed) and adds
one line to `eval/results.csv` (committed), which is the before/after history. Gemini replies are cached,
so an interrupted run picks up where it stopped.

### Results history

Baseline (Phase 1 engine: plain vector search, fixed 220-word chunks, top 5). Answers by
gemini-3.5-flash-lite, with gemini-3.1-flash-lite as backup.

| Metric | Internal | External |
|---|---|---|
| Right document in top 5 (hit@5) | 97% | 100% |
| All expected documents in top 5 | 97% | 87% |
| MRR | 0.93 | 0.93 |
| Different documents in top 5 | 2.6 | 2.5 |
| Key facts in the answer | 97% | 100% |
| Correct (judge) | 97% | 100% |
| Faithful to sources (judge) | 100% | 99% |
| Answers with citations | 100% | 100% |
| Says "I don't know" when it should | 100% | 100% |
| Refuses when it shouldn't | 0% | 0% |
| Used an outdated policy | 1% | 0% |

Leak tests: **0% leak rate** (30 attacks; 2 order-lookup tests wait for Phase 5).

Weak spots found:
- Follow-up questions find the right document only 67% of the time (internal). Search only sees the
  latest message, so "And how many of those can I carry over?" has no topic.
- Multi-document questions find all the documents they need only 33% of the time (external).
- Only about 2.5 of the 5 search results are different documents, because English and Indonesian copies
  of the same text both take a slot.

## Project layout

```
data/docs/{public,internal,restricted}/{en,id}/   knowledge base
profiles/{internal,external}.yaml                 assistant settings (collections, prompt, fallback)
src/rag/                                          the shared engine
app/streamlit_app.py                              demo UI with an assistant switch
eval/                                             test sets, scoring, and results history
tests/                                            unit and security tests
```
