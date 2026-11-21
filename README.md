# Brightnest Knowledge Assistants

Two chatbots for one (fictional) Indonesian home & kitchen retailer, **Brightnest Home & Living**, built on a
single retrieval-augmented generation (RAG) engine:

- **Employee Helper (internal)**: HR policies, leave, public holidays, remote work, reimbursements,
  benefits, IT help, onboarding.
- **Customer Help (external)**: products, store hours, contact, refunds and returns, shipping, warranty,
  loyalty program, careers.

Both answer in English or Bahasa Indonesia, using only the company documents, with numbered citations.

> Status: **Phase 3 (search quality) done.** Security hardening, tools and deployment come next.
> See [docs/PLAN.md](docs/PLAN.md).

## How it works

```
                 ┌──────────────── ingest (offline) ────────────────┐
data/docs/*.md → validate → split at headings → embed (BGE-M3) → storage/qdrant/public/
                                                               → storage/qdrant/internal/
                                                               → storage/qdrant/restricted/

question → profile (internal | external): only the profile's databases can be opened
  → follow-up? rewrite it into a standalone question (LLM)
  → two questions in one? also search each part on its own
  → meaning search (vectors) + keyword search (BM25), combined by rank fusion
  → keep one copy of each section (EN or ID, not both)
  → reranker (bge-reranker-v2-m3) picks the best 5 of ~20 candidates
  → numbered sources → LLM (Gemini or Ollama) → answer with [1] [2] citations
```

Each search step can be switched off in `.env` (`RETRIEVAL_HYBRID`, `RETRIEVAL_MERGE_TWINS`,
`RERANKER_MODEL=none`, `QUERY_REWRITE`, `RETRIEVAL_SPLIT_QUESTIONS`), which is how the eval measures what each one adds.

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
python -m rag.ingest                  # build the indexes (first run downloads BGE-M3, ~2.2 GB;
                                      # the reranker, ~2.2 GB more, downloads on first search)
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

Answers by gemini-3.5-flash-lite (gemini-3.1-flash-lite as backup). Every row in `eval/results.csv`.

**Search, step by step** (each change measured on its own, both test sets):

| Step | Internal: all needed docs found | Internal MRR | External: all needed docs found | External MRR |
|---|---|---|---|---|
| Baseline: vector search, fixed 220-word chunks | 97% | 0.93 | 87% | 0.93 |
| A. One chunk per section (split at headings) | 95% | 0.97 | 95% | 1.00 |
| B. Keep one language copy of each section | 97% | 0.97 | 97% | 1.00 |
| C. Add keyword search (BM25), fuse rankings | 95% | 0.95 | 97% | 0.96 |
| D. Add a reranker | 95% | 0.97 | 98% | 0.98 |
| E. Rewrite follow-up questions before searching | 98% | 1.00 | 98% | 0.98 |
| F. Search each part of a two-part question | **100%** | **1.00** | **100%** | **0.98** |

MRR = 1 / position of the first right document, averaged (1.00 = always ranked first).

Notes:
- Keyword search on its own made the order slightly worse (C), but it found extra candidates that the
  reranker then put in the right order (D). With the reranker, turning keyword search off changed the
  results by about one question, so it stays on: real users type exact codes and order numbers that the
  test set barely covers.
- Follow-up questions went from 67% to 100% with rewriting (E).
- Two-part questions ("how much leave do I get, and how do I add the baby to my insurance?") lost one half
  of the answer after reranking, because each section answers only half the question. Combining the
  reranker's and the first search's rankings didn't help; searching each part separately did (F).

**Answers, baseline vs Phase 3:**

| Metric | Internal before | Internal after | External before | External after |
|---|---|---|---|---|
| Key facts in the answer | 97% | **100%** | 100% | 100% |
| Correct (judge) | 97% | **98%** | 100% | 99% |
| Faithful to sources (judge) | 100% | 98% | 99% | 99% |
| Answers with citations | 100% | 100% | 100% | 100% |
| Says "I don't know" when it should | 100% | 100% | 100% | 100% |
| Refuses when it shouldn't | 0% | 0% | 0% | 0% |
| Used an outdated policy | 1% | 3% | 0% | 0% |

Leak tests: **0% leak rate** before and after (30 attacks; 2 order-lookup tests wait for Phase 5). The
assistant refused 100% of attacks after Phase 3, up from 93%.

Known issue for Phase 4: the reranker only judges how well text matches the question, not which policy
version is current, so the old 2024 remote-work policy can outrank the 2026 one. The answers still give
the current rules, but sometimes mention or cite the old one. Version handling is planned for Phase 4.

## Project layout

```
data/docs/{public,internal,restricted}/{en,id}/   knowledge base
profiles/{internal,external}.yaml                 assistant settings (collections, prompt, fallback)
src/rag/                                          the shared engine
app/streamlit_app.py                              demo UI with an assistant switch
eval/                                             test sets, scoring, and results history
tests/                                            unit and security tests
```
