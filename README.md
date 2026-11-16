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

## Project layout

```
data/docs/{public,internal,restricted}/{en,id}/   knowledge base
profiles/{internal,external}.yaml                 assistant settings (collections, prompt, fallback)
src/rag/                                          the shared engine
app/streamlit_app.py                              demo UI with an assistant switch
tests/                                            unit and security tests
```
