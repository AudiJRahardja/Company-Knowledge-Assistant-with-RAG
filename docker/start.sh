#!/bin/sh
# Container start: build what is missing, then start the API.
#   - the search index (only changed documents are re-embedded; the first start downloads the models)
#   - the mock HR / orders database for the tools
set -e

if [ -n "$INGEST_COLLECTIONS" ]; then
  python -m rag.ingest --collections "$INGEST_COLLECTIONS"
else
  python -m rag.ingest
fi

if [ ! -f storage/mock/brightnest.sqlite ]; then
  python -m rag.mockdb
fi

exec uvicorn api.main:app --host 0.0.0.0 --port 8000 ${UVICORN_ARGS}
