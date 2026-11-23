# Brightnest assistants: API + web pages (chat widget, internal chat page).
#
#   docker compose up --build            then open http://localhost:8000
#
# CPU by default (works everywhere, ~2-4 s slower per answer). For an NVIDIA GPU build with
#   --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cu126
# and use docker-compose.gpu.yml.

FROM python:3.12-slim

ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src:/app \
    HF_HOME=/cache/huggingface

WORKDIR /app

# Dependencies first, so editing the code does not reinstall PyTorch.
COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url ${TORCH_INDEX} \
 && pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY api ./api
COPY app ./app
COPY web ./web
COPY profiles ./profiles
COPY data ./data
COPY docker/start.sh ./start.sh

# Run as a normal user, not root.
RUN useradd --create-home --uid 1000 brightnest \
 && mkdir -p /app/storage /cache/huggingface \
 && chown -R brightnest /app/storage /cache \
 && sed -i 's/\r$//' start.sh && chmod +x start.sh
USER brightnest

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=300s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["./start.sh"]
