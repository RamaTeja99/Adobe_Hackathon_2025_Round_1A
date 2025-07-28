FROM python:3.10-slim AS builder
WORKDIR /tmp
ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends gcc g++

COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

FROM python:3.10-slim
WORKDIR /app

COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY pdf_utils/ pdf_utils/
COPY persona_miner/ persona_miner/
COPY analyze_collection.py .
RUN mkdir -p /app/input/outlines /app/output /tmp/hf_cache && \
    chmod 777 /tmp/hf_cache

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TRANSFORMERS_CACHE=/tmp/hf_cache \
    NLTK_DATA=/tmp/hf_cache
RUN adduser --system --group appuser && \
    chown -R appuser:appuser /app /tmp/hf_cache
USER appuser

HEALTHCHECK CMD python -c "import pdf_utils, persona_miner" || exit 1
CMD ["python", "analyze_collection.py"]
