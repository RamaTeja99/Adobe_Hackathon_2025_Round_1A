FROM python:3.10-slim AS builder
WORKDIR /tmp

COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir -r requirements.txt

FROM python:3.10-slim
WORKDIR /app

COPY --from=builder /usr/local/lib/python3.10/site-packages \
                     /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

COPY output_schema.json .
COPY pdf_utils/ pdf_utils/
COPY process_pdfs.py .

RUN mkdir /app/output
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    INPUT_DIR=/app/input OUTPUT_DIR=/app/output

RUN adduser --system --group appuser && chown -R appuser:appuser /app
USER appuser

HEALTHCHECK CMD python -c "import pdf_utils" || exit 1
CMD ["python", "process_pdfs.py"]
