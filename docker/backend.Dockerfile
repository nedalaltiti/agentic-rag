FROM python:3.11-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl \
  && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
COPY src ./src

RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir .

FROM python:3.11-slim AS runtime

WORKDIR /app

COPY --from=builder /usr/local /usr/local

COPY src ./src

ENV PYTHONUNBUFFERED=1
EXPOSE 8000

HEALTHCHECK --interval=10s --timeout=3s --retries=10 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health').read()" || exit 1

CMD ["uvicorn", "agentic_rag.backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
