# ---- Builder ----
FROM python:3.11-slim AS builder

WORKDIR /app

# System deps (keep minimal)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl \
  && rm -rf /var/lib/apt/lists/*

# Copy project metadata first for better caching
COPY pyproject.toml ./
COPY src ./src

# Install the package (includes runtime deps)
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir .

# ---- Runtime ----
FROM python:3.11-slim AS runtime

WORKDIR /app

# Copy installed site-packages and entrypoints from builder
COPY --from=builder /usr/local /usr/local

# Copy source (for templates/config files inside src/)
COPY src ./src

ENV PYTHONUNBUFFERED=1
EXPOSE 8000

# Healthcheck hits the backend health endpoint
HEALTHCHECK --interval=10s --timeout=3s --retries=10 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health').read()" || exit 1

CMD ["uvicorn", "agentic_rag.backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
