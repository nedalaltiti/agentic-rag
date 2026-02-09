#!/usr/bin/env sh
set -e

OLLAMA_URL="http://ollama:11434"
MAX_RETRIES=30
RETRY_INTERVAL=2

attempt=0
until curl -sf "${OLLAMA_URL}/api/tags" > /dev/null; do
  attempt=$((attempt + 1))
  if [ "$attempt" -ge "$MAX_RETRIES" ]; then
    echo "ERROR: Ollama not reachable after ${MAX_RETRIES} attempts" >&2
    exit 1
  fi
  echo "Waiting for Ollama to become ready (attempt ${attempt}/${MAX_RETRIES})..."
  sleep "$RETRY_INTERVAL"
done

echo "Ollama is ready. Pulling required models..."

for model in "$LLM_MODEL" "$EMBEDDING_MODEL"; do
  if [ -z "$model" ]; then
    continue
  fi
  echo "Pulling: ${model}"
  ollama pull "$model"
done

echo "All models pulled successfully."
