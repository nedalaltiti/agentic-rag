#!/usr/bin/env bash
set -euo pipefail

if [[ -f ".env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

LLM_MODEL="${LLM_MODEL:-qwen3:1.7b}"
EMBEDDING_MODEL="${EMBEDDING_MODEL:-qwen3-embedding:0.6b}"
EVAL_MODEL="${EVAL_MODEL:-qwen3:4b}"

echo "Pulling models:"
echo "  LLM_MODEL=${LLM_MODEL}"
echo "  EMBEDDING_MODEL=${EMBEDDING_MODEL}"
echo "  EVAL_MODEL=${EVAL_MODEL}"

ollama pull "${LLM_MODEL}"
ollama pull "${EMBEDDING_MODEL}"
ollama pull "${EVAL_MODEL}"
