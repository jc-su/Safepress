#!/usr/bin/env bash
set -euo pipefail

# Example: serve a quantized HF model with vLLM (OpenAI-compatible API).
# Requires: pip install vllm
#
# Usage:
#   ./scripts/serve_vllm.sh /path/to/model_quantized 8000
#
# Then point AgentDojo/OpenAI client at:
#   OPENAI_BASE_URL=http://localhost:8000/v1

MODEL_PATH="${1:-}"
PORT="${2:-8000}"

if [[ -z "${MODEL_PATH}" ]]; then
  echo "Usage: $0 <MODEL_PATH> [PORT]"
  exit 1
fi

python -m vllm.entrypoints.openai.api_server \
  --model "${MODEL_PATH}" \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --dtype auto \
  --max-model-len 4096
