#!/usr/bin/env bash
set -euo pipefail

# Run AgentDojo using an OpenAI-compatible model endpoint (e.g., vLLM).
#
# Usage:
#   OPENAI_BASE_URL=http://localhost:8000/v1 OPENAI_API_KEY=EMPTY \
#   ./scripts/run_agentdojo.sh <MODEL_ID> <DEFENSE> <ATTACK>
#
# MODEL_ID is the model name exposed by the OpenAI-compatible server.
# Many servers accept any OPENAI_API_KEY; use "EMPTY".

MODEL="${1:-}"
DEFENSE="${2:-none}"
ATTACK="${3:-none}"

if [[ -z "${MODEL}" ]]; then
  echo "Usage: $0 <MODEL_ID> [DEFENSE] [ATTACK]"
  exit 1
fi

python -m agentdojo.scripts.benchmark \
  --model "${MODEL}" \
  --defense "${DEFENSE}" \
  --attack "${ATTACK}"
