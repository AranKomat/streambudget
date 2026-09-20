#!/usr/bin/env bash
set -euo pipefail
: "${MODEL_ID:?Set MODEL_ID to your downloaded/supported VLM checkpoint}"
# Run in a separate GPU environment; pin vLLM/checkpoint versions in your run manifest.
exec vllm serve "$MODEL_ID" --host 127.0.0.1 --port "${PORT:-8000}" \
  --limit-mm-per-prompt '{"image":16}' "$@"
