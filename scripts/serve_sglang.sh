#!/usr/bin/env bash
set -euo pipefail
: "${MODEL_ID:?Set MODEL_ID to your downloaded/supported VLM checkpoint}"
# Confirm flags against the installed SGLang version. No GPU server was launched here.
exec python -m sglang.launch_server --model-path "$MODEL_ID" \
  --host 127.0.0.1 --port "${PORT:-30000}" "$@"
