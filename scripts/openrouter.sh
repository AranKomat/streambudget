#!/usr/bin/env bash
# Load a trusted local credential file without copying secrets into the checkout.
set +x
set -euo pipefail

project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
credential_file="${STREAMBUDGET_ENV_FILE:-$project_dir/../.env}"
python_bin="${STREAMBUDGET_PYTHON:-$project_dir/.venv/bin/python}"

if [[ -f "$credential_file" ]]; then
    source "$credential_file"
fi
if [[ -z "${OPENROUTER_API_KEY:-}" ]]; then
    echo "Set OPENROUTER_API_KEY or provide a trusted STREAMBUDGET_ENV_FILE." >&2
    exit 1
fi
export OPENROUTER_API_KEY

if [[ $# -eq 0 ]]; then
    set -- doctor --config "$project_dir/configs/openrouter.yaml"
fi
exec "$python_bin" -m streambudget.cli "$@"
