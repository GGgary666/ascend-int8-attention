#!/usr/bin/env bash
set -euo pipefail
if [[ $# -eq 0 ]]; then
    echo "Usage: $0 MODEL [existing vllm serve arguments...]" >&2
    exit 2
fi
project_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
source "$project_dir/configs/int8_stage1.env"
exec vllm serve "$@"
