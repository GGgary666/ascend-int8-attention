#!/usr/bin/env bash
set -euo pipefail
project_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
if [[ $# -gt 1 || ($# -eq 1 && "$1" != "--device-only") ]]; then
    echo "Usage: $0 [--device-only]" >&2
    exit 2
fi
cmake_bin=${SHAREDKV_CMAKE:-cmake}
"$cmake_bin" -S "$project_dir" -B "$project_dir/build" -G Ninja -DCMAKE_BUILD_TYPE=Release
"$cmake_bin" --build "$project_dir/build" --target sharedkv_c8 sharedkv_int8 -j "${SHAREDKV_BUILD_JOBS:-2}"
if [[ $# -eq 0 ]]; then
    python3 "$project_dir/tools/build_host.py"
fi
