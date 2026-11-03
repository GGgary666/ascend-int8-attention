#!/usr/bin/env bash
set -euo pipefail
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
project_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$project_dir"
python3 tools/check_publication.py
python3 -m pytest tests/unit "$@"
mkdir -p build/cpu
"${CXX:-c++}" -std=c++17 -O2 -Itests/cpp/stubs -Icsrc/kernels/c8 \
    tests/cpp/test_schedule.cpp -o build/cpu/test_schedule
build/cpu/test_schedule
