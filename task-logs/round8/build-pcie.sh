#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
g++ -O3 -std=c++17 -Wall -Wextra -Iggml/include -Iggml/src -Ivendor \
    task-logs/round8/pcie-misses.cpp -Lbuild-cuda/bin -lggml-cuda -lggml-cpu -lggml-base -lggml -lcudart \
    -Wl,-rpath,"$PWD/build-cuda/bin" -o task-logs/round8/pcie-misses
