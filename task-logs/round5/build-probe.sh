#!/usr/bin/env bash
set -eu
cd "$(dirname "$0")/../.."
nice -n 10 c++ -O2 -std=c++17 -Iinclude -Isrc -Icommon -Iggml/include -Ivendor -Ivendor/nlohmann \
    task-logs/round5/moe-probe.cpp -Lbuild-cuda/bin -Wl,-rpath,"$PWD/build-cuda/bin" \
    -lllama-common -lllama -lggml -lggml-base -o task-logs/round5/moe-probe
