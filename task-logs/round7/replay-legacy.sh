#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
trace=${1:-/opt/llm-gateway/data/build/llama-codex2/task-logs/round4-hitmiss/qwen-routes-r1.routes.jsonl}
for slots in 1805 3000; do
    for admit in 8 100000; do
        build-cpu/bin/test-moe-cache "$trace" "$slots" "$admit" 1 > "task-logs/round7/legacy-$slots-$admit-1.log"
    done
done
for admit in 8 32 100000; do
    build-cpu/bin/test-moe-cache "$trace" 1805 "$admit" 4 > "task-logs/round7/legacy-1805-$admit-4.log"
done
