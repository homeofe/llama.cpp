#!/usr/bin/env bash
set -euo pipefail
unset GGML_MOE_HYBRID GGML_MOE_HYBRID_ADAPTIVE GGML_MOE_HYBRID_PROFILE GGML_MOE_HYBRID_ADMIT
cd "$(dirname "$0")/../.."
trace=${1:-/opt/llm-gateway/data/build/llama-codex2/task-logs/round4-hitmiss/qwen-routes-r1.routes.jsonl}
logs=task-logs/round6
mkdir -p "$logs/models"
cmake -S . -B build-cpu > "$logs/configure-cpu.log" 2>&1
nice -n 10 cmake --build build-cpu -j 16 --target test-moe-cache test-backend-ops test-llama-archs > "$logs/build-cpu.log" 2>&1
# Use a CPU-only binary even on hosts with CUDA installed.
build-cpu/bin/test-llama-archs -a qwen3moe -s 6 -o "$logs/models" > "$logs/model-gen.log" 2>&1
env -u GGML_MOE_HYBRID -u GGML_MOE_HYBRID_ADMIT build-cpu/bin/test-moe-cache --ordering "$logs/models/qwen3moe-moe.gguf" "$logs/warm.profile" > "$logs/ordering.log" 2>&1
env -u GGML_MOE_HYBRID_PROFILE GGML_MOE_HYBRID=1 GGML_MOE_HYBRID_ADAPTIVE=1 GGML_MOE_HYBRID_MIB=1 build-cpu/bin/test-llama-archs -a qwen3 -s 6 -v 2 -o "$logs/models" > "$logs/no-experts-context.log" 2>&1
env GGML_MOE_HYBRID=1 GGML_MOE_HYBRID_ADAPTIVE=1 GGML_MOE_HYBRID_MIB=1 GGML_MOE_HYBRID_PROFILE="$logs/draft-mismatch.profile" build-cpu/bin/test-llama-archs -a qwen3moe -s 6 -v 2 -o "$logs/models" > "$logs/profile-mismatch-context.log" 2>&1
ctest --test-dir build-cpu -R '^test-moe-cache$' --output-on-failure > "$logs/ctest.log" 2>&1
for slots in 1805 3000; do
    for admit in 8 100000; do
        build-cpu/bin/test-moe-cache "$trace" "$slots" "$admit" 1 > "$logs/replay-$slots-admit$admit-ub1.log"
    done
done
for admit in 8 32 100000; do
    build-cpu/bin/test-moe-cache "$trace" 1805 "$admit" 4 > "$logs/replay-1805-admit$admit-ub4.log"
done
nice -n 10 build-cpu/bin/test-backend-ops test -b CPU -o MUL_MAT_ID_MASKED > "$logs/masked-cpu.out" 2> "$logs/masked-cpu.err"
nice -n 10 build-cpu/bin/test-backend-ops test -b CPU -o MUL_MAT_ID -p 'type_a=(iq3_s|iq3_xxs|iq2_s|iq4_nl|iq4_xs|q2_0|q4_K|q5_K|q6_K|iq2_xxs),.*n=(1|4|5|8),' > "$logs/unmasked-cpu.out" 2> "$logs/unmasked-cpu.err"
nice -n 10 cmake --build build-cuda -j 16 > "$logs/build-cuda.log" 2>&1
git diff --check
