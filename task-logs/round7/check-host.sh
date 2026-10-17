#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
unset GGML_MOE_HYBRID GGML_MOE_HYBRID_MIB GGML_MOE_HYBRID_PROFILE GGML_MOE_HYBRID_ADAPTIVE GGML_MOE_HYBRID_ADMIT
unset GGML_MOE_HYBRID_PREFILL GGML_MOE_HYBRID_ADMIT_PREFILL GGML_MOE_HYBRID_STATIC_PCT GGML_MOE_HYBRID_DECAY GGML_MOE_HYBRID_TRACE
unset GGML_MOE_HYBRID_DRAFT_MIB GGML_MOE_HYBRID_DRAFT_PROFILE GGML_MOE_HYBRID_DRAFT_STATIC_PCT
logs=task-logs/round7
trace=${1:-/opt/llm-gateway/data/build/llama-codex2/task-logs/round4-hitmiss/qwen-routes-r1.routes.jsonl}
mkdir -p "$logs/models"
# Use the existing CPU-only and architecture-75 CUDA build trees. No CUDA executable is run.
nice -n 10 cmake --build build-cpu -j 12 --target test-moe-cache test-backend-ops test-llama-archs > "$logs/build-cpu.log" 2>&1
build-cpu/bin/test-llama-archs -a qwen3moe -s 6 -o "$logs/models" > "$logs/model-gen.log" 2>&1
cat > "$logs/warm.profile" <<'PROFILE'
llama-moe-profile-v1
0 0 1
0 1 1
PROFILE
build-cpu/bin/test-moe-cache --ordering "$logs/models/qwen3moe-moe.gguf" "$logs/warm.profile" "$logs/host-trace.jsonl" > "$logs/ordering.log" 2>&1
build-cpu/bin/test-moe-cache --contexts "$logs/models/qwen3moe-moe.gguf" "$logs/context-trace.jsonl" > "$logs/contexts.log" 2>&1
ctest --test-dir build-cpu -R '^test-moe-cache$' --output-on-failure > "$logs/ctest.log" 2>&1
for kind in host context; do
    build-cpu/bin/test-moe-cache --replay-log "$logs/$kind-trace.jsonl" 2 1 2 0 64 > "$logs/replay-$kind-trace.log"
done
python3 "$logs/make_replay.py" "$trace"
bash "$logs/replay.sh"
bash "$logs/replay-legacy.sh" "$trace"
nice -n 10 build-cpu/bin/test-backend-ops test -b CPU -o MUL_MAT_ID_MASKED > "$logs/masked-cpu.out" 2> "$logs/masked-cpu.err"
nice -n 10 build-cpu/bin/test-backend-ops test -b CPU -o MUL_MAT_ID -p 'type_a=(iq3_s|iq3_xxs|iq2_s|iq4_nl|iq4_xs|q2_0|q4_K|q5_K|q6_K|iq2_xxs),.*n=(1|4|5|8),' > "$logs/unmasked-cpu.out" 2> "$logs/unmasked-cpu.err"
nice -n 10 cmake --build build-cuda -j 12 > "$logs/build-cuda.log" 2>&1
git diff --check
