#!/usr/bin/env bash
# Verification chain for the bf16-as-f16 patch: standalone check, mutation cuts, test-backend-ops.
set -u
KEY=$(cat /opt/llm-gateway/data/api-key)
H="Authorization: Bearer $KEY"
gw() { curl -s -X "$1" -H "$H" -H 'Content-Type: application/json' ${3:+-d "$3"} "http://127.0.0.1:8790/api/v1/gpu/$2" > /dev/null; }
gw POST free '{}'; gw POST hold '{"reason":"vibe: bf16-as-f16 verification","ttlS":1800}'
trap 'gw DELETE hold' EXIT
cd /opt/llm-gateway/data/build/llama-vibe
echo '== 1. clean check'
./bound/check_bf16_f16; echo "exit $?"
echo '== 2. tbo mul_mat bf16 (switch off = unchanged path)'
./build-cuda/bin/test-backend-ops test -b CUDA0 -o MUL_MAT -p 'bf16' 2>&1 | tail -3
echo '== 3. tbo mul_mat bf16 (switch on)'
GGML_CUDA_BF16_AS_F16=1 ./build-cuda/bin/test-backend-ops test -b CUDA0 -o MUL_MAT -p 'bf16' 2>&1 | tail -3
