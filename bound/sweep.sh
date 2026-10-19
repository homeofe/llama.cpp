#!/usr/bin/env bash
set -u
KEY=$(cat /opt/llm-gateway/data/api-key)
H="Authorization: Bearer $KEY"
curl -s -X POST http://127.0.0.1:8790/api/v1/gpu/free -H "$H" > /dev/null
curl -s -X POST http://127.0.0.1:8790/api/v1/gpu/hold -H "$H" -H 'Content-Type: application/json' -d '{"reason":"vibe: mmvq sweep","ttlS":1800}' > /dev/null
trap 'curl -s -X DELETE http://127.0.0.1:8790/api/v1/gpu/hold -H "$H" > /dev/null' EXIT
cd /opt/llm-gateway/data/build/llama-vibe
for cfg in "4 2" "2 2" "2 1" "4 1" "8 2" "8 1"; do
  set -- $cfg
  nw234=$1; nw58=$2
  nice -n 10 cmake --build build-cuda -j 24 --target ggml-cuda > /dev/null 2>&1
  # rebuild with the sweep values via compile definitions
  touch ggml/src/ggml-cuda/mmvq.cu
  cmake -B build-cuda -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75 -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CUDA_FLAGS="-DTURING_MVQ_NWARPS_234=$nw234 -DTURING_MVQ_NWARPS_58=$nw58" > /dev/null 2>&1
  nice -n 10 cmake --build build-cuda -j 24 --target ggml-cuda > /dev/null 2>&1
  echo "== nwarps(2-4)=$nw234 nwarps(5-8)=$nw58"
  ./bound/mmv_bench Q6_K 10240 4 2560 40 2>/dev/null
  ./bound/mmv_bench Q6_K 2560 4 6144 40 2>/dev/null
  ./bound/mmv_bench Q4_K 6144 4 2560 40 2>/dev/null
  ./bound/mmv_bench Q6_K 10240 5 2560 40 2>/dev/null
done
# restore clean config flags
cmake -B build-cuda -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75 -DCMAKE_BUILD_TYPE=Release -UCMAKE_CUDA_FLAGS > /dev/null 2>&1
