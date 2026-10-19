#!/usr/bin/env bash
set -u
KEY=$(cat /opt/llm-gateway/data/api-key)
H="Authorization: Bearer $KEY"
curl -s -X POST http://127.0.0.1:8790/api/v1/gpu/free -H "$H" > /dev/null
curl -s -X POST http://127.0.0.1:8790/api/v1/gpu/hold -H "$H" -H 'Content-Type: application/json' -d '{"reason":"vibe: mmq bound","ttlS":900}' > /dev/null
trap 'curl -s -X DELETE http://127.0.0.1:8790/api/v1/gpu/hold -H "$H" > /dev/null' EXIT
shapes=(
"Q6_K 10240 4 2560"
"Q4_K 6144 4 2560"
"Q6_K 2560 4 6144"
"IQ4_NL 2560 4 640"
"IQ4_XS 640 4 2560"
"Q6_K 10240 5 2560"
"Q6_K 10240 1 2560"
"Q6_K 10240 8 2560"
)
for round in 1 2 3; do
  echo "== round $round"
  for s in "${shapes[@]}"; do
    echo -n "  mvq: "; ./mmv_bench $s 50 2>/dev/null
    echo -n "  mmq: "; GGML_CUDA_FORCE_MMQ=1 ./mmv_bench $s 50 2>/dev/null
  done
done
