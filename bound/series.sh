#!/usr/bin/env bash
set -u
KEY=$(cat /opt/llm-gateway/data/api-key)
H="Authorization: Bearer $KEY"
curl -s -X POST http://127.0.0.1:8790/api/v1/gpu/free -H "$H" > /dev/null
curl -s -X POST http://127.0.0.1:8790/api/v1/gpu/hold -H "$H" -H 'Content-Type: application/json' -d '{"reason":"vibe: dense-path bound series","ttlS":900}' > /dev/null
trap 'curl -s -X DELETE http://127.0.0.1:8790/api/v1/gpu/hold -H "$H" > /dev/null' EXIT
shapes=(
"BF16 320 4 10240"
"BF16 10240 4 320"
"BF16 512 4 2560"
"Q6_K 10240 4 2560"
"Q4_K 6144 4 2560"
"Q6_K 2560 4 6144"
"IQ4_NL 2560 4 640"
"Q4_K 640 4 2560"
"IQ4_XS 640 4 2560"
"F16 320 4 10240"
"Q8_0 10240 4 2560"
"BF16 320 5 10240"
"Q6_K 10240 5 2560"
)
for round in 1 2 3; do
  echo "== round $round"
  for s in "${shapes[@]}"; do
    ./mmv_bench $s 50 2>/dev/null
  done
done
