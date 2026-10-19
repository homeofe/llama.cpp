#!/usr/bin/env bash
set -u
KEY=$(cat /opt/llm-gateway/data/api-key)
H="Authorization: Bearer $KEY"
curl -s -X POST http://127.0.0.1:8790/api/v1/gpu/free -H "$H" > /dev/null
curl -s -X POST http://127.0.0.1:8790/api/v1/gpu/hold -H "$H" -H 'Content-Type: application/json' -d '{"reason":"vibe: dense-path bound series 2","ttlS":900}' > /dev/null
trap 'curl -s -X DELETE http://127.0.0.1:8790/api/v1/gpu/hold -H "$H" > /dev/null' EXIT
shapes=(
"F16 10240 4 320"
"BF16 10240 4 320"
"BF16 10240 4 640"
"BF16 10240 4 1280"
"BF16 10240 8 320"
"BF16 10240 1 320"
"BF16 20480 4 320"
"F16 320 4 10240"
)
for round in 1 2 3; do
  echo "== round $round"
  for s in "${shapes[@]}"; do ./mmv_bench $s 50 2>/dev/null; done
done
