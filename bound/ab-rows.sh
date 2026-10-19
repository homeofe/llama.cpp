#!/usr/bin/env bash
set -u
KEY=$(cat /opt/llm-gateway/data/api-key)
H="Authorization: Bearer $KEY"
curl -s -X POST http://127.0.0.1:8790/api/v1/gpu/free -H "$H" > /dev/null
curl -s -X POST http://127.0.0.1:8790/api/v1/gpu/hold -H "$H" -H 'Content-Type: application/json' -d '{"reason":"vibe: mmvq rows A/B","ttlS":900}' > /dev/null
trap 'curl -s -X DELETE http://127.0.0.1:8790/api/v1/gpu/hold -H "$H" > /dev/null' EXIT
cd /opt/llm-gateway/data/build/llama-vibe
while [ "$(cut -d. -f1 /proc/loadavg)" -ge 3 ]; do sleep 5; done
for round in 1 2 3; do
  echo "== round $round"
  for s in "Q6_K 10240 4 2560" "Q6_K 2560 4 6144" "Q4_K 6144 4 2560" "Q6_K 10240 5 2560" "Q6_K 10240 1 2560"; do
    echo -n "  base(2w,2r): "; ./bound/mmv_bench $s 50 2>/dev/null
    echo -n "  rows1(2w,1r): "; LD_LIBRARY_PATH=/opt/llm-gateway/data/build/llama-vibe/bound/libs-tuned ./bound/mmv_bench $s 50 2>/dev/null
  done
done
