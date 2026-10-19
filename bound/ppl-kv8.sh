#!/usr/bin/env bash
# Perplexity: the package (tuned lib + bf16-as-f16) with f16 KV vs int8 KV, ub4 and ub512.
set -u
KEY=$(cat /opt/llm-gateway/data/api-key)
H="Authorization: Bearer $KEY"
R=/opt/llm-gateway/data/build/llama-vibe/build-cuda/bin
T=/opt/llm-gateway/data/build/llama-vibe/bound/libs-tuned
OUT=/opt/llm-gateway/data/moe-quant/qwen38/vibe-kv8-ppl
mkdir -p $OUT
gw() { curl -s -X "$1" -H "$H" -H 'Content-Type: application/json' ${3:+-d "$3"} "http://127.0.0.1:8790/api/v1/gpu/$2" > /dev/null; }
trap 'gw DELETE hold' EXIT
for ub in 4 512; do
  for v in f16 kv8; do
    while [ "$(cut -d' ' -f1 /proc/loadavg | cut -d. -f1)" -ge 3 ]; do sleep 5; done
    gw POST free '{}'; gw POST hold '{"reason":"vibe: kv8 perplexity","ttlS":1800}'
    KV=""
    [ "$v" = kv8 ] && KV="-ctk q8_0 -ctv q8_0"
    env LD_LIBRARY_PATH=$T:$R OMP_PLACES=cores OMP_PROC_BIND=close $R/llama-perplexity \
      -m /opt/llm-gateway/models/Qwen3.8-Flash-Next/IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf \
      -f /opt/llm-gateway/data/build/texts/test-de.txt -c 512 -b 512 -ub $ub --chunks 12 -ngl 99 -ncmoe 48 -fa on -t 24 $KV \
      < /dev/null > $OUT/ppl-$v-ub$ub.log 2>&1
    echo "$v ub$ub exit $?: $(grep -m1 'Final estimate' $OUT/ppl-$v-ub$ub.log)"
    gw DELETE hold
  done
done
