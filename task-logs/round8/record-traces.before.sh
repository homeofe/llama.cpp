#!/usr/bin/env bash
# Route traces for round 8: the 8-prompt benchmark on Qwen3.8 and Ling (one round each, server with MTP and the cache),
# and a long session (Qwen3.8 perplexity over 6 chunks of test-de in 4-token batches).
set -u
KEY=$(cat /opt/llm-gateway/data/api-key)
T=/opt/llm-gateway/data/moe-quant/traces
R=/opt/llm-gateway/data/build/llama-l5/build-cuda/bin
gw() { curl -s -X "$1" -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' ${3:+-d "$3"} "http://127.0.0.1:8790/api/v1/gpu/$2"; }
cd /opt/llm-gateway/data/moe-quant/qwen38
python3 -u stage_b.py /opt/llm-gateway/data/moe-quant/qwen38/run-traces 23:59 --rounds 1 --only q-trace ling-trace > run-traces.log 2>&1
echo "bench traces exit $?" >> $T/record.txt
gw POST free '{}' > /dev/null; gw POST hold '{"reason":"claude: long-session trace","ttlS":900}' > /dev/null
env LD_LIBRARY_PATH=$R OMP_PLACES=cores OMP_PROC_BIND=close GGML_SCHED_OVERLAP=1 LLAMA_MOE_SHEXP_EARLY=1 GGML_MOE_HYBRID=1 GGML_MOE_HYBRID_ADAPTIVE=1 \
  GGML_MOE_HYBRID_ADMIT=32 GGML_MOE_HYBRID_MIB=2560 GGML_MOE_HYBRID_PREFILL=1 GGML_MOE_HYBRID_DECAY=128 GGML_MOE_HYBRID_TRACE=$T/qwen38-long-de.jsonl \
  $R/llama-perplexity -m /opt/llm-gateway/models/Qwen3.8-Flash-Next/IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf \
  -f /opt/llm-gateway/data/build/texts/test-de.txt -c 512 -b 512 -ub 4 --chunks 6 -ngl 99 -ncmoe 48 -fa on -t 24 < /dev/null > $T/long-de.log 2>&1
echo "long trace exit $?" >> $T/record.txt
gw DELETE hold > /dev/null
ls -la $T >> $T/record.txt
