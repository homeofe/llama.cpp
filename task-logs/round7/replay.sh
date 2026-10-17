#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
logs=task-logs/round7
for workload in long short; do
    for decay in 64 128 512; do
        for policy in lfu prompt static combined; do
            prefill=0
            static=0
            case "$policy" in
                prompt) prefill=1024 ;;
                static) static=25 ;;
                combined) prefill=1024; static=25 ;;
            esac
            build-cpu/bin/test-moe-cache --replay-log "$logs/$workload.jsonl" 1501 32 "$prefill" "$static" "$decay" "$logs/base.profile" > "$logs/replay-$workload-$policy-$decay.log"
        done
    done
done
