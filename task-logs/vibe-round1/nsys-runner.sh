#!/usr/bin/env bash
# Stage 1 profiling: Strata and ours on the same code prompt, one after the other.
set -u
cd /opt/llm-gateway/data/build/llama-vibe/task-logs/vibe-round1
python3 stage1.py nsys-strata --no-second 2>&1 | tee -a nsys-runner.log
python3 stage1.py nsys-ours --no-second 2>&1 | tee -a nsys-runner.log
echo "runner done" >> nsys-runner.log
