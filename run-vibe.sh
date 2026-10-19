#!/usr/bin/env bash
# Runs Vibe (GLM-5.3) on one task file in the llama-vibe worktree, headless. The owner starts it.
# Vibe has no sandbox like Codex's: with the agent auto-approve every tool call runs, so the task file's rules are the
# only fence. Limits: wall clock (timeout, 3 h 15 min), turns and price (adjust VIBE_TURNS / VIBE_PRICE).
#   usage: run-vibe.sh [TASK-vibe-strata-parity.md]
#   log:   /opt/llm-gateway/data/build/llama-vibe/vibe-<task>.log (follow with tail -f; milestones in data/moe-quant/NOTE.log)
set -eu
W=/opt/llm-gateway/data/build/llama-vibe
TASK=$(basename "${1:-TASK-vibe-strata-parity.md}")
[ -f "$W/$TASK" ] || { echo "no such task in $W: $TASK" >&2; exit 2; }
OUT="$W/vibe-${TASK%.md}"
cd "$W"
echo "$(date +%H:%M:%S) Vibe: Start mit $TASK (Log $OUT.log)" >> /opt/llm-gateway/data/moe-quant/NOTE.log
rc=0
timeout 11700 nice -n 10 vibe --workdir "$W" --agent auto-approve --trust \
  --add-dir /opt/llm-gateway/data/moe-quant --max-turns "${VIBE_TURNS:-400}" --max-price "${VIBE_PRICE:-25}" \
  --output text -p "$(cat "$W/$TASK")" > "$OUT.log" 2>&1 < /dev/null || rc=$?
echo "$(date +%H:%M:%S) Vibe: Ende mit Status $rc ($TASK)" >> /opt/llm-gateway/data/moe-quant/NOTE.log
exit $rc
