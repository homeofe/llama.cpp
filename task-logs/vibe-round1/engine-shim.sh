#!/usr/bin/env bash
# Engine shim for profiling: the Strata server spawns this as the engine (config copy "exe" points
# here). nsys wraps the real engine directly, so its CUDA activity is traced; the server's stdin/
# stdout pipes pass through nsys untouched, and QUIT (server shutdown) lets nsys write its report.
# The server calls: engine-shim.sh --serve <engine-args...>.  Real engine path from $STRATA_REAL_EXE.
set -u
exec nsys profile --trace=cuda,osrt,nvtx --sample=none --cpuctxsw=none \
     --kill=none --force-overwrite=true --delay=2 --duration=1800 \
     -o "${NSYS_OUT:-/opt/llm-gateway/data/build/llama-vibe/task-logs/vibe-round1/strata-shim/strata-shim}" \
     "${STRATA_REAL_EXE:?set STRATA_REAL_EXE}" "$@"
