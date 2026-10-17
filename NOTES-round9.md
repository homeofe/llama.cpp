# Round 9: real traces and GPU measurements

Started 2026-10-17 18:56 UTC; deadline 22:56 UTC. Worktree `llama-l5`, branch `l5-cache`. The handoff authorizes local stage commits, no push or PR, and no live runtime/settings changes.

## Initial state

The three real traces are present. The announced `data/moe-quant/qwen38/smoke-r8/` directory is absent at the start, so its claimed smoke results cannot inform selection. Gateway access works outside the command sandbox. GPU preflight reports no compute allocation. Historical round-7 and Strata figures remain historical controls until explicitly remeasured.

## Progress

Task 1: replaying the production policy with recorded group slots, the actual Qwen benchmark warm profile, cold Ling and cold long-text starts. Evaluating decay, upload caps, prefill caps, byte ranking, soft budgets and causal accepted-only learning.
