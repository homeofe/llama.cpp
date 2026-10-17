# Round 9: real traces and GPU measurements

Started 2026-10-17 18:56 UTC; deadline 22:56 UTC. Worktree `llama-l5`, branch `l5-cache`. The handoff authorizes local stage commits, no push or PR, and no live runtime/settings changes.

## Initial state

The three real traces are present. The announced `data/moe-quant/qwen38/smoke-r8/` directory is absent at the start, so its claimed smoke results cannot inform selection. Gateway access works outside the command sandbox. GPU preflight reports no compute allocation. Historical round-7 and Strata figures remain historical controls until explicitly remeasured.

## Progress

Task 1: 39 successful production-policy replays, using recorded group slots, the actual Qwen benchmark warm profile, cold Ling and cold long-text starts. All speculative spans resolve. Raw rows, commands and environment are in `task-logs/round9/replay.json`, with the readable table in `replay-table.md`.

| Trace | Round 7 byte hits | Byte ranking + decay 64 | Round 7 / candidate upload MiB per evaluated token |
| --- | ---: | ---: | ---: |
| Qwen benchmark, decode | 43.0767% | 44.6011% | 17.9206 / 18.6546 |
| Ling benchmark, decode | 55.4319% | 56.4842% | 28.1410 / 28.5434 |
| Qwen long German, all tokens | 36.6082% | 36.7259% | 2.2097 / 2.1907 |

The long trace is six perplexity chunks, not a long generated answer: 3072 of 3074 evaluated tokens are marked prefill. Decode-only coverage is therefore meaningless for this trace. Replay now also reports byte-weighted coverage over all evaluated tokens; prefill flags are preserved. Faster decay does not change this workload's prompt-only admission ranking.

Accepted-only learning loses coverage on both benchmark traces (Qwen 42.5405%, Ling 54.3841%). Soft budgets add churn without a useful gain. Smaller prompt bursts lose early coverage; admission 64 costs substantially more traffic. The selected GPU candidate is byte ranking, decay 64, admission 32, prefill cap 1024, no soft budget and no accepted-only runtime policy. This is a replay selection pending speed and quality gates.

The end-to-end run uses `stage_b.py` unchanged through `bench.py`; the wrapper only supplies the `codex: round 9` lease reason, clears inherited cache flags, and hashes full output. The original foreign-process gates, prompts, sampling and round rotation are retained. No load-average gate is enabled. The original harness and exact configuration dictionary are copied into each result directory.
