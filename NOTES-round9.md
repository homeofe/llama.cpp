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

Measurement caveat: the mandated foreign-process harness retries a contaminated request in the same server. That extra request also trains the persistent cache. For example, Ling round 2 retries `code-intervals`, and the later English outputs differ from round 1. Qwen's candidate repeats all eight hashes and draft acceptance counts exactly between its first two rounds. Report the prescribed request medians, but do not interpret sub-percent differences as an isolated kernel gain or ignore the cache-history effect of retries.

The three interleaved rounds finished at 19:55 UTC: 96 accepted requests, 12 observations per model/configuration/category. Data: `/opt/llm-gateway/data/moe-quant/qwen38/round9-policy/runs.jsonl`; aggregate: `task-logs/round9/policy-summary.json`.

| Configuration | Code tok/s | English tok/s | Code / English draft acceptance | Peak VRAM MiB |
| --- | ---: | ---: | ---: | ---: |
| Qwen round 7, remeasured | 46.8245 | 39.4449 | 91.94% / 68.56% | 10696 |
| Qwen byte + decay 64 | 47.0907 | 39.7055 | 92.92% / 72.27% | 10696 |
| Ling round 7, remeasured | 50.7870 | 45.7818 | 68.24% / 52.72% | 9940 |
| Ling byte + decay 64 | 50.6905 | 47.0570 | 67.28% / 53.32% | 9940 |

Only 9/24 output hashes match the control for either candidate. These are end-to-end observations with changed numerical/cache paths, not equal-work kernel comparisons. The policy is not a universal speed win; the conservative default remains decay 128 and byte ranking off pending all quality and subsequent slot experiments.

The replay metric checker passes against the new binary (exit 0) and fails against the pre-histogram binary (exit 1, missing histogram), as intended. The synthetic trace covers cache hits, misses, a dropped group and expert multiplicities 1, 2 and 4. Quality runs started after GPU release and the CPU-only checker/probe builds.
