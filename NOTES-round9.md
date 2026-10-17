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

Qwen quality: cache-off 3.8093, round 7 3.8025 (-0.1785%), byte/decay64 3.8117 (+0.0630%). Both pass the owner's +0.5% limit. The two disabled-cache probe runs are byte-identical over four topics, 128 positions and 127139840 bytes of full-vocabulary F32 logits (SHA256 `b7facadb625ef4a18eb43b74959a4daa766b477a0dd2bb0bdeec6f0b984fd5a0`). These use the unchanged production CUDA libraries; the new C++ instrumentation is confined to the replay test executable.

Quality scheduling initially counted the entire model load, inference and teardown as foreign-process observation. Ling's first probe was discarded at 2.59 foreign cores; the second passed at 0.60. Subsequent quality jobs gate the actual evaluation phase (after model/threadpool readiness), retaining whole-process CPU figures separately. This avoids confusing loading/teardown activity with inference load. Quality-job elapsed times are not throughput benchmarks. The required end-to-end harness remains unchanged.

## Hardware audit requested during the run

MSI Creator TRX40 (MS-7C59), BIOS 1.98 dated 2026-06-15; Threadripper 3960X, 24 physical cores plus SMT, eight 16 MiB L3 groups, one NUMA node. DMI reports eight dual-rank 16 GiB DIMMs, two in each of channels A/B/C/D, all configured at 3200 MT/s. Thus all four memory channels are populated; there is no missing-channel explanation for the performance gap. This is firmware-reported configuration, not a measured DRAM bandwidth result.

The RTX 2080 Ti has a full x16 link, with device capability 8 GT/s (PCIe 3.0). Initial idle samples show Gen1 and low clocks, which is expected power management. During Ling perplexity at 20:19 UTC the link reaches Gen3 x16, SM clock 1350 MHz, memory clock 6800 MHz, GPU utilization 35%, memory utilization 11%, power 78.11 W. This single sample does not measure SM occupancy or prove that all 4352 FP32 lanes are busy. CPU governor is `schedutil`, boost is enabled. Transparent hugepages are `madvise`; the snapshot reports no hugepage use. No system, firmware, governor or memory settings were changed.

The actual compiler command uses `-O3 -march=native -fopenmp`; disassembly confirms AVX2 instructions in the IQ3 kernel. The separate CMake AVX2/FMA booleans being OFF does not disable those instructions when native compilation is enabled. CUDA is built with `-O3` and `sm_75`; CUDA Graphs and Flash Attention are enabled. Merely enabling these basic features is therefore not an unexplored source of a large gain. Exact compiler flags, topology and hardware facts are archived in `task-logs/round9/hardware.json` and `cpu-topology.json`.

Installed tools verified: CUDA Toolkit/nvcc 12.0.140, Nsight Compute 2022.4.1.0, Nsight Systems 2022.4.2.50. No toolkit installation is required to begin profiling. GPU hardware counters have `RmProfilingAdminOnly=1`; the owner explicitly authorized passwordless sudo. Prepared diagnoses use a bounded perf sample on our server and Nsight on the real four-token cached target path. Nsight disables CUDA graphs to expose kernels on this tool version; these runs must not be used as production speed comparisons. The normal benchmark retains graphs.

## Memory compaction investigation

Three Ling cache-off PPL runs each produced 4.5559, but their evaluation phases failed the foreign-CPU gate (1.30-1.36 cores). Independent per-process accounting over the complete 89-second phase identified `kcompactd0` at 0.995 cores. Signed versus per-interval-clamped accounting gave the same result: accounting bias was not the explanation. The quality harness now sums signed deltas before clamping the complete observation, matching the original performance harness's aggregate method.

A process-local `PR_SET_THP_DISABLE` wrapper was verified with `THP_enabled: 0`; it did not solve the issue: `kcompactd0` still used 0.993 cores, total foreign activity 1.320 cores. The process used about 60.5 GiB RSS and zero swap. This is direct evidence that the observed penalty is not model swapping. The failed timing gates remain failed; these diagnostics are not accepted speed results. Raw process/time accounting: `cpu-accounting.json` and `cpu-accounting-nothp.json`.

The owner explicitly approved the global temporary experiment and said the machine was exclusively available for these measurements. At 20:39:37 UTC, `compaction_guard.py` saved `vm.compaction_proactiveness=20` and set it to 0. It restores the previous value on release, termination or the 22:56 UTC deadline. This narrowly authorized exception supersedes the handoff's write-location restriction for this one kernel setting. No permanent sysctl configuration is written. Subsequent controls and candidates must state that this setting is 0; they must not be silently pooled with the earlier measurements at 20. See Linux's [VM documentation](https://docs.kernel.org/admin-guide/sysctl/vm.html#compaction-proactiveness).

The first compaction-disabled run passes: independent whole-phase foreign CPU is 0.345 cores (quality harness 0.348), down from 1.364. Cache-off PPL remains exactly 4.5559. This identifies avoidable background work, but is not an interleaved tokens/s comparison. Use fresh controls under the same system setting for the subsequent slot and drafter comparisons. Evidence: `cpu-accounting-compaction0.json`, `ling-off-compaction0-ppl/attempt-1.log`.
