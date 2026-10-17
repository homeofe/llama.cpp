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

Task 1 completed at 20:52 UTC. All six quality checks pass. Ling cache-off is 4.5559; both round 7 and byte/decay64 are 4.5498 (-0.1339%). Ling's single-size expert layout and prefill-only quality workload make the two policy runs identical here. Its disabled-cache runs are byte-identical over 128 positions, 80478208 bytes, SHA256 `83429a23024228e61b2f64515128019cc75b04e30bc97c7dfd853580094a0552`. The normal cache policy unit suite also passes. Full quality evidence is indexed in `task-logs/round9/quality-summary.json`.

Chosen policy defaults remain round 7: adaptive LFU, decay 128, admission 32, prefill warming enabled (cap 1024), byte ranking off, no soft per-layer budget, no accepted-only learning. The small aggregate policy gains do not establish a robust cross-model win. In particular, Ling's pooled English median rises, but three of its four per-prompt medians decline slightly; changed outputs/cache history prevent treating the pooled change as a uniform speedup.

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

## CPU and GPU profiling

The instrumented Qwen round-7 server completed all eight original requests with compaction disabled. This is diagnostic evidence, not another speed comparison. Full data are in `task-logs/round9/q-warm-draft-profile/`; `cpu-profile-summary.json` contains the aggregate. `perf` collected 564k cycle samples with zero lost samples. About 57% of samples are in libgomp spin loops (verified in `openmp-spin.txt`); IQ4_NL, IQ3_XXS, IQ2_S and IQ3_S together account for 30.85%. Busy CPU cores therefore substantially overstate useful arithmetic. The sampled task-clock and cycle counters imply about 3.91 GHz while running; the request-phase reported CPU frequency median is 4.055 GHz. There is no evidence here of an unenabled AVX2 path or a CPU stuck at idle clocks.

The CPU backend wrapper records 25,970 four-token graph calls taking 17.157 seconds, versus 41.823 seconds of decoded output time across the eight requests. These graph durations include warmup and can overlap GPU work, so this is not an exclusive critical-path decomposition. OpenMP creates a parallel region for every CPU graph in the existing build. A bounded wait-policy and thread-placement experiment is justified after the mandatory drafter and multiplicity stages; reducing spin cycles alone is not proof of faster tokens. The relevant control is documented by [GNU libgomp](https://gcc.gnu.org/onlinedocs/gcc-9.5.0/libgomp/GOMP_005fSPINCOUNT.html).

Across 252 request-phase GPU sensor samples, the link stays at Gen3 x16, GPU activity averages 43.86% (maximum 64%), memory activity averages 16.17%, power averages 147 W (maximum 188 W), and SM clock reaches 1890 MHz. These activity sensors do not measure how many FP32 lanes execute useful operations. All four populated memory channels are also exposed by the temporarily loaded `amd_uncore` module: DF controllers 2, 3, 4 and 5 count traffic; 0, 1, 6 and 7 remain zero in the first scan. The module will be unloaded after measurement. The Linux Zen 2 event definitions count 64-byte DRAM data requests: [primary event source](https://raw.githubusercontent.com/torvalds/linux/master/tools/perf/pmu-events/arch/x86/amdzen2/data-fabric.json).

Nsight Systems initially failed its automatic importer lookup; invoking the installed matching `QdstrmImporter` directly recovered the existing capture without another GPU run. The 80-token target-only diagnostic spans 1.906 seconds from first to last CUDA kernel. It contains 79,652 kernels, 622 ms of union kernel activity and 135 ms of union copy activity (751 ms combined union). Host-to-device copies transfer 1.394 GB in 112 ms summed copy duration. CUDA graphs are disabled for this diagnostic, so launch gaps and overlap cannot be extrapolated to the graph-enabled production benchmark. Raw capture and SQLite are kept locally; `nsys-summary.json` records the reproducible aggregate. Initial Nsight Compute matched no kernels because its demangled booleans are `(bool)1`, not `true`; the corrected filter is queued before advancing to DSpark.

DF counters during the first slot screen provide 36 complete one-second samples within Ling control requests: mean 33.110 GB/s, minimum 27.972 and maximum 38.172 GB/s. All four counters ran continuously (100%, no multiplexing); per-channel traffic is balanced. This is whole-system DRAM traffic across an entire request, including GPU/CPU-wait phases, not the peak bandwidth of an isolated CPU kernel. The measured intervals and calculation inputs are in `dram-slots-summary.json`, `dram-slots.csv` and `dram-slots-meta.json`.

## Task 2: more slots

The first actual allocation tests fit Qwen target 2944 MiB plus draft 512 MiB at 10156 MiB VRAM, and Ling target 6144 MiB at 10334 MiB. Both use draft ubatch 32. Qwen has more headroom than the prior estimate: another actual-fit test at target 3456 MiB is queued, followed by a same-condition comparison and quality checks.

The initial one-round screen finished at 21:17 UTC (32 accepted requests, four observations per category). Every request passed the original foreign-CPU gate on the first attempt; compaction was 0 throughout. Raw data: `/opt/llm-gateway/data/moe-quant/qwen38/round9-slots/`. These screening numbers do not replace the three-round policy comparison.

| Configuration | Code tok/s | English tok/s | Peak VRAM MiB | Matching control outputs |
| --- | ---: | ---: | ---: | ---: |
| Qwen round 7 control | 46.9894 | 39.7515 | 10696 | 8/8 |
| Qwen target 2944, draft ubatch 32 | 47.8567 | 39.2530 | 10160 | 5/8 |
| Ling round 7 control | 50.5623 | 47.2623 | 9940 | 8/8 |
| Ling target 6144, draft ubatch 32 | 51.4992 | 45.7750 | 10338 | 3/8 |

Qwen's four code outputs and draft counts match exactly; their individual gains are 1.48-2.31%. English acceptance falls from 69.57% to 65.56%, with three changed outputs. Ling's code median rises 1.85%, but English falls 3.15%; English acceptance falls from 53.02% to 49.96%, with all four outputs changed. Thus the first larger-cache screen does not justify promoting a universal speed improvement.

The corrected Nsight Compute job reached its ten-minute limit after four complete kernel profiles and part of a fifth. Only the four complete records are imported; the failed overall run remains recorded as exit 124. Each completed record required 31-32 replay passes; backing up device memory to system RAM dominated the profiler's wall time and blocked NVML queries for minutes. There are no remaining processes in the owned profiler group. The saved report is readable offline, so repeating this expensive capture is unnecessary.

The completed IQ3_S/IQ4_NL GPU-hit kernels show achieved warp occupancy of 43.60-64.92%, SM throughput of 20.83-35.58% of its reported peak, only 0.43-0.55 eligible warps per scheduler per active cycle, and L2 hit rates of 81.31-89.36%. Actual DRAM throughput is 30.75-58.84 GB/s for these small kernels; DRAM active cycles are 4.83-9.02%. The dominant sampled stalls include local/global memory instruction queue pressure, long-scoreboard dependencies and execution waits. Thus these kernels are not simply saturating the GPU's external memory bus or FP32 units. HMMA and IMMA executed-instruction counters are zero: these IQ matrix-vector kernels do not use tensor matrix instructions. That is not a disabled hardware feature that can be enabled without a different algorithm/data layout. The integer arithmetic and float arithmetic pipes both execute work.

These are diagnostic kernel-replay measurements, with profiler-controlled clocks/cache handling and no CUDA graphs. They are not end-to-end speed results. This tool version exposes only the aggregate for `sm__cycles_active.sum`; `--print-metric-instances values` does not produce 68 per-SM values for that counter. Therefore the measurements prove substantial activity and limited occupancy, but not that every individual SM or every one of the 4352 FP32 lanes is simultaneously busy. Evidence and units: `task-logs/round9/gpu-ncu-retry/summary.json`, `metrics.csv`, `instances.csv`, saved `kernels.ncu-rep`.

The extended Qwen allocation fits 3456 MiB target plus 512 MiB draft cache, with draft ubatch 32: 10646 MiB in the fit request and 10650 MiB peak in the full screen. The additional comparison finished at 21:35 UTC: target 2944 yields 47.5094 code / 39.1679 English tok/s; target 3456 yields 48.3685 / 40.1187. Four code outputs and draft counts match, with per-prompt gains of 0.43-2.25%. Five of eight text hashes match overall; the English numbers include changed draft acceptance (65.56% to 69.25%). One screening round cannot establish a robust new default. Evidence: `slots-extra-summary.json`, `slots-extra-runs.jsonl`, and the exact `slots-extra-configs.json`.

The quality checks completed at 21:41 UTC: Qwen target 3456 PPL 3.8126 (+0.0866% against cache-off 3.8093); Ling target 6144 PPL 4.5513 (-0.1010% against 4.5559). Both meet the +0.5% limit with 12 chunks, context 512 and ubatch 4. The actual cache holds 1248 / 1438 / 1690 Qwen target expert slots at 2560 / 2944 / 3456 MiB, and 1668 / 2001 Ling slots at 5120 / 6144 MiB. Route-count hit rates rise from 35.70% to 39.68% for Qwen, and 45.33% to 48.75% for Ling; these are not byte-hit rates. The limited extra coverage is consistent with the modest measured gains.

## Task 3: Ling DSpark

The existing converter supports `LingDSparkModel` through its DFlash-derived implementation. The `--dspark` export switch mentioned in the handoff is specific to DeepSeek-V4 and must not be applied here. Pinned draft revision `d60b0e1291dacb256194c6d48154398730cf8986` converted successfully to Q8_0 with target tokenizer metadata; no production source change was needed. Exact commands: `dspark-preparation.json`. The first actual runtime request succeeds, using 5832 MiB without expert cache.

The bound uses the vendor's domain-specific acceptance lengths, which include the bonus token: code 5.33-6.57, English 3.51-3.92, math 6.29-6.40. Current MTP lengths are roughly 2.93-3.23 for code and 2.29-2.77 for English. Those imply gross token-per-round opportunities of roughly 2x / 1.4x, before the larger verification window and separate draft-model cost. We use seven draft tokens plus the anchor (block size eight) because the hybrid cache supports at most eight verification tokens; this is smaller than the vendor's eight draft tokens plus bonus. `--spec-draft-p-min 0` is explicit and matches the current source default; there is no implicit 0.75 confidence cutoff. Actual observed mean lengths and end-to-end throughput decide the result.

DSpark's actual fit leaves room for a 4608 MiB target cache (1540 slots): 10440 MiB measured total. The separate 12-chunk cache-budget quality check passes at PPL 4.5420 (-0.3051% versus cache-off). This is a check of the target's cache budget, not a perplexity claim about speculative acceptance. The eight-prompt MTP / DSpark-off / DSpark-cache comparison started at 21:53 UTC, with `n_max=7`, `p_min=0`, draft ubatch 32 and compaction disabled for every variant.

For the later CPU experiment, the wrapper additionally samples the owned server's aggregate `/proc/PID/stat` user+system ticks before and after each request. `engine_cpu_s_per_token` includes prompt work and waiting spins across all threads; it is CPU resource cost, not energy in joules or decoded-token latency. Thread affinity is archived after the uncounted warmup. The original benchmark harness and its acceptance gates remain unchanged.

Native-pool comparison detail: `libggml-base` still links libgomp, so `OMP_PROC_BIND=close` can narrow inherited masks even when only `libggml-cpu` is rebuilt without OpenMP. The native candidate therefore uses the same explicit 24-physical-core masks as the OpenMP physical-core variant. The parity probe also uses explicit physical masks. This avoids accidentally benchmarking all native workers on one inherited core pair. The untouched OpenMP variant with those masks is the appropriate control for isolating the threadpool difference.

Task 3 finished at 22:06 UTC: 24 accepted requests, all first-attempt foreign-CPU gates passed. One screening round, four requests per category, same compaction setting. The new drafter changes verification width and numerical execution, so output differences are disclosed rather than treated as identical-work kernel gains. Full output hashes, draft counts, CPU ticks and server logs are in `task-logs/round9/dspark-results/`; per-request observed mean lengths are in `dspark-lengths.json`.

| Ling configuration | Code tok/s | English tok/s | Code / English acceptance | Median accepted length incl. bonus (code / English) | Peak VRAM MiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| MTP3, 5120 MiB cache | 50.6659 | 47.4519 | 67.88% / 53.02% | 3.045 / 2.665 | 9940 |
| DSpark7, cache off | 64.0945 | 32.7139 | 81.87% / 33.44% | 7.135 / 3.510 | 5862 |
| DSpark7, 4608 MiB cache | 76.4583 | 41.1413 | 78.28% / 32.87% | 7.135 / 3.425 | 10476 |

DSpark with cache improves the code median by about 50.9%, but loses about 13.3% on English. Its CPU cost per generated token, including prompt work, is 0.3979 CPU-seconds for code versus MTP's 0.5432; English rises from 0.5071 to 0.5349. This is a promising code-specific profile, not a universal replacement. The vendor's math acceptance length cannot be substituted for actual English behavior. The general-purpose default remains MTP3 pending any bounded confidence/shorter-window follow-up. These Ling results must not be presented as beating Strata's Qwen result: they are different models.

## Task 4: four-token CPU-miss multiplicity

The six real-trace replays finished after the drafter comparison, using the production cache policy, actual slot layouts and all missed experts including uncached groups. The selected round-7 policy gives Qwen decode means IQ4_NL 1.2664, IQ3_XXS 1.2441, IQ2_S 1.2686 and IQ3_S 1.3316; all expert triplets together are 1.2743. Every relevant IQ type fails the handoff's approximately 1.6 gate. Byte/decay64 is slightly lower. Therefore no new Qwen multi-token IQ kernel family is built. This is an evidence-based stop, not an untested implementation omission. Full per-type integer histograms and byte-weighted means are in `multiplicity.json`.

Qwen benchmark prefill means are higher (about 1.92-2.02 for these types), but do not describe the generation bottleneck. The long German trace is overwhelmingly perplexity prefill; it cannot establish generated-token reuse. Keep those phases separate. Existing CPU graph timing and the large OpenMP spin fraction still justify task 5.

## Task 5: CPU screen

The bounded screen compares the unchanged round-7 Qwen configuration, a shorter OpenMP spin count, 16 physical cores spread across all eight CCX groups, all 24 physical cores, all 48 SMT threads, and the existing native ggml threadpool. Both target and draft batch threads are set explicitly: four-token verification uses `-tb`, so changing only `-t` would not test the intended work. Native polling is explicitly 1 for target/draft and their batch pools; its CPU library and wait policy form one candidate, not an isolated compiler-flag comparison. The native variant is admitted only after full-vocabulary cache-off logits are bit-identical and mapped-library inspection proves that only `libggml-cpu` was replaced.

The graph-enabled Nsight check completed successfully at 22:08 UTC and records 1279 `cudaGraphLaunch` runtime calls. CUDA graphs are therefore actually used by the tested target path. The installed Nsight version defaults to graph-level tracing, which omits individual graph-node activities; its 21179 captured standalone kernels are not a complete GPU timeline. Do not divide their union time by the capture span to claim GPU idle percentage. The graph-disabled diagnostic remains separately labeled. Exact commands, runtime calls and this limitation are in `gpu-nsys-graphs/command.json` and `nsys-graphs-summary.json`.

The DSpark-cache code gain is especially useful because all four generated code outputs have exactly the same text hashes as the MTP control. Their individual speed gains are +15.57%, +49.41%, +52.39% and +60.35%. Draft counts intentionally differ between these two algorithms; that does not invalidate an end-to-end drafter comparison with identical final code. All four English outputs differ and all four English rates decline.

The native-pool correctness gate passes: 128 positions across four topics, 127139840 bytes of complete F32 vocabulary logits, SHA256 identical to the original disabled-cache run (`b7facadb625ef4a18eb43b74959a4daa766b477a0dd2bb0bdeec6f0b984fd5a0`). Mapped libraries show the new native CPU library and the original CUDA/base/llama libraries. Compiler command comparison finds only `-DGGML_USE_OPENMP` and `-fopenmp` removed from both CPU core and x86 quantizer sources. Evidence: `native-pool-proof.json`, `native-compile-comparison.json` and `qwen-off-nativepool-probe/`.

The CPU screen completed all eight requests for each of the five OpenMP variants. Every output hash and draft count matches the control, so these comparisons use identical generated work. One screening round, four requests per category. The native variant was stopped after a catastrophic regression; its incomplete English category must not be presented as an eight-prompt result. Full records, flags and observed thread masks are in `cpu-results/`; `cpu-screen-outcome.json` records completion and the intentional stop.

| Qwen CPU configuration | Code tok/s | English tok/s | CPU s/token code / English (includes prompt) | Request-phase DRAM GB/s |
| --- | ---: | ---: | ---: | ---: |
| qwen-cpu-control | 46.9723 | 39.8787 | 0.6483 / 0.6284 | 27.656 |
| qwen-cpu-spin1000 | 39.7876 | 33.9927 | 0.4377 / 0.3889 | 24.279 |
| qwen-cpu-t16-spread | 44.5442 | 37.6058 | 0.4714 / 0.4502 | 25.978 |
| qwen-cpu-t24-physical | 47.0124 | 39.7647 | 0.6484 / 0.6301 | 27.585 |
| qwen-cpu-t48-smt | 41.2761 | 34.8049 | 1.4406 / 1.4151 | 24.388 |

The 48-thread variant really uses masks 0 through 47; it loses throughput and more than doubles CPU seconds per token. Sixteen physical cores and shorter spin waiting reduce CPU cost but also lose speed. Strict physical placement at 24 threads is essentially neutral. Retain the original 24-thread OpenMP setup. The DF intervals use all four controllers without multiplexing; they measure complete request phases, not isolated peak RAM bandwidth. No BIOS, governor, channel or PCIe configuration switch was found to explain the performance gap.

### Native threadpool integration failure

The isolated target-only native library passes bitwise correctness, but the server with MTP does not have equivalent threadpool integration. `common_speculative_init_result::impl` owns only the draft model and context (`common/speculative.cpp`); its constructor never attaches `common_threadpools`, unlike `common_init_result`. The CPU backend starts with a null threadpool, passes it through `ggml_graph_plan`, and `ggml_graph_compute` creates a disposable native pool when the pointer is null. The target's explicit native placement moves the caller to CPU 23. The draft fallback workers inherit this placement instead of the requested draft mask. The warmup thread snapshot shows the persistent target workers on CPUs 0-22, the caller on 23 and a transient extra worker on 23. This source/affinity evidence explains the observed approximately 1.7-2.2 busy cores and 1.3-1.6 tok/s despite correct output.

Four code requests and one English request completed before SIGTERM stopped our benchmark at 22:40 UTC; the remaining native requests are deliberately incomplete. The command to stop it required sandbox escalation, and the stop occurred substantially later than the decision to reject the candidate. The already queued optional confidence screen reached its 22:31 start cutoff and skipped itself. No confidence-threshold or fixed-shorter-DSpark result is claimed. The driver originally stamped a generic completion marker even after the benchmark's graceful stop; that marker is preserved as `cpu-screen-stopped.txt`, and the driver now checks `all_complete` before issuing a success marker.

A proper C++ follow-up is to give the speculative-init owner persistent draft threadpools with context-before-pool destruction, propagate the draft CPU parameters, and verify both external-draft and same-model MTP contexts. Test pool reuse/placement and teardown, use a mutation removing the attach call, recheck output hashes and 12-chunk PPL, and require at least 3% end-to-end gain before adoption. Merely disabling OpenMP is not a valid optimization of this server path. No production-source change or runtime installation was made for this unsuccessful experiment.

## Final decisions and end-to-end table

All five mandatory directions have a measured decision. Retain round-7 policy defaults and the original 24-thread OpenMP build for general use. More slots fit, but their gains are small and output/acceptance changes prevent a universal recommendation from the single screening rounds. Ling DSpark with 4608 MiB cache is a substantial code-specific candidate with identical code outputs and a passing target-cache PPL gate; keep MTP3 for English. The CPU multi-token rewrite is stopped by the predeclared multiplicity bound. The native-threadpool build is rejected in the current MTP integration.

| Model / engine / configuration | Code tok/s | English tok/s | Evidence |
| --- | ---: | ---: | --- |
| Qwen, no cache | 40.1 | 32.8 | Historical handoff |
| Qwen, round 7 | 47.0 | 39.8 | Historical handoff |
| Qwen, round-7 settings, fresh CPU control | 46.97 | 39.88 | Round 9, one eight-prompt pass |
| Qwen, target cache 3456 MiB, draft ubatch 32 | 48.37 | 40.12 | Round 9 screening pass; separate fresh 2944 control |
| Qwen, Strata | 70.8 | 61.0 | Historical separate engine; not rerun or modified |
| Ling, round 7 | 50.8 | 47.5 | Historical handoff |
| Ling, MTP3 with 5120 MiB cache | 50.67 | 47.45 | Round 9 drafter control |
| Ling, DSpark7 without cache | 64.09 | 32.71 | Round 9 screening pass |
| Ling, DSpark7 with 4608 MiB cache | 76.46 | 41.14 | Round 9 screening pass; four identical code hashes |

The new slot, drafter and CPU rows use compaction 0. Historical rows and the earlier three-round policy comparison must not be pooled as same-condition repetitions. Four prompts contribute to each category in each screening pass. The policy experiment has three interleaved rounds and twelve observations per category; it is reported separately above. No new default is justified by a single small or mixed gain.

Exact engine arguments and environment are exported in `task-logs/round9/tested-engine-profiles.json`. This is a reviewable profile artifact, not an applied gateway configuration. For the Ling code candidate the decisive additions are the pinned Q8_0 DSpark model, `--spec-type draft-dspark --spec-draft-n-max 7 --spec-draft-p-min 0 --spec-draft-n-ubatch 32 -ngld 99`, with adaptive target cache 4608 MiB, decay 128, admit 32, prefill warming 1024. Its GGUF size and SHA256 are in `dspark-artifact.json`.

Quality evidence is consolidated in `quality-summary.json`: Qwen cache-off 3.8093, retained R7 3.8025, byte/decay64 3.8117, 3456-MiB slots 3.8126; Ling cache-off 4.5559, R7 and byte/decay64 4.5498, 6144-MiB slots 4.5513, DSpark's 4608-MiB target-cache budget 4.5420. Every listed candidate passes the +0.5% limit. The intermediate Qwen 2944-MiB allocation is only a screening control, not separately quality-qualified for deployment. Cache-off logit parity passes for both models and for the isolated native CPU library.

## Restoration and scope

At 22:41:04 UTC the guard restored `vm.compaction_proactiveness` from 0 to its original 20. At 22:41:05 UTC restoration verified the temporary `amd_uncore` module unloaded, zero GPU compute processes, no gateway GPU hold, no loaded gateway model and no active request. All owned orchestration sessions have exited. Evidence: `compaction-restored.json` and `restoration.json`. No toolkit installation, BIOS change, governor change, permanent sysctl, live runtime change, gateway settings edit, push or PR occurred.

The production `src/` and `ggml/` trees remain unchanged from the starting runtime source. The C++ source change is confined to replay instrumentation in `tests/test-moe-cache.cpp`, with the positive metric checker and old-binary negative cut archived. Python helper type checking and `git diff --check` pass. Native compilation changes only OpenMP build flags; that build remains a rejected worktree artifact. Small raw logs/configurations supporting the report are committed; large profiler captures, full logit dumps and model/build binaries remain local with their summaries or hashes. Pre-existing unrelated worktree files are preserved.
