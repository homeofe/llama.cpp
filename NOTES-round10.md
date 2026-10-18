# Round 10: Ling confirmation, panels and whole-round profiling

Started 2026-10-18 07:00 UTC; deadline 11:00 UTC (13:00 Europe/Berlin), explicitly confirmed by the owner after lifting the separate four-hour constraint. Branch `l5-cache`. No system changes, live gateway changes, pushes or PRs. The owner subsequently authorized CPU sampling and sudo explicitly; use those for task 3 without changing kernel settings.

## Measurement protocol

The unchanged stage-B foreign-process gate is reused. A rejected request invalidates its whole prompt pass; restart the server before retrying, so the adaptive cache always sees the same warmup and eight-prompt history. Screening: MTP3 control and seven DSpark settings, four code and four English prompts per setting. Confirmation: best two candidates and control, three rotating rounds. Raw output hashes, draft counts, accepted lengths, CPU cost and exact configurations are archived.

## Screening (completed)

| Configuration | Code tok/s | English tok/s | Acceptance code / English | Mean length code / English |
|---|---:|---:|---:|---:|
| ling-cache-mtp3 | 50.7452 | 47.4517 | 67.88% / 53.02% | 3.045 / 2.665 |
| ling-dspark-n7-p00 | 77.3175 | 41.7307 | 78.28% / 32.87% | 7.135 / 3.425 |
| ling-dspark-n3-p00 | 66.3210 | 51.6762 | 95.84% / 57.64% | 3.870 / 2.830 |
| ling-dspark-n4-p00 | 69.8386 | 51.6521 | 86.64% / 48.45% | 4.755 / 3.155 |
| ling-dspark-n5-p00 | 73.9066 | 49.5328 | 84.69% / 43.17% | 5.570 / 3.345 |
| ling-dspark-n7-p30 | 76.6976 | 45.4170 | 79.02% / 43.26% | 7.040 / 3.570 |
| ling-dspark-n7-p50 | 76.8802 | 47.1952 | 89.57% / 61.11% | 7.135 / 3.075 |
| ling-dspark-n7-p70 | 78.3123 | 47.4197 | 89.35% / 79.02% | 7.040 / 3.180 |

Four prompts per category, one complete clean pass per configuration. Three foreign-load failures discarded the entire pass before retrying with a fresh server/cache. All 64 accepted requests are archived in `task-logs/round10/screen-results/`.

The original selection rule chose n3/p0 and n4/p0: both beat code and English, ranked by the lower of their two category/control ratios. The fastest-code candidate n7/p0.7 had English just 0.0668% below control. Before confirmation, it was added as a boundary candidate (screening proximity within 1%, not a final non-inferiority criterion). Thus confirmation has the required best two plus control and one additional candidate, three rotating rounds, twelve requests per category per configuration. The initial control load was interrupted before any request; its artifacts are kept in `data/moe-quant/qwen38/round10-confirm-aborted-load/`.

Mean length includes the bonus token and is the median of four request means. Speed uses the unchanged stage-B client first-to-last streamed-token metric. Raw server timing is also retained; a batched final token/stop event can produce a boundary difference, especially on short answers. Full output hashes are in `task-logs/round10/screen-summary.json`; input prompt texts and SHA256 hashes are in `task-logs/round10/prompts.json`.

## Panel protocol

Historical controls were read, not modified: Qwen iq3_s hard 56/56 and agent 69/72; Bolt hard 41/42 and agent 36/36. These are different models and older runs, not paired runtime controls. The unchanged panel executable will run with three seeds, card sampling and thinking enabled, through port 8791. A local proxy adds the foreign-process gate and archives complete request/response hashes. The chosen speed profile has an 8192-token context; the approximately 20000-token long-document case exceeds that limit and must be disclosed separately, without silently changing the tested allocation.

Ling card provenance: inclusionAI/Ling-3.0-flash README, revision `fcc60a52b34c69c6271d0ac8a8cb0815020b6a43`, downloaded read-only via hf CLI into `data/moe-quant/round10-metadata/target`. It recommends temperature 0.6, top_p 0.95, top_k 20 and thinking enabled.

## Profiling preparation

The ordinary llama-l5 backend has no GGML_SCHED_TRACE instrumentation. Its DEBUG_TIMINGS server scope is not compiled. No instrumentation build is made. The existing round-9 CPU graph interposer can provide measured graph totals, while Nsight records CPU stacks, CUDA APIs, kernels, transfers and request NVTX ranges. CPU sampling initially failed under perf_event_paranoid=4; after the owner explicitly authorized sudo, the unchanged kernel setting passes Nsight process-tree sampling checks. Use graph-disabled traces for individual GPU activity and separate graph-enabled wall measurements.

A CPU-only Nsight smoke capture during model loading succeeded: 316 sample events, 4235 stack frames and a bounded 200 ms NVTX range. The UID-dropping launcher exited 134 without a trace; launching the measured process under the explicitly authorized sudo profiler succeeded. Captures therefore use root-owned measurement processes and worktree-local output/cache directories. No kernel settings changed. The installed CLI cannot locate its importer automatically; the matching existing `/usr/lib/nsight-systems/host-linux-x64/QdstrmImporter` plus `nsys export` successfully converts the raw capture. Plain-string capture ranges require the per-process `NSYS_NVTX_PROFILER_REGISTER_ONLY=0`, as documented in the [NVIDIA guide](https://docs.nvidia.com/nsight-systems/UserGuide/index.html). Evidence: `task-logs/round10/profiler-smoke/summary.json` and logs.

## Confirmation (completed)

96 accepted requests: four configurations, three rotating rounds, twelve requests per category/configuration. Three whole passes were discarded after the unchanged foreign-load gate rejected one request. Every configuration/prompt produced one stable output SHA256 and identical draft/accept counts across its three repetitions. Full hashes and per-prompt rates: `task-logs/round10/confirm-summary.json`; exact raw runs and logs: `confirm-results/`.

| Configuration | Code tok/s | English tok/s | Acceptance code / English | Mean length code / English | Peak VRAM MiB |
|---|---:|---:|---:|---:|---:|
| ling-cache-mtp3 | 50.893 | 47.509 | 67.88% / 53.02% | 3.045 / 2.665 | 9940 |
| ling-dspark-n3-p00 | 66.509 | 51.854 | 95.84% / 57.64% | 3.870 / 2.830 | 10138 |
| ling-dspark-n4-p00 | 69.870 | 51.865 | 86.64% / 48.45% | 4.755 / 3.155 | 10246 |
| ling-dspark-n7-p70 | 78.319 | 47.402 | 89.35% / 79.02% | 7.040 / 3.180 | 10672 |

**General recommendation: DSpark n3, p_min 0, target cache 4608 MiB.** Code +30.68%, English +9.15% relative to MTP3/cache5120. All eight individual prompt medians improve. The four English gains are +4.32%, +8.31%, +12.88%, +14.12%. n4 raises the category medians (+37.29% code, +9.17% English), but loses 8.75% on en-2. This is why n3 is preferred for mixed workloads.

**Code-focused alternative: n7, p_min 0.7.** Code +53.89%; English -0.23% overall, including -5.36% on en-1. This is not a demonstrated English tie. Selection and paired ratios are in `recommendation.json`. These findings cover the eight specified prompts, not all possible workloads.

Across configurations, n3 matches MTP text on three of four code prompts; n4/n7 match all four. All English hashes differ from MTP. Repetition stability does not establish cross-configuration bit equivalence. Cache-off full-logit checks and the task panels provide separate evidence. Mean accepted lengths include the bonus token; category values are medians of request means, not token-weighted averages.

Exact commands and environment are in `task-logs/round10/configs.json` and each raw run. The recommended profile uses `--spec-type draft-dspark --spec-draft-n-max 3 --spec-draft-p-min 0 -ngld 99 --spec-draft-n-ubatch 32`, the existing DSpark Q8_0 file, target IQ4_XS, context 8192, batch 2048, ubatch 512, 24 threads, full dense offload and CPU experts with adaptive GPU expert cache 4608 MiB (admit32, prefill1, decay128). Control uses MTP3/cache5120. No live profile has been changed.

## Fresh numerical quality (completed)

All three checks passed the unchanged foreign-process gate without system changes. Full-vocabulary cache-off probe: 80,478,208 bytes, bit-exact against the round-9 cache-off reference despite changed inactive policy knobs; SHA256 `83429a23024228e61b2f64515128019cc75b04e30bc97c7dfd853580094a0552`. Foreign CPU: 0.223 cores.

Twelve test-de chunks, context512, batch512, ubatch4: cache off PPL **4.5559**, adaptive4608 PPL **4.5420** (-0.3051%, within +0.5%). Evaluation foreign CPU 0.559 / 0.660 cores. Both exactly reproduce the prior numerical estimates with the original compaction setting20. These are target/cache tests, not a perplexity measurement of speculative acceptance; actual DSpark behavior is checked by the response hashes and panels. Commands, raw logs, gates and comparison: `task-logs/round10/quality-summary.json` and its referenced directories.

## Panels (completed)

Three seeds, card sampling (temperature0.6, top_p0.95, top_k20), thinking enabled, the exact n3/p0/cache4608 profile through the isolated gateway8791.

| Model / configuration | Hard | Agent | Scope |
|---|---:|---:|---|
| Ling DSpark n3/p0, cache4608, ctx8192 | **39/42** | **36/36** | Fresh, three runs each |
| Qwen3.8 IQ3_S | 56/56 | 69/72 | Historical control, read only |
| Bolt | 41/42 | 36/36 | Historical control, read only |

Ling's three misses are all `long-combine` HTTP400 context rejections: about20,005 input tokens against8192 configured. All39 in-context hard cases pass: code15/15, tools9/9, German9/9, format6/6. Agent: escalation12/12, memory9/9, media9/9, documentation6/6. These small panels do not establish universal quality equivalence between the models; the historical controls have different configurations and sample counts.

There were 137 HTTP chat requests, of which 135 passed the post-request foreign-load gate. Every request passed the pre-gate. Correctness outcomes of the 2 loaded requests are retained; their elapsed times are not clean performance measurements. The complete event list, seeds, input prefixes and hashes are in `panels/summary.json`, `runs.jsonl` and `responses.jsonl`. Panel elapsed times also include the added pre-request gate and must not be compared directly to ordinary user latency.

The first code-duration answer used7693 completion tokens, largely thinking, while still passing its hidden tests. This illustrates why the short decode benchmark and complete-task latency are different measurements. No tuning was made from individual panel answers.

Actual spawned argv/environment: `panels/actual-engine.json`; model-card entry: `model-entry.json`; panel command: `command.json`; raw selected history rows: `results.jsonl`; engine/bench logs are archived beside them. Historical raw control rows: `task-logs/round10/panel-controls.json`. The second gateway was unloaded and stopped, and its main-gateway hold released before Qwen profiling. The live settings and live benchmark history were not edited.

## Whole Qwen round profile (completed)

Eight independent server loads, one measured request each after the same 16-token warmup. All eight foreign-load gates pass; text SHA256, token counts and draft/accept counts match exactly across the four modes for each prompt. Code: 218 tokens, 180 drafts / 158 accepted, 60 rounds; English: 256 tokens, 253 drafts / 170 accepted, 85 rounds. Code SHA256 `6085cbab37d9cbea233e6bed234d4985c784feda97c27313fb9bc20c350633a7`; English `6b875f1abf21cc6ca2fbfde72d3906b1ca9f1676045ded0ce964c41fce8b1801`. This is diagnostic attribution from one request per mode, not a repeated benchmark claim.

| Mode | Code tok/s | English tok/s | Code ms/round | English ms/round |
|---|---:|---:|---:|---:|
| off | 37.090 | 32.394 | 97.513 | 92.644 |
| off-counters | 44.301 | 39.245 | 81.638 | 76.442 |
| on-counters | 43.932 | 39.202 | 82.371 | 76.525 |
| on-plain | 44.259 | 39.424 | 81.716 | 76.095 |

`off`: Nsight CPU/CUDA capture with graphs disabled and existing CPU interposer; `*-counters`: existing interposer plus level5 server log, no Nsight; `on-plain`: graphs allowed, level4 log, no interposer or profiler. Graphs enabled means permitted by the runtime, not proof that every mixed CPU/GPU subgraph is captured. Nsight increases round time by 19.4% / 21.2% versus the matching graph-off counter run. Counter/log instrumentation costs about 0.8% / 0.6% versus plain graph-on in these individual runs. The on/off counter difference is small and is not a demonstrated graph speedup.

The table below is the mean of 59 code / 83 English complete three-draft, four-token-verify rounds. First/last partial rounds are excluded, including the English final two-token verify. `n_batch (effective)=2048` is capacity, not verify width. Existing queue-yield log boundaries separate target verification from MTP catch-up; the next boundary separates sampling/acceptance. The wall columns partition each round. All GPU, copy, CPU and wait clocks are overlapping measurements and must not be added. Pairs are code / English, milliseconds.

| Phase | Wall, graphs allowed, no Nsight | Wall, Nsight graphs off | GPU dense/other, off | GPU cache hits, off | Copies, off | Host CUDA sync, off |
|---|---:|---:|---:|---:|---:|---:|
| draft_1_with_setup | 2.817 / 2.853 | 3.507 / 3.607 | 1.827 / 1.827 | 0.000 / 0.000 | 0.312 / 0.363 | 1.545 / 1.574 |
| draft_2 | 2.423 / 2.371 | 2.927 / 2.939 | 1.830 / 1.827 | 0.000 / 0.000 | 0.330 / 0.266 | 1.593 / 1.526 |
| draft_3 | 2.316 / 2.391 | 2.822 / 2.924 | 1.827 / 1.828 | 0.000 / 0.000 | 0.227 / 0.291 | 1.490 / 1.551 |
| draft_tail | 0.055 / 0.067 | 0.330 / 0.122 | 0.000 / 0.000 | 0.000 / 0.000 | 0.000 / 0.000 | 0.000 / 0.000 |
| verify_target | 70.813 / 65.538 | 83.378 / 78.761 | 27.632 / 27.724 | 5.015 / 6.127 | 8.940 / 8.856 | 13.367 / 13.438 |
| mtp_catchup | 1.961 / 1.983 | 2.539 / 2.720 | 0.618 / 0.618 | 0.271 / 0.268 | 0.202 / 0.202 | 0.190 / 0.181 |
| sampling_accept_host | 1.372 / 1.180 | 1.450 / 1.257 | 0.000 / 0.000 | 0.000 / 0.000 | 0.634 / 0.632 | 0.031 / 0.027 |
| emit_and_host | 0.063 / 0.068 | 0.164 / 0.121 | 0.000 / 0.000 | 0.000 / 0.000 | 0.016 / 0.018 | 0.000 / 0.000 |

CPU expert-graph elapsed time in the graph-on counter runs, per actual request round: draft one-token bucket **0.635 / 0.705 ms**; four-token verify plus MTP catch-up bucket **35.251 / 29.765 ms**; English final two-token window contributes another **0.194 ms** when averaged over all85 rounds. No-expert CPU graphs add0.186 /0.215 ms. The existing interposer cannot split four-token CPU time exactly between target misses and draft catch-up, nor individual draft steps; the enclosing catch-up wall interval is only about1.96 /1.98 ms. These are measured graph elapsed times, including internal dispatch/waits, not aggregate CPU-seconds or isolated dot-product time. They can overlap GPU work.

CPU sampling found30,428 code and40,903 English samples. libgomp accounts for63.34% /66.64% of aggregate sampled CPU activity; ggml-cpu accounts for33.01% /29.50%. The hottest libgomp address resolves to ELF offset0x256c2, the `add` instruction immediately after `pause` in a polling loop with a futex fallback. The corresponding disassembly and SHA256 are archived in `profile-code-duration-off/gomp-hotspot.*`. Other prominent samples are IQ4_NL, IQ3_XXS, IQ2_S and IQ3_S vector dot products. **CPU sample percentages are not wall-time savings.** Busy waiting across many workers explains high reported CPU utilization without equivalent useful arithmetic.

### Three largest actionable directions and ceilings

1. **CPU expert computation and thread synchronization:** eliminating every measured expert CPU graph would remove at most **35.89 / 30.66 ms per round** in the graph-on counter runs. Actual savings are smaller because CPU and GPU work overlap. First test the missing persistent draft pool and native worker placement; do not interpret the OpenMP sample share as a speed forecast. This measured envelope exceeds the3% threshold for a bounded experiment.
2. **GPU dense/output-head work:** the graph-off trace contains at most **33.73 / 33.74 ms per round** in dense-and-other kernels, including the three draft output-head steps. Reusing/fusing work or improving compatible low-batch kernels can only recover a fraction. This is a trace-domain zero-cost ceiling, not a predicted graph-on gain; the category also includes small non-expert kernels.
3. **Transfer scheduling and avoiding redundant copies:** eliminating all recorded copies would remove at most **11.03 / 10.78 ms per round** in the same trace. Overlap makes the critical-path gain smaller. Host-to-device copies move6.31 /8.68 GB over507 /697 ms of copy activity, about12.44 GB/s during transfers, while average whole-decode traffic is only about1.08 /1.10 GB/s. This points toward reducing or overlapping transfers, not a claim that extra motherboard lanes alone solve the round. Copy kinds are verified against installed `cupti_activity.h` (1 HtoD,2 DtoH,8 DtoD).

These ceilings overlap and are not additive. GPU occupancy or use of all4352 arithmetic lanes cannot be inferred from Nsight Systems utilization; no new instruction-level kernel claim is made. The exact GPU kernel list, intervals, API counts, CPU stacks and transfer totals are in the local captures and committed summaries.

Compared with the historical Strata figures51ms code /41ms English in `patches/llama.cpp/strata-parity/PLAN-DE.md`, the present plain Qwen runs take **81.72 /76.09 ms per round** (1.60x /1.86x). The gap remains in target verification. This is a historical comparison with different cache state/configuration and per-round output counts, not a fresh paired Strata benchmark. Strata was not run or modified.

All captures used the original runtime and no instrumentation build. The first code capture is rejected because raw token bytes caused UTF-8 log parsing to fail; it remains under `profile-code-duration-off-failed-utf8/`. Human-readable log parsing now uses replacement decoding, while response hashes still use the original valid streamed response. A reparented Nsight collector is included in owned CPU accounting only when its per-capture XDG_CONFIG_HOME matches exactly. The sampling period was raised to18,000,000 cycles to reduce collection cost; sample counts, not an assumed sampling frequency, are reported. Evidence: `profile-correction.json`, `profile-queue-results.json`, `profile-summary.json`, `profile_sql.py`, `profile_logs.py`, and all eight capture directories. Large raw captures remain local and are ignored by Git.
