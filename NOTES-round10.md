# Round 10: Ling confirmation, panels and whole-round profiling

Started 2026-10-18 07:00 UTC; deadline 11:00 UTC. Branch `l5-cache`. No system changes, live gateway changes, pushes or PRs. The owner subsequently authorized CPU sampling and sudo explicitly; use those for task 3 without changing kernel settings.

## Measurement protocol

The unchanged stage-B foreign-process gate is reused. A rejected request invalidates its whole prompt pass; restart the server before retrying, so the adaptive cache always sees the same warmup and eight-prompt history. Screening: MTP3 control and seven DSpark settings, four code and four English prompts per setting. Confirmation: best two candidates and control, three rotating rounds. Raw output hashes, draft counts, accepted lengths, CPU cost and exact configurations are archived.

## Screening (completed; confirmation running)

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
