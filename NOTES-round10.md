# Round 10: Ling confirmation, panels and whole-round profiling

Started 2026-10-18 07:00 UTC; deadline 11:00 UTC. Branch `l5-cache`. No system changes, live gateway changes, pushes or PRs. The owner subsequently authorized CPU sampling and sudo explicitly; use those for task 3 without changing kernel settings.

## Measurement protocol

The unchanged stage-B foreign-process gate is reused. A rejected request invalidates its whole prompt pass; restart the server before retrying, so the adaptive cache always sees the same warmup and eight-prompt history. Screening: MTP3 control and seven DSpark settings, four code and four English prompts per setting. Confirmation: best two candidates and control, three rotating rounds. Raw output hashes, draft counts, accepted lengths, CPU cost and exact configurations are archived.

## Initial findings (screening, not confirmation)

| Configuration | Code tok/s | English tok/s | Acceptance code / English |
|---|---:|---:|---:|
| ling-cache-mtp3 | 50.7452 | 47.4517 | 67.88% / 53.02% |
| ling-dspark-n7-p00 | 77.3175 | 41.7307 | 78.28% / 32.87% |
| ling-dspark-n3-p00 | 66.3210 | 51.6762 | 95.84% / 57.64% |
| ling-dspark-n4-p00 | 69.8386 | 51.6521 | 86.64% / 48.45% |
| ling-dspark-n5-p00 | 73.9066 | 49.5328 | 84.69% / 43.17% |
| ling-dspark-n7-p30 | 76.6976 | 45.4170 | 79.02% / 43.26% |
| ling-dspark-n7-p50 | 76.8802 | 47.1952 | 89.57% / 61.11% |

Four prompts per category, one complete clean pass per configuration. Foreign-load failures discard the entire pass before retrying with a fresh server/cache. Selection is pending the remaining screen and three interleaved confirmation rounds.

The reported speed is the unchanged stage-B client first-to-last streamed-token metric. Raw server timing is also retained; a batched final token/stop event can produce a boundary difference, especially on short answers. Output hashes are recorded in `screen-summary.json`; the prompt texts and their SHA256 hashes are in `task-logs/round10/prompts.json`.

## Panel protocol

Historical controls were read, not modified: Qwen iq3_s hard 56/56 and agent 69/72; Bolt hard 41/42 and agent 36/36. These are different models and older runs, not paired runtime controls. The unchanged panel executable will run with three seeds, card sampling and thinking enabled, through port 8791. A local proxy adds the foreign-process gate and archives complete request/response hashes. The chosen speed profile has an 8192-token context; the approximately 20000-token long-document case exceeds that limit and must be disclosed separately, without silently changing the tested allocation.

Ling card provenance: inclusionAI/Ling-3.0-flash README, revision `fcc60a52b34c69c6271d0ac8a8cb0815020b6a43`, downloaded read-only via hf CLI into `data/moe-quant/round10-metadata/target`. It recommends temperature 0.6, top_p 0.95, top_k 20 and thinking enabled.

## Profiling preparation

The ordinary llama-l5 backend has no GGML_SCHED_TRACE instrumentation. Its DEBUG_TIMINGS server scope is not compiled. No instrumentation build is made. The existing round-9 CPU graph interposer can provide measured graph totals, while Nsight records CPU stacks, CUDA APIs, kernels, transfers and request NVTX ranges. CPU sampling initially failed under perf_event_paranoid=4; after the owner explicitly authorized sudo, the unchanged kernel setting passes Nsight process-tree sampling checks. Use graph-disabled traces for individual GPU activity and separate graph-enabled wall measurements.
