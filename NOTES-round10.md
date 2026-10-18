# Round 10: Ling confirmation, panels and whole-round profiling

Started 2026-10-18 07:00 UTC; deadline 11:00 UTC. Branch `l5-cache`. No system changes, live gateway changes, pushes or PRs. The owner subsequently authorized CPU sampling and sudo explicitly; use those for task 3 without changing kernel settings.

## Measurement protocol

The unchanged stage-B foreign-process gate is reused. A rejected request invalidates its whole prompt pass; restart the server before retrying, so the adaptive cache always sees the same warmup and eight-prompt history. Screening: MTP3 control and seven DSpark settings, four code and four English prompts per setting. Confirmation: best two candidates and control, three rotating rounds. Raw output hashes, draft counts, accepted lengths, CPU cost and exact configurations are archived.

## Initial findings (screening, not confirmation)

MTP3 reproduces the previous control: code 50.7452, English 47.4517 tok/s. DSpark7 with 4608 MiB gives 77.3186 / 41.73 (rounded English), repeating the code-only gain. DSpark3 gives 66.32 / 51.68 in its first pass and is the first candidate improving both categories. Final selection waits for every screened setting and the three-round confirmation.

## Panel protocol

Historical controls were read, not modified: Qwen iq3_s hard 56/56 and agent 69/72; Bolt hard 41/42 and agent 36/36. These are different models and older runs, not paired runtime controls. The unchanged panel executable will run with three seeds, card sampling and thinking enabled, through port 8791. A local proxy adds the foreign-process gate and archives complete request/response hashes. The chosen speed profile has an 8192-token context; the approximately 20000-token long-document case exceeds that limit and must be disclosed separately, without silently changing the tested allocation.

Ling card provenance: inclusionAI/Ling-3.0-flash README, revision `fcc60a52b34c69c6271d0ac8a8cb0815020b6a43`, downloaded read-only via hf CLI into `data/moe-quant/round10-metadata/target`. It recommends temperature 0.6, top_p 0.95, top_k 20 and thinking enabled.

## Profiling preparation

The ordinary llama-l5 backend has no GGML_SCHED_TRACE instrumentation. Its DEBUG_TIMINGS server scope is not compiled. No instrumentation build is made. The existing round-9 CPU graph interposer can provide measured graph totals, while Nsight records CPU stacks, CUDA APIs, kernels, transfers and request NVTX ranges. CPU sampling initially failed under perf_event_paranoid=4; after the owner explicitly authorized sudo, the unchanged kernel setting passes Nsight process-tree sampling checks. Use graph-disabled traces for individual GPU activity and separate graph-enabled wall measurements.
