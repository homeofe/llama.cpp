# Round 11: asynchronous cache uploads and Ling at 32K

Started 2026-10-18 11:23 UTC. Four-hour deadline: 15:24 UTC (17:24 Europe/Berlin). Branch `l5-cache`; original source/runtime from round 10. No live gateway changes, installation, push, system changes or other-worktree edits. Sudo profiling is authorized if needed. The handoff's foreign-process CPU gate overrides the generic load-average gate in the measurement skill.

## Predeclared measurement and adoption

1. One fresh eight-prompt pass per model and admission limit 32, 8, 0; prompt bursts remain enabled. Qwen uses `q-warm-draft`, Ling uses round-10 DSpark n3/p_min 0/cache 4608 MiB. Identical level-5 logging provides an exact speculative-round denominator. Final cache counters include the short warmup and prefill; uploaded MiB/round excludes initialization and subtracts prompt bursts. A changed hit rate or output is a confound, not a clean upload-only speedup.
2. Implement the separately ordered upload path only if the measured bound supports at least 3%. The switch defaults off. Adoption requires three interleaved rounds, at least 3% in both code and English categories, checked outputs and draft counts, cache-off bit-exactness, and test-de perplexity within +0.5%. Changed speculative outputs require additional quality evidence and prevent claiming a shape-identical speedup.
3. Ling 32K: keep the cache budget if it fits; otherwise reduce only enough to fit. Compare eight short prompts once with 8K and run the hard/agent panels through the isolated gateway on 8791.
4. Prepare exact gateway entries without applying them; the final binary path is left for Claude.

## Design under investigation

Use a second backend instance on each CUDA device for its independent nonblocking upload stream. Existing backend events order reuse after previous graph readers. Poll copy-completion events at graph boundaries without a host wait; only then publish new host/device map entries for later graphs. Evicted entries become misses for later graphs until the replacement is complete. Pending slots must be reserved against another eviction. The original stream-ordered path remains intact with the switch off. Context teardown must drain uploads before freeing their sources, destinations or events.

NVIDIA documents that asynchronous host transfers require pinned memory and that stream waits order later work on the waiting stream: [CUDA asynchronous execution](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html). Event queries test completion without a blocking host synchronization: [CUDA event API](https://docs.nvidia.com/cuda/cuda-runtime-api/cuda_runtime_api/group__CUDART__EVENT.html). The implementation will validate source-buffer support and exercise these dependencies with the existing deferred host backend before GPU measurements.

Results pending. A percentage gate is not a technical speed ceiling; cumulative improvements must be remeasured against the same functioning baseline.
