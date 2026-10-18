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

## Upload screening result

48 accepted requests, two contaminated Ling/admit8 passes discarded and repeated in full. Completed 11:54 UTC. Four prompts per category, one accepted pass per configuration. Cache hit rates below include prefill and the 16-token warmup; MiB/round subtracts prompt bursts and excludes model-load warm starts. Small negative residuals from two-decimal log rounding are clamped to zero.

| Model / admit | Code tok/s | English tok/s | Target / draft hit rate | Decode upload MiB/round |
|---|---:|---:|---:|---:|
| qwen-admit32 | 47.255 | 40.196 | 35.70% / 63.29% | 78.246 |
| qwen-admit8 | 47.899 | 40.700 | 30.08% / 63.15% | 33.628 |
| qwen-admit0 | 46.584 | 39.419 | 22.41% / 42.01% | 0.000 |
| ling-admit32 | 66.266 | 51.543 | 43.81% / n/a | 94.366 |
| ling-admit8 | 68.291 | 55.626 | 37.47% / n/a | 23.991 |
| ling-admit0 | 67.491 | 53.989 | 27.00% / n/a | 0.000 |

**The equal-hit-rate premise does not hold.** Disabling decode admission lowers target hit rate by 13.29 percentage points for Qwen and 16.81 points for Ling. Net speed is therefore not an upload-only ceiling. Qwen admit8 gains 1.36%/1.25%, admit0 loses 1.42%/1.93%. Ling admit8 gains 3.06%/7.92%, admit0 gains 1.85%/4.75%. Output hashes also change (Qwen admit8 4/8 same, admit0 3/8; Ling admit8 3/8). These screens do not qualify a policy for adoption.

The stop condition of less than 3% at equal hit rate is not established. The measured round-10 envelope of roughly 9ms in verification transfers and the Ling screen justify a bounded two-stream implementation. This is an upper-envelope argument, not a forecast of the achievable gain. Exact hashes, counts, medians/ranges, raw logs and the decision are in `task-logs/round11/bound*`.

The read-only CPU diagnostic observed compaction/reclaim during model loading. It did not capture either rejected request, so it does not prove the cause of those failures. The third admit8 pass met the unchanged gate. No kernel setting was changed.


## Two-stream protocol and host proof

The switch is `GGML_MOE_HYBRID_ASYNC_UPLOAD=1`, default off. Each supported device owns a separate upload backend/stream, a graph-reader event and an upload-completion event. CUDA exposes only a nonblocking event-query extension through its existing function registry. HIP, MUSA and CUDA builds without backend events retain the original path. Sources must use the device's pinned host buffer type; otherwise the feature logs a fallback.

At the end of every graph, including an aborted graph, the compute stream records a reader fence after CUDA has joined its worker streams. The upload stream waits for the recorded fence before overwriting selected slots. Evicted map entries are invalidated in compute-stream order, so later graphs use CPU misses while uploads are pending. LFU reserves the planned slots; no second admission plan runs while a device still has uploads pending. At the beginning of a later graph, a successful copy-event query permits publishing the new host/device map entries. Device-map writes remain on the compute stream after its previous graphs. Re-recording the reader event cannot change an upload stream wait that already captured an older record. Deferred prompt bursts are retained until the pending batch can finish.

No event/stream synchronization is added to inference admission or publication. The copy stream is synchronized only during cache teardown, before its destination banks and pinned model sources can disappear. Teardown does not dereference the context's compute-backend pointer because that backend may already have been destroyed. The default stream-ordered path is preserved.

The deferred host fixture proves old readers see old weights, new readers see new weights, partial copies remain invisible, pending slots cannot be admitted again, event generations remain independent, aborts record reader fences and teardown drains an in-flight copy. All checks are deterministic queue operations without timing sleeps.

- Current implementation: exit 0. Original library: exit 86 (the new upload contract is absent).
- Publish before copy completion: exit 86, expected map-visibility assertion.
- Reuse before old readers complete: exit 86, expected blocked-copy assertion.
- Tiny map-publication cut (wrong scalar): exit 86, expected slot assertion.
- Restored implementation: exit 0. Existing single-stream ordering and LFU/prefill tests also pass.
- ASan/UBSan: changed cache implementation plus the test executable instrumented, exit 0; the existing base CPU/model libraries are not sanitizer-instrumented.

The first fixture attempt correctly refused its mmap-backed source; the fixture now loads owned host storage with the fork's `load_mode` API. Expected assertion failures use a test-only abort callback with exit 86, avoiding the operating system's core collector. Evidence: `task-logs/round11/host-verdict.json`, `host-complete.json` and the matching raw logs.
