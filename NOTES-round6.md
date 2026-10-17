# Round 6: adaptive GPU-hit / CPU-miss cache

Worktree `/opt/llm-gateway/data/build/llama-l5`, branch `l5-cache`, base `004054e83`. Started 2026-10-17 14:18 UTC. The supplied round-5 GPU results are the starting point; no GPU execution or measurement was attempted here. No Git metadata, other worktrees, gateway configuration, Strata, or `/opt/agent-tools` was modified.

## Status

Adaptive LFU admission is implemented and disabled by default. CUDA compilation, CPU operator tests, C++ trace replay, and deferred-backend ordering checks pass. GPU correctness, overlap logging, throughput, perplexity, and the MTP server comparison remain for Claude. There is no new GPU performance claim.

The source diff and new test/documentation files are saved in `task-logs/round6/round6.patch`. No commit or PR was made.

## Design

- Enable with both `GGML_MOE_HYBRID=1` and `GGML_MOE_HYBRID_ADAPTIVE=1`, plus `GGML_MOE_HYBRID_MIB`. `GGML_MOE_HYBRID_ADMIT` is a nonnegative integer, default 8, limiting expert uploads across all cache groups/devices per ubatch. Zero collects statistics without runtime admissions. Invalid/overflowing values throw during initialization. Existing `--moe-cache-mib` remains mutually exclusive with hybrid mode.
- `llama_moe_cache_lfu`, declared in the existing internal cache header and implemented in the cache source, is the same policy object used by production and replay. Keys are `(layer, expert)`. Counts are doubles so halving retains fractional history, as in the Python reference. Counts halve before tokens 513, 1025, etc. Each selected route counts, including duplicate IDs and speculative tokens that are evaluated but later rejected.
- A small `MAP_CUSTOM1` node passes routing IDs through on the CPU branch. It records the IDs already copied for the CPU expert matmuls, handles strided IDs, and uses one CPU task. It does not read back from a device, synchronize a backend, install an eval callback, or mutate residency. The CPU backend finishes these nodes before scheduler submission returns.
- At the successful ubatch boundary, the cache processes routes in token/layer order, then chooses admissions. Candidates are nonresident experts observed in that ubatch, hottest first; residents are coldest first. Equal counts do not evict. Empty slots fill first. Tie order is deterministic by key. Candidates not admitted must be observed again to compete on a later ubatch. Groups receive the remaining global upload allowance in rotating order, avoiding permanent priority for the first layout/device.
- The token clock comes from the actual ubatch. If the final FFN has been cropped to output rows, the existing host output flags place those rows at the correct token indices before decay. Warmup is excluded. Failed/aborted evaluations discard their collected routes and do not upload.
- Layout/device grouping and memory budgeting reuse the existing cache. In adaptive mode a group must have at least `min(8*n_expert_used_max, n_expert)` slots: the maximum distinct working set of one layer's supported eight-token window. Smaller groups are disabled; if none fit, initialization fails. This is a per-layer window bound, not the sum of all layers' eight-token routes, which would exceed the requested 1805-slot budget for this model.
- A supplied frozen profile populates the same banks/maps at initialization and seeds LFU from its training counts. Without a profile, all maps start at -1 and all counts at zero. Adaptive graphs include eligible empty layers so later admissions do not require graph rebuilding. Existing LoRA, bias/scale, precision, layout, warmup and token-window gates remain. Ordinary fallback routes in eligible groups are counted as misses and can train admission for later supported windows.
- With adaptation off, observation returns the original ID tensor, boundary hooks do nothing, profile loading and static residency retain their previous behavior, and no extra graph nodes or runtime uploads are created. The host test checks frozen no-op behavior and equal initial bank/map contents between frozen and adaptive warm starts. GPU bitwise parity still needs the requested runtime check.
- The existing draft-profile mismatch guard remains. Adaptive no-profile contexts with no routed experts now also disable the cache with a warning, rather than trying to allocate a MoE cache for a dense/draft context.
- Teardown prints WARN-level evaluated tokens, hits, misses, hit rate, runtime uploads, total runtime MiB, and hits/misses/uploads/MiB per token. Runtime MiB includes weight slices and map updates. Warm-start uploads are logged separately and excluded from runtime admission rates. Statistics cover eligible cache groups; layers not assigned a cache are outside the denominator.

## Upload ordering and overlap

`process_ubatch()` begins collection immediately before `graph_compute()` and calls `end_graph()` after it returns successfully. At that point all graph operations have been submitted and CPU collectors/misses have completed. GPU tails may still be running.

Only the owning cache backend reads a group's banks in the masked GPU branch. Every new weight slice and both map changes (old expert -> -1, new expert -> slot) are enqueued with `ggml_backend_tensor_set_async` on that same backend, after graph submission. CUDA's setter uses `cuda_ctx->stream()`, as do graph launches; CUDA's concurrent worker streams join the main stream. Thus the device order is:

```
current graph's bank/map reads -> weight uploads -> map changes -> next graph's reads
```

No slot is overwritten while an earlier reader is executing, and the next lookup cannot see a new map entry before its weights. This requires neither a host wait nor an upload triggered synchronously by a miss. The ordering argument is also beside the upload loop in the source.

Weight upload sources are immutable model storage. Map upload sources are an immutable per-group array `[-1, 0, ..., n_slots-1]`; each map update copies one scalar from it. Unlike a mutable host map or a temporary fill object, these pointers remain valid through any number of in-flight evaluations. Context teardown already synchronizes backends before freeing cache storage. No staging-buffer reuse event or new synchronization is required.

The observation node is constructed only after the GPU hit branch has been expanded and is pinned to the CPU. It consumes the same selected IDs needed by the old CPU branch and adds no later GPU input producer. The existing miss-mask producer remains the last input producer for the CPU split; the scheduler itself is unchanged. This is a graph/dependency argument, not a new GPU overlap trace. Claude should verify `GGML_SCHED_DEBUG=2` still reports part A ending at `ffn_moe_miss_mask-N` with adaptation enabled.

The host test uses the actual cache, a generated two-layer MoE model, a real CPU collector, and a backend whose async writes are queued with their original source pointers. It queues a reader of the old residency, forces cross-layer evictions without draining the stream, and queues a reader of the new residency. Both observe the expected maps and all expert weight bytes when the queue is drained. It asserts no backend synchronization was added, checks strided IDs, excludes aborted evaluations, tests warm starts and frozen mode, rejects an undersized budget, and tests cropped output rows across the 512-token decay boundary. It does not emulate CUDA kernels or validate CUDA's stream implementation.

## Replay

Trace: `/opt/llm-gateway/data/build/llama-codex2/task-logs/round4-hitmiss/qwen-routes-r1.routes.jsonl`. Topic order is `de, en, code, agent`; 4096 evaluated tokens and 1,966,080 routes. One group, empty start, no device. Python was run directly from `/opt/llm-gateway/data/moe-quant/cache/sim_lfu.py`; exact results were also collected through its existing `lfu()` function.

| Policy | Slots | Tokens/ubatch | Admission cap | Hit rate | Uploads/token |
| --- | ---: | ---: | ---: | ---: | ---: |
| Python immediate LFU | 1805 | 1 | unlimited | 44.1656% | 5.3113 |
| C++ deferred LFU | 1805 | 1 | unlimited | 44.1848% | 5.1196 |
| C++ default | 1805 | 1 | 8 | 43.2629% | 4.2390 |
| Python immediate LFU | 3000 | 1 | unlimited | 56.0845% | 7.7791 |
| C++ deferred LFU | 3000 | 1 | unlimited | 56.1058% | 7.5525 |
| C++ default | 3000 | 1 | 8 | 54.3998% | 5.9399 |
| C++ deferred LFU | 1805 | 4 | unlimited | 43.8102% | 4.6738 |
| C++ default | 1805 | 4 | 8 | 38.9340% | 1.9839 |
| C++ cap ablation | 1805 | 4 | 32 | 42.9538% | 3.9067 |

Unlimited C++ admission matches Python within 0.022 percentage points. Python replaces entries immediately as each route arrives; C++ keeps residency fixed during the ubatch, then chooses the hottest candidates. This changes within-ubatch churn and reduces uploads slightly.

At 1805 slots the default cap loses 0.903 points versus Python. At 3000 slots it loses 1.685 points: even one-token ubatches are throttled during cold fill and bursts, so the larger cache takes longer to populate and uploads fall from 7.779 to 5.940/token. This is throttling, not a group-boundary discrepancy. Four-token ubatches with cap 8 permit only two uploads/token; the cold-start result falls 5.232 points below Python's one-token unthrottled reference. Unlimited four-token admission reaches 43.810%, isolating a 0.355-point loss from deferred window updates. Cap 32 recovers to 42.954%, still 1.212 points below the Python reference: 0.355 points from the window and another 0.856 from burst throttling. A warm profile should avoid much of the initial fill cost, but this has not been measured on the gateway workload.

Raw results are `task-logs/round6/replay-*.log` and `sim-lfu-exact.log`. Reproduce, for example:

```sh
build-cpu/bin/test-moe-cache /opt/llm-gateway/data/build/llama-codex2/task-logs/round4-hitmiss/qwen-routes-r1.routes.jsonl 1805 8 1
```

Arguments after the trace are slots, admissions/ubatch, and tokens/ubatch. Use 100000 for the unlimited replay rows.

## Validation

| Check | Result | Log under `task-logs/round6/` |
| --- | --- | --- |
| CPU policy CTest | Pass | `ctest.log` |
| Deferred backend, strided collector, abort, budget, warm/frozen and cropped-row checks | Pass | `ordering.log` |
| CPU no-expert context, adaptive enabled without a profile | Cache disabled with WARN; context created | `no-experts-context.log` |
| CPU context with a profile naming another model's layers | Cache disabled with WARN; context created | `profile-mismatch-context.log` |
| CPU `MUL_MAT_ID_MASKED` | 1314/1314 | `masked-cpu.out` |
| CPU ordinary `MUL_MAT_ID` regression subset | 122/122 | `unmasked-cpu.out` |
| Full CUDA build, Release, architecture 75 | Pass; no runtime execution | `build-cuda.log` |
| Diff whitespace, changed-source ASCII, and local EditorConfig rules | Pass | `checks.log` |

The standalone `editorconfig-checker` executable is unavailable; LF endings, UTF-8/ASCII, final newlines, spaces instead of tabs, and trailing whitespace were checked locally on the changed/new source files. Indent-size enforcement is disabled by the repository's `.ecrc`. No model naming mappings changed.

`bash task-logs/round6/check-host.sh` rebuilds the relevant CPU targets, generates the tiny MoE fixture, runs the host tests/replays/operator checks, and builds the CUDA tree. It does not run a CUDA binary. The new test is excluded from system-ggml, dynamically loaded CPU backend, and Windows shared-library configurations because its deferred backend uses internal interfaces/direct CPU symbols.

## GPU handoff and open risks

`task-logs/round6/adaptive.env` contains the requested default switches and overlap/CPU-affinity settings. Source it, choose the actual cache budget beside the MTP head, and either set `GGML_MOE_HYBRID_PROFILE` to a profile that fits that budget or unset it for cold start. Do not reuse the 4096 MiB profile unchanged if its residents exceed the smaller budget.

- Baseline: override `GGML_MOE_HYBRID=0`.
- Adaptive: `GGML_MOE_HYBRID=1 GGML_MOE_HYBRID_ADAPTIVE=1 GGML_MOE_HYBRID_ADMIT=8`.
- Frozen comparison: `GGML_MOE_HYBRID=1 GGML_MOE_HYBRID_ADAPTIVE=0` with a profile.
- Optional pp4 throttle comparison: adaptive with `GGML_MOE_HYBRID_ADMIT=32`.

Use the planned pp4/pp1 off/on/on/off sequence, test-de perplexity at `-ub 4` and `-ub 1`, and the MTP-3 `stage_b.py` comparison against Strata. Capture WARN teardown summaries and an untimed `GGML_SCHED_DEBUG=2` run for overlap. Routing eval callbacks should remain off during timing because they disable scheduler overlap.

The main remaining risk is GPU runtime correctness/order under real CUDA graph reuse and multiple streams/devices; the host test proves the enqueue protocol only. PCIe uploads consume bandwidth and may delay output copies because they are submitted before the context extracts outputs. Pageable model-memory staging and many small map writes can add host overhead even without an explicit synchronization. The default eight-upload budget can also limit adaptation during multi-token prefill/verification; the replay quantifies this but does not predict wall-clock speed. Ordinary fallback windows still train the policy while reporting actual cache hits as zero. Mixed layout groups share one cap in rotating order, which is fair over time but is not a globally optimized allocation of upload bandwidth. Warm-profile count magnitudes influence how quickly old residents age out. Non-CUDA adaptive runtime behavior has not been tested.
