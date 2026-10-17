# Round 7: prompt warming, static base and independent draft cache

Worktree `/opt/llm-gateway/data/build/llama-l5`, branch `l5-cache`, base `8c6362243`. This continues the supplied round-6 results and runtime fixes. No GPU execution, network access, commits, pushes, Git metadata changes or other-worktree changes were made. Claude owns GPU validation and commits.

## Result

All four requested pieces are implemented and opt-in. The replay favors prompt warming over a 25% static base on these workloads. On eight short prompts with decay 512, prompt warming raises decode hits from 33.55% to 43.38% and first-32-token hits from 22.29% to 44.46%, but runtime uploads increase from 3.886 to 8.718 experts per evaluated token. Decay 64 raises decode hits to 49.01% while saturating the 32-upload allowance on every four-token decode window. These are routing-policy results, not throughput measurements.

CPU tests, host ordering tests, CPU trace/logit parity, the replay matrix, and the full CUDA build pass. GPU stream behavior, TTFT, throughput, perplexity and draft acceptance remain unmeasured. The patch is `task-logs/round7/round7.patch`; it includes source, this note and the reproducibility scripts. Generated traces, profiles, models and raw logs remain under `task-logs/round7/` and are not embedded in the patch.

## Switches and default behavior

| Switch | Default | Effect |
| --- | --- | --- |
| `GGML_MOE_HYBRID_PREFILL` | `0` | Enable prompt warming in adaptive contexts. |
| `GGML_MOE_HYBRID_ADMIT_PREFILL` | `1024` | Maximum expert uploads in a prompt burst. Requires `PREFILL=1`; setting the cap alone does not enable warming. |
| `GGML_MOE_HYBRID_STATIC_PCT` | `0` | Pin this integer percentage of each target cache group's slots from `GGML_MOE_HYBRID_PROFILE`. Requires adaptive mode and a profile. |
| `GGML_MOE_HYBRID_DECAY` | `512` | Positive evaluated-token interval for halving LFU counts, independently in each context. |
| `GGML_MOE_HYBRID_TRACE` | unset | Append JSONL routes. Also works with the cache/master switch off, including a draft with zero cache budget. |
| `GGML_MOE_HYBRID_DRAFT_MIB` | `0` | Independent draft budget in MiB. Positive values enable adaptive draft caching under `GGML_MOE_HYBRID=1`. |
| `GGML_MOE_HYBRID_DRAFT_PROFILE` | unset | Draft-only profile; without it, start the draft cache empty. |
| `GGML_MOE_HYBRID_DRAFT_STATIC_PCT` | `0` | Optional draft-only pinned fraction. Does not inherit the target's fraction. |

Existing `GGML_MOE_HYBRID`, `GGML_MOE_HYBRID_MIB`, `GGML_MOE_HYBRID_ADAPTIVE`, `GGML_MOE_HYBRID_ADMIT` and the profile format remain. Draft and target share the admission/decay/prefill configuration values, but not budgets, storage, counters or token clocks. Invalid integers, percentages above 100 and zero decay are rejected. A prefill cap of zero suppresses prefill admissions when warming is enabled; disabling `PREFILL` restores ordinary round-6 admission during prefill.

With the new switches unset, target graph construction, candidate ordering, decay and admissions retain round-6 semantics. All seven original replay rows match the published round-6 hit rates and upload rates exactly at four decimal places. Frozen/adaptive initial maps and weights still match in the host test. The original adaptive statistics log format is preserved; new-feature context labels use separate lines. Actual GPU bitwise parity is for Claude to check. Draft contexts explicitly default to no cache; they no longer accidentally inherit a target adaptive cold-start budget.

## Replay method

Source: `/opt/llm-gateway/data/build/llama-codex2/task-logs/round4-hitmiss/qwen-routes-r1.routes.jsonl`. Topic order: German, English, code, agent. The old positional replay remains available and was rerun unchanged at 1805/3000 slots, admission 8/unlimited, and one/four-token windows.

`make_replay.py` preserves every route in the long workload: four 1024-token topic segments, 4096 tokens and 1,966,080 routes. It marks the first 40 tokens of each topic as a single prefill ubatch, then uses four-token decode windows. The short workload is eight segments of 40 prefill plus 256 decode tokens, cycling the four topics twice: source offsets 0..295 on the first cycle and 296..591 on the second. It contains 2368 evaluated tokens and 1,136,640 routes. Cache state persists across topic changes.

The static profile uses tokens 592..1023 from all four topics. It is disjoint from the short workload, but overlaps the long workload: the long static rows are an optimistic, in-sample comparison. These are synthetic prompt boundaries on existing routes, not routes measured by actually re-prefilling those text segments. They cannot establish performance or routing transfer on the real eight-prompt server benchmark.

All matrix rows use 1501 slots and normal admission 32. `prompt` admits up to 1024 at the prompt boundary; `static` pins 25% (375 slots) and leaves the rest initially empty; `combined` enables both. All policies use the production `llama_moe_cache_lfu`. Counts and hits include evaluated speculative tokens, even if later rejected. First-32 metrics count evaluated decode tokens, not accepted output tokens.

All-route hit rates include prefill misses; decode and first-32 rates exclude prefill. Uploads/token and MiB/token use all evaluated tokens, include runtime prompt bursts and map writes, and exclude initialization. Synthetic weight slices are estimated at 2,050,000 bytes per expert, matching the supplied approximately 2.05 MB; runtime traces record actual slice bytes. Static rows additionally upload 375 experts / 733.137 MiB of weights at initialization. Raw logs contain hit counts, miss counts, uploads, initialization bytes and individual first-32-token results for every prompt.
### Long workload

| Policy | Decay | All-route hits | Decode hits | First 32 hits | Uploads/token | MiB/token |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| lfu | 512 | 38.47% | 40.03% | 21.37% | 3.265 | 6.383 |
| prompt | 512 | 41.77% | 43.47% | 45.44% | 4.801 | 9.387 |
| static | 512 | 37.13% | 38.64% | 24.84% | 2.815 | 5.503 |
| combined | 512 | 39.75% | 41.37% | 45.07% | 4.291 | 8.390 |
| lfu | 128 | 45.19% | 47.02% | 24.76% | 7.254 | 14.181 |
| prompt | 128 | 46.79% | 48.69% | 45.64% | 8.224 | 16.079 |
| static | 128 | 42.90% | 44.64% | 28.31% | 6.729 | 13.155 |
| combined | 128 | 44.02% | 45.81% | 45.26% | 7.657 | 14.970 |
| lfu | 64 | 45.91% | 47.77% | 24.80% | 7.714 | 15.081 |
| prompt | 64 | 47.48% | 49.41% | 46.03% | 8.679 | 16.968 |
| static | 64 | 43.77% | 45.55% | 28.51% | 7.688 | 15.031 |
| combined | 64 | 44.83% | 46.65% | 45.57% | 8.598 | 16.809 |

### Short workload

| Policy | Decay | All-route hits | Decode hits | First 32 hits | Uploads/token | MiB/token |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| lfu | 512 | 29.02% | 33.55% | 22.29% | 3.886 | 7.596 |
| prompt | 512 | 37.52% | 43.38% | 44.46% | 8.718 | 17.044 |
| static | 512 | 29.09% | 33.63% | 24.49% | 3.474 | 6.792 |
| combined | 512 | 36.00% | 41.62% | 43.23% | 8.327 | 16.280 |
| lfu | 128 | 35.42% | 40.96% | 22.69% | 6.867 | 13.424 |
| prompt | 128 | 41.02% | 47.43% | 44.50% | 10.275 | 20.089 |
| static | 128 | 35.17% | 40.67% | 26.60% | 6.639 | 12.980 |
| combined | 128 | 39.20% | 45.32% | 43.68% | 10.029 | 19.607 |
| lfu | 64 | 36.73% | 42.47% | 23.17% | 7.027 | 13.738 |
| prompt | 64 | 42.39% | 49.01% | 45.34% | 10.378 | 20.290 |
| static | 64 | 36.60% | 42.31% | 26.87% | 7.025 | 13.734 |
| combined | 64 | 40.43% | 46.74% | 44.23% | 10.341 | 20.217 |

The static base has little total-hit benefit on the short workload and reduces the prompt-warming benefit. It is available for calibration, but this profile does not support making it a default. Faster decay is a bandwidth tradeoff: the short prompt/64 row uploads 24,576 experts, consisting of all eight 1024-expert bursts and the maximum 32 uploads after every decode window. A 1024-expert target burst has an ideal transfer floor of approximately 175 ms at 12 GB/s, before host overhead and contention. A higher replay hit rate is not sufficient evidence of lower end-to-end latency.

## Prompt burst and ordering

The existing thread-zero CPU collector still passes through the IDs already needed by the host expert branch. It collects large ordinary-path prefill routes, strided IDs, cropped final-layer rows and evaluated speculative routes. It does not mutate device residency. Prompt counts are separate from the decaying LFU history and accumulate across prefill ubatches. At the boundary, prompt-only counts rank both nonresident candidates and possible victims. Empty slots fill first; equal scores retain residents; ties use expert keys. Old LFU history cannot prevent a new prompt's hotter experts from being warmed. Normal LFU resumes afterward with its original history, so subsequent re-admission can still cause churn.

The decoder passes logical-batch information into the cache hook. A logical batch above eight tokens is classified as prefill, including its small final ubatch. Consecutive prefill calls without outputs retain the prompt histogram. The final ubatch of a logical batch requesting outputs ends the prompt and submits the burst. Aborts clear pending prompt counts; a new position-zero prompt also resets them. This recognizes the serial server prompt path without a new request API. It is a context-wide aggregate heuristic: mixed requests, large verification batches, all-logit intermediate chunks, cancelled prefix-reused requests and standalone prompts of at most eight tokens are not reliably distinguished as semantic request boundaries. See the risks below.

The burst uses exactly the normal admission upload loop, with a different allowance and ranking source. Each layout/device group gets the remaining global allowance in the existing rotating order. There is no new transfer stream, staging pool, event protocol or synchronous upload on a miss:

```
current graph bank/map reads -> new expert weight slices -> old/new map updates -> next graph bank/map reads
```

After successful graph submission, CPU collectors have finished, while GPU tails can still be in flight. Every overwrite and map update is queued on the owning backend stream after those reads; CUDA joins worker streams before returning. New map entries follow all weight slices. Sources are immutable model storage and the existing immutable per-group scalar array `[-1, 0, ..., n_slots-1]`. They remain valid until context teardown, which already synchronizes backends. A larger burst changes the number of ordered writes, not this protocol.

The WARN burst line reports context ID, uploads, MiB and host enqueue/planning milliseconds. It deliberately does not report that as device completion time. Timing GPU completion would need an additional event/timing mechanism or a synchronization; neither was added. Claude should measure the actual TTFT delta. Pageable source staging can block a host API call despite the absence of an explicit backend wait.

The deferred host backend retains original source pointers, queues old readers, performs a multi-ubatch prefill burst, queues new readers, then drains. It verifies every weight slice and map, no extra synchronize calls, the small final tail, prompt-only ranking and abort reset. It proves the enqueue protocol, not CUDA stream execution.

## Static base

With a positive static percentage, the profile may exceed the cache budget. Entries are ranked by descending count and then `(layer, expert)`; the hottest `floor(group_slots * percentage / 100)` entries per group are loaded and pinned. The remaining slots start empty and use LFU. A group with insufficient profile coverage fails initialization, and duplicate entries are rejected. Entries for dropped adaptive groups retain round-6 skip behavior. With percentage zero, the previous profile-in-file-order warm start and over-budget rejection remain.

Pinned slots are absent from both ordinary LFU and prompt-burst victim lists. The host tests cover an unsorted profile larger than the budget, deterministic ties, missing/duplicate profile entries, pinned survival under a burst, and use of the unpinned remainder.

## Draft plumbing

`common_base_params_to_speculative()` still clears the inherited legacy `moe_cache_size` and now marks `common_params::moe_cache_draft`. Conversion carries that flag into the experimental `llama_context_params::moe_cache_draft`, default false. Both server initialization and fit-parameter estimation already use this conversion path.

A marked draft uses only `DRAFT_MIB` and `DRAFT_PROFILE`; zero budget bypasses target environment parsing entirely. A positive budget selects adaptive mode even if the target is frozen. A supplied draft profile seeds its cache; without it, maps and counts start empty. No process environment is modified. A mismatched profile still disables the affected context; the existing unmarked-context mismatch guard remains tested.

MTP can share the target's entire model object. For an MTP draft, allocation is restricted to `[hparams.n_layer(), model.layers.size())`, so its budget is spent on the head rather than the trunk. Other draft architectures use their own full model. The target's existing allocation range is unchanged. Banks, maps, scalar upload sources, LFU state and backend streams belong to each context; shared model weights are only read. The host test keeps target and draft caches alive together, checks the head-only range, and verifies draft writes leave target maps and weights unchanged.

There is no automatic VRAM stealing or budget fitting between the two caches. The existing minimum group size, `min(8 * n_expert_used_max, n_expert)`, still applies. Draft experts may be substantially larger than target experts, so a small MiB reservation can fail this bound. Inspect actual draft slice sizes and allocation logs before choosing a budget.

## Route trace and replay interface

`GGML_MOE_HYBRID_TRACE=/path/routes.jsonl` appends one version-1 JSON record per successful non-warmup ubatch. Each record includes a context ID, per-context ubatch number, draft/trace-only flags, ubatch size, prefill/end flags, primary token positions, sequence IDs, and layer records. Layer records contain group ID, expert count, actual bytes per expert, whether the hybrid branch was used, evaluated row indices and selected ID arrays. Cropped rows retain their original ubatch indices. Failed/aborted graphs are excluded, as in the admission policy.

A zero-budget trace-only object needs no GPU backend or cache allocation, so the draft can be traced while `DRAFT_MIB=0`, or the whole workload can be traced with the master hybrid switch off. Only host-resident expert layers are collected; GPU-resident expert layers are not read back. File writes are serialized across contexts in one process and flushed through the C++ stream, with no `fsync`, device readback, eval callback or backend synchronization. Formatting/file I/O still has host cost: capture traces separately from performance runs and use a separate file per process.

Replay a runtime log with:

```sh
build-cpu/bin/test-moe-cache --replay-log routes.jsonl 1501 32 1024 0 512 - CONTEXT_ID
```

Arguments are file, slots, normal cap, prefill cap (`0` disables warming), static percentage, decay, optional profile (`-` means none), and optional context ID. Omitting the context ID replays each context independently with the same requested settings. Select a context when target and draft need different slot budgets/profiles. For a static run, replace `-` with the appropriate profile and set the percentage. First-32 counters are per sequence, and shared evaluated rows are counted once.

The replay currently supports one cache group per context, matching the supplied homogeneous target workload; it explicitly rejects multiple groups rather than pretending they share banks. Dropped groups in a cache-enabled log are excluded. A trace-only log has no actual GPU hits, so its replay estimates supported windows from each layer's evaluated row count (at most eight); backend/LoRA/bias/precision eligibility must be validated on the GPU. It starts LFU empty unless a static base is requested, so it does not reconstruct a prior partially warmed cache or an ordinary profile warm start.

Reproduce the supplied-data matrix with `python3 task-logs/round7/make_replay.py SOURCE_TRACE` followed by `bash task-logs/round7/replay.sh`. `bash task-logs/round7/replay-legacy.sh` reproduces the seven original replay rows. `bash task-logs/round7/check-host.sh` builds the CPU targets, regenerates a tiny MoE fixture, runs all host/CPU checks and replays, then builds CUDA without running a CUDA executable. It expects the existing configured CPU-only and architecture-75 CUDA build trees.

## Validation

| Check | Result | Log in `task-logs/round7/` |
| --- | --- | --- |
| Complete host reproduction script | Pass | `check-host.log` |
| LFU, prompt ranking, pinning, decay and environment validation | Pass | `ctest.log` |
| Four-thread collector, strided IDs, cropped rows, abort and deferred burst ordering | Pass | `ordering.log` |
| Static ranking, invalid profiles and simultaneous target/draft storage | Pass | `ordering.log` |
| CPU baseline vs trace-only logits | Bitwise equal | `contexts.log` |
| Logical prefill split 16+4, next decode, draft conversion, zero budget and mismatch guards | Pass; positive cold draft correctly fails without a GPU | `contexts.log` |
| Runtime JSONL host/CPU log replay | Pass | `replay-host-trace.log`, `replay-context-trace.log` |
| Four policies x two workloads x three decay intervals | Pass | `replay-*.log` |
| Seven original round-6 replay rows | Exact published-metric matches | `legacy-*.log`, `legacy-parity.log` |
| CPU `MUL_MAT_ID_MASKED` | 1314/1314 | `masked-cpu.out` |
| CPU ordinary `MUL_MAT_ID` regression subset | 122/122 | `unmasked-cpu.out` |
| Full CUDA Release build, architecture 75 | Pass; compile only | `build-cuda.log` |
| Whitespace, changed-line ASCII, local EditorConfig rules, naming and script syntax | Pass | `checks.log` |

`editorconfig-checker` and Python `ty` are unavailable in this sandbox. Local format checks, Python compilation and execution of the generator were used; the repository's full Python type checker was not run. No new kernels or backend operator implementations were changed.

## Claude's GPU runs and remaining risks

1. Capture the real eight-prompt server workload once with `TRACE` enabled and `DRAFT_MIB=0`, keeping the measured target settings. Both target and host-resident draft routes will be logged. Replay target and draft context IDs separately. Turn tracing off for timing.
2. Keep the current target MiB budget, `ADAPTIVE=1`, `ADMIT=32`, `STATIC_PCT=0`, and no profile for the cold-start comparison. Compare `PREFILL=0` against `PREFILL=1 ADMIT_PREFILL=1024 DECAY=512`; also test caps 256/512. Then compare decay 128 and 64. Record TTFT and total request duration, not just steady-state tok/s.
3. Test `STATIC_PCT=25` with a genuine multi-topic calibration profile in `GGML_MOE_HYBRID_PROFILE`, first alone and then with prompt warming. The generated `base.profile` is suitable for the supplied trace study, not evidence of calibration transfer to the server suite. Do not pass that full profile with static percentage zero: the old over-budget check intentionally remains.
4. Test a separate draft reservation, for example `DRAFT_MIB=768`, with `DRAFT_PROFILE` unset for cold start or set to a head-only profile. Because VRAM is already full, reduce the target MiB budget by the chosen draft reservation before launching; the implementation does not do this automatically. Check the logged draft slot count/minimum before timing. `DRAFT_STATIC_PCT=25` is optional when a draft profile exists. Compare target loss against draft latency/acceptance gains at a fixed total budget.
5. Repeat the round-6 off/on/on/off pp4/pp1 sequence, long-text perplexity, and MTP-3 code/English runs. Use one untimed `GGML_SCHED_DEBUG=2` run to confirm the CPU split still ends at the miss mask and that the draft uses its own GPU-hit branch.

Open risks are CUDA graph reuse and multi-stream ordering under a much larger burst; pageable staging and map-update overhead; shared PCIe traffic from two independently adapting contexts; and the request-boundary heuristic under continuous/mixed batching. GPU quality is not claimed bitwise identical: changing which experts use GPU kernels can change rounding, target perplexity, draft outputs and acceptance. The synthetic replay neither models those routing changes nor predicts elapsed time. Static-base coverage and the one-group replay limitation should be checked before applying these results to another layout/device configuration.
