# Round 5: frozen GPU-hit / CPU-miss cache

Started 2026-10-17 13:07:30 UTC; deadline 16:07:30 UTC. Worktree `llama-l5`, branch `l5-cache`.

## Status

The port, persistent device maps, 1..8-token masked kernels, and GPU-hit-before-CPU-miss graph order are implemented, off by default. CPU correctness checks pass. CUDA compilation passes; GPU runtime validation is blocked by the execution environment. This is unvalidated WIP, not a successful performance checkpoint. No adaptive admission was started.

The required GPU measurements and commits cannot run in this sandbox:

- Creating a network socket raises `PermissionError: [Errno 1] Operation not permitted`, including for localhost. The mandatory gateway GPU free/hold endpoints are inaccessible. No GPU run or hold was started.
- `/proc/1/comm` is `codex`; `ps` lists only the per-command sandbox processes. Host load averages are visible, but foreign process CPU use is not. Therefore the required clean/watch audit cannot be established. The prepared runner refuses to measure with this process visibility.
- The worktree Git metadata is `/opt/llm-gateway/data/build/llama-upstream/.git/worktrees/llama-l5`. The authorized `git add` and stage commit fail to create `index.lock` because that filesystem is read-only. No commit was made, and no push or PR was attempted. Stage patches are preserved locally instead.

The user was asked to enable the missing access while local work continued. These are environment blockers, not evidence that the cache is fast or slow.

## What changed

Reconstructed the round-4 delta against `task-logs/round4-hitmiss/source-before.patch` in a temporary directory. The source checkout `llama-codex2` was read only. Older scheduler and AVX2 changes from that tree were not imported.

- Added `GGML_OP_MUL_MAT_ID_MASKED` with an F32 mask shaped like the IDs. Zero mask entries produce exact zero columns; active entries retain ordinary indirect-matmul semantics and valid IDs.
- CPU excludes masked routes from expert row groups and clears inactive output ranges. The reference path computes the ordinary matmul and then selects zeros, independently of the optimized masking path. Existing tiled expert computation remains in use.
- CUDA reuses MMVQ. Multi-token masking is per `(token, expert position)`, with a separate mask stride from the possibly strided ID tensor. A masked specialization permits eight warps/tokens without changing the existing unmasked kernel's launch bound. Inactive warps return before reading expert weights.
- The requested IQ3_S, IQ3_XXS, IQ2_S, IQ4_NL, IQ4_XS, Q2_0, Q4_K, Q5_K, Q6_K, and IQ2_XXS types are included in the CUDA support gate. Unsupported weight/layout/precision combinations retain the ordinary FFN for the affected layer. CUDA requires output rows divisible by 32 and F32 activations, and rejects a forced F32 activation-precision request.
- Frozen profiles retain the round-4 format: header `llama-moe-profile-v1`, then `layer expert training_count`. All entries are validated before uploading weights. There are no miss uploads.
- Frozen slot maps have device tensors owned by the cache/context and are uploaded once during initialization. Their allocation is included in the cache budget. Host maps remain available to the existing cache infrastructure. The frozen path never uses the scheduler's per-token map-copy callback.
- A layer with no resident experts returns no hybrid map and uses the ordinary graph. A header-only profile therefore selects the ordinary graph throughout; bit-identical logits still require the requested runtime comparison.
- Hybrid lookup makes top-k IDs contiguous before flattening the multi-token view. Both FFN branches preserve expert order and join before routing weights and reduction.
- LoRA, expert bias/scales, Hadamard transforms, warmup, batches outside 1..8, and unsupported GPU cases retain the ordinary graph. The existing `LLAMA_MOE_SKIP*` switches were not changed.

Enable with `GGML_MOE_HYBRID=1`, `GGML_MOE_HYBRID_MIB=<budget>`, and `GGML_MOE_HYBRID_PROFILE=<file>`. Do not combine with `--moe-cache-mib`. With the environment guard off, no hybrid cache or masked graph is constructed.

## Overlap: construction complete, proof pending

Graph order is routing/lookup/masks and the FFN input, then the GPU-hit FFN, then the CPU-miss FFN, then the join. The input is explicitly expanded before the hit branch. This gives the existing scheduler a possible cut after the last CPU-input producer, with hit computation in part B. No scheduler algorithm change was made apart from recognizing the new op in allocation expansion.

This is source inspection, not runtime proof. No `GGML_SCHED_DEBUG=2` overlap log or scheduler trace was collected. In particular, routing callbacks disable the scheduler overlap path, so trace/probe timings must not be used as overlap evidence.

Once gateway access and host process visibility are available, use `llama-perplexity -v` with `GGML_SCHED_DEBUG=2`, `GGML_SCHED_OVERLAP=1`, `LLAMA_MOE_SHEXP_EARLY=1`, a nonempty profile, and `-ub 4`. Confirm the reported part A endpoint is a CPU-input producer and the GPU hit ops lie in part B before the CPU split. Repeat for `-ub 1`. Also verify that no `moe_cache.slot_map_device-*` host-to-device copy occurs during evaluation.

## Validation

Build configuration: `GGML_CUDA=ON`, architecture 75, `/usr/bin/nvcc`, Release. Builds used `nice -n 10 cmake --build build-cuda -j 16`; no GPU hold was taken during compilation. A separate CPU-only build supplied a test binary that does not initialize CUDA.

| Check | Result | Evidence under `task-logs/round5/` |
| --- | --- | --- |
| Stage 1 single-token CPU masked tests | 474/474 pass | `stage1-cpu-final2.out` |
| Final CPU masked tests | 1314/1314 pass | `masked-cpu.out` |
| Ordinary CPU MUL_MAT_ID regression subset | 122/122 pass | `unmasked-cpu.out` |
| CPU support query | 1314 supported cases | `masked-cpu-support.csv` |
| Full CUDA build | Pass | `build-final.log` |
| Trace/logit probe compilation | Pass | `build-probe-final.log` |
| CUDA masked runtime tests | Not run: gateway access blocked | No result |
| Header-only profile logits, ub 1 and 4 | Not run | No result |
| Real-profile model correctness | Not run | No result |

Tests include zero/all/mixed masks, distinct and genuinely repeated IDs, broadcast and per-expert activations, strided expert tensors and ID views, all token counts 1..8, unsupported odd row counts, and longer reduction dimensions. The old round-4 test's `duplicate` argument only selected an offset ID view; this port explicitly fills repeated IDs. The first port test run accidentally repeated the case matrix three times; the authoritative stage-1 count above is after removing that duplicate registration.

CPU support records were appended to `docs/ops/CPU.csv`. `docs/ops.md` explicitly marks CUDA runtime validation pending; no CUDA support CSV was fabricated.

Reproduce local correctness:

```sh
build-cpu/bin/test-backend-ops test -b CPU -o MUL_MAT_ID_MASKED
build-cpu/bin/test-backend-ops test -b CPU -o MUL_MAT_ID -p 'type_a=(iq3_s|iq3_xxs|iq2_s|iq4_nl|iq4_xs|q2_0|q4_K|q5_K|q6_K|iq2_xxs),.*n=(1|4|5|8),'
```

## Performance checkpoint: not reached

No round-5 throughput, window latency, medians, spread, current hit rate, or VRAM availability was measured. Historical round-4 or user-provided bound figures are not substituted for new measurements.

| Model / placement | pp4 C/H/H/C, r30 | pp1 C/H/H/C, r30 | Held-out text hit rate |
| --- | --- | --- | --- |
| Qwen3.8, `-ncmoe 48` | Not run | Not run | Not run |
| Qwen3.6 UD Q4_K_M, `-cmoe` | Not run | Not run | Not run |
| Qwen3.6 UD Q4_K_M, `--n-cpu-moe 32` | Not run | Not run | Not run |

The required environment remains `OMP_PLACES=cores OMP_PROC_BIND=close GGML_SCHED_OVERLAP=1 LLAMA_MOE_SHEXP_EARLY=1`, with `llama-bench -ngl 99 -fa 1 -t 24 -n 0 -p <4 or 1> -r 30` and the model's CPU-MoE placement. Use the same frozen profile and budget for each hybrid invocation. Preserve raw repetitions, then compute medians and spread separately for control and hybrid. Repeat any invocation with foreign load.

## Perplexity: not run

The requested `/opt/llm-gateway/data/build/texts/test-de.txt`, 12 chunks, ctx 512 comparisons are pending for both `-ub 1` and `-ub 4`. There is no measured delta against the +0.5% gate. No quality-parity claim is made.

## Profile and experiment tools

`task-logs/round5/run.py` wraps one command with the required GPU free/hold/release lifecycle, a load-below-3 start gate, per-process CPU sampling, hold renewal, evidence files, and up to four attempts on foreign load. It refuses a sandbox PID namespace and enforces this round's deadline. It reads the API key internally and never prints it. Do not use it in its current sandbox to claim host cleanliness.

`moe-probe.cpp` is adapted from round 4 and compiled by `build-probe.sh`. `GGML_MOE_PROBE_WINDOW=1..8` evaluates windows; `GGML_MOE_PROBE_TRACE=1` records routing IDs and hit masks; mode 2 reads CPU miss masks. `GGML_MOE_PROBE_LOGITS=1` dumps every token's logits, and `GGML_MOE_PROBE_PREFIX` selects the held-out prefix length. It records device memory and buffer alignment. Callback tracing is for routing evidence only, since it disables overlap.

`analyze_routes.py` splits each topic in half for training/evaluation, groups host-resident layers by expert layout, and includes bank padding and device maps in its budget. It supports the partial CPU-offload Qwen3.6 case, accepts `--budget-mib` and `--reserve-mib`, and reports residency hits over host-resident layers. It is prepared for a longer trace; no new training trace or selected production profile was generated. Measure memory with the intended MTP head loaded before fixing the budget. Do not treat the default reserve as an MTP memory measurement.

## What the next round needs

First complete the blocked CUDA tests, empty-profile logits, overlap proof, longer trace/profile, interleaved checkpoint, and both perplexity checks. The performance gate is unresolved, so there is no measured basis yet for proceeding to adaptive LFU.

If those checks justify adaptive work, retain the context-owned maps, support gating, complementary masks, expert-order join, and measured split boundary. Admission/upload work needs its own measurements for CPU savings, upload cost and interference with overlap. No LFU decay, admission, eviction, upload queue, or upload-on-miss policy was added here.

Final local checks: `git diff --check`, changed-source ASCII/whitespace checks, and Python compilation pass. The profile analyzer passed a synthetic partial-offload test covering empty/full residency and budget accounting (`profile-analysis-smoke.log`); this is not a model hit-rate measurement. All tracked build/test processes finished successfully, and no GPU hold needs release.

Local work stopped at 2026-10-17T13:22:30+00:00 with GPU measurement and Git-write access still unavailable. `stage1.patch` preserves the initial port; `stage2-3-source.patch` preserves the final tracked diff; `round5.patch` also includes the notes and experiment source files.
