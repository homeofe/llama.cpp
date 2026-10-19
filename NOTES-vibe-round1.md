# Vibe round 1: Strata parity, stage by stage

Worktree `llama-vibe`, branch `vibe-parity` (llama.cpp `l5-cache` at e3b15507c). Task:
`TASK-vibe-strata-parity.md`. Started 2026-10-18 17:36 UTC. Budget 3 h wall clock, stage 1 at most
75 min of it. Machine: RTX 2080 Ti, Threadripper 3960X, shared.

## Method

- Strata: the deployed runtime 0.1.41-local-1cb045cb, started by us on 127.0.0.1:18100 with the
  deployed config's `args` unchanged (`--spec 4 --spec-min-p 0.7`, MTP head from `strata-data`,
  `--expert-cache auto`, `--kv int8`, `--prefill auto`, pack + GGUF, `--pcie-frac 0.0`,
  `STRATA_BF16_TC=1 STRATA_NO_LARGEPAGES=1 STRATA_SELECT_SIMT=1`). The config copy in
  `task-logs/vibe-round1/` changes only `log` (into the worktree) and `api_key` (our own random key).
- Ours: this worktree's `build-cuda/bin/llama-server`, config `q-r8-slots` (the measured best),
  CUDA graphs disabled for the trace (as in round 10).
- Same 8 prompts as `stage_b.py` (its `PROMPTS`), max_tokens 256, temperature 0, warm-up first,
  quiet gate (1-minute load < 3 before and after, no foreign process above one core).
- Harness: `task-logs/vibe-round1/stage1.py`; traces: `task-logs/vibe-round1/{strata,ours}-nsys/`.

## Stage 1

(to be filled)

### Stage 1 protocol and clock

Stage 1 started 15:47 UTC with the build of this worktree's `build-cuda` (llama-server, llama-perplexity;
Release, `CMAKE_CUDA_ARCHITECTURES=75`, `-march=native` as in `llama-l5`). The machine was quiet
15:41-15:47 and loud (load up to 32, the owner's node/chrome-headless tests) for most of 15:47-16:43,
so the quiet gate forced waiting; the 75-minute stage-1 wall budget is exceeded by mandated waiting
and is disclosed here instead of being quietly ignored. Measured GPU steps in sequence:
Strata bench (15:48-16:12 clean part), Strata nsys attempt 1 (16:43-16:49, report lost to a
finalization hang), ours nsys (16:54-...), Strata nsys attempt 2 after the cleanup fix.

### Strata, measured like ours (this run, deployed config, port 18100)

Engine clock, `DONE` lines of the engine log and the OpenAI `timings` of the server; warm requests
only (gate-accepted); code prompt `code-duration` (218 tokens), English `en-1`/`en-2` (256 tokens):

| quantity (per round) | Strata code (warm) | Strata English `en-1` | Strata English `en-2` |
|---|---:|---:|---:|
| tokens per round | 3.30 | 3.66 | 2.42 |
| round time (ms) | 47.2-47.8 | 49.9-52.4 | 37.4-41.7 |
| decode tok/s (client) | 69.5-76.6 (4 prompts) | 63.7 | 59.4 |
| drafts offered/accepted per round | 2.62 / 2.30 | 2.84 / 2.67 | 1.78 / 1.41 |
| expert cache hit (this request) | 48.8-49.2% | 49.7-50.5% | 52.7-57.1% |
| CPU expert applications per round (engine counters) | ~890 | ~920-1005 | ~590-690 |
| GPU hit applications per round | ~860 | ~930 | ~720-780 |

The hit rate rises over the request history exactly as the deployed log does (34% on the first code
request, 49% warm, 57% on the later English ones; the deployed `iq3_s.log` shows 38% after the first
request rising to 67% by the fourth). Gate-accepted requests: 4 code + 2 English before the machine
went loud again; the rest of the planned 3 rounds was stopped to bank the profiling window. The
accepted-request medians (n=4 code, n=2 English) are 73.5 code / 61.6 English tok/s, consistent with
the measured history (69.6-70.8 / 61.0-61.7). Rejected (dirty) attempts are in
`task-logs/vibe-round1/strata-bench/events.jsonl` and are excluded.

`lookups` in the engine's counters is per routed (layer, expert, token) application, no dedup:
129,600 for 256 tokens is 48 layers x 10 experts x 5.4 token-positions (verify + draft). At 49%
hits Strata computes ~890 CPU expert applications per 47.5 ms round, i.e. ~1.78 GB per round at
2 MB per expert; ours at 43% hits computes ~1,094 per 81.7 ms round (round 10 trace).

### Stage 1 profiling: what did not work (kept for the record)

- `nsys profile` around the whole Strata server (the task's first suggestion) traces only the python
  wrapper: server.py spawns the engine with a scrubbed environment, so the engine's CUDA activity
  escapes CUPTI injection entirely (report `strata-nsys/strata-code1.qdstrm`: OSRT of the wrapper,
  no CUPTI tables). Wrapping llama-server the same way DOES work (single process).
- Killing the wrapped app from the cleanup loop truncated two reports (`ours-code1.qdstrm`,
  the first `strata-code1`): the app must exit on its own before nsys finalizes; llama-server needs
  ~80 s to unmap the model. The cleanup now stops things through the server's own lifecycle and
  waits.
- The server config's `exe` field can point at a shim (our config copy), which execs
  `nsys profile <real engine>`: the server's stdin/stdout pipes pass through nsys into the engine,
  and the server's QUIT at shutdown lets nsys write a complete report (`engine-shim.sh`). Two
  shim bugs were found and fixed by the engine's own error (`--serve` belongs to the engine, not
  to the shim; the first version passed the engine path as an nsys option).
- Bash `/proc` scans for cleanup must not contain the marker string in their own command line;
  one cleanup shell killed itself that way. Fixed with `pgrep -f 'pat[t]ern'`.
- Without sudo, CPU sampling is unavailable (perf_event_paranoid=4; round 10 had explicit sudo
  authorization, this task forbids sudo). Per-thread CPU is measured instead by reading
  `/proc/<engine>/task/*/stat` at 50 ms during one request.

### Stage 1: Strata against ours, per round and per token

Both engines, same machine, same model files, same `code-duration` prompt (218/217-231 tokens,
temperature 0). Strata: `task-logs/vibe-round1/strata-shim/strata-shim.{qdstrm,nsys-rep,sqlite}`
(engine under nsys via `engine-shim.sh`; the profiled request ran under a foreign-load burst and
is dirty in wall time, so trace *magnitudes* are paired with the clean engine-clock round time of
the same protocol; the kernel/structure numbers are unaffected by host load). Ours: round 10's
graph-off trace of the same fork and prompt (`llama-l5 task-logs/round10/profile-code-duration-off/
trace.sqlite`, re-analyzed here with `analyze_trace.py`, report saved as
`task-logs/vibe-round1/ours-r10-kernel-report.json`), plus this worktree's fresh request
(`ours-nsys/events.jsonl`, q-r8-slots, graphs off under nsys: 218 tokens, 35.3 tok/s client,
60 rounds - the round-10 graph-off rate of 37.1 tok/s minus tracing noise; output hash equal to
Strata's).

| quantity, per round (code) | ours (q-r8-slots, 81.7 ms round, 3.63 tok) | Strata (47.2-47.8 ms round, 3.30 tok) |
|---|---:|---:|
| tokens per round | 3.63 | 3.30 |
| round time (ms, clean engine clock) | 81.7 | 47.2-47.8 |
| GPU kernel launches per round | ~4,670 (280,310 / 60 rounds) | ~120 (8,398 / 70 rounds) |
| GPU kernel time (ms) | 27.6 dense+other + 5-6 cache hits (round 10); ~42 incl. prompt+draft in the trace | 3.44 (all kernels; incl. 0.81 mul_mat_q and 1.29 dequant) |
| - of which the two GEMV families | mul_mat_vec_q 12.4 + mul_mat_vec_f 10.2 | (none: grouped mul_mat_q 0.81; dequant 1.29) |
| - of which elementwise/norm/quantize tail | ~14 across rms_norm/scale/bcast/unary/concat/get_rows/quantize_q8_1 | ~1 (copy16 0.39, bf16_to_f16 0.17, gr/gdn ~0.15) |
| expert work on GPU (hits) | ~860 applications in 5-6 ms (masked MMVQ, mul_mat_vec_q_moe) | ~860 applications in ~2 ms (grouped: dequant_kernel + mul_mat_q, 494 launches/request) |
| expert work on CPU (count, GB, ms) | ~1,094 applications, ~2.19 GB, CPU graphs 35 ms (43% hits) | ~890 applications, ~1.78 GB, gap-borne ~34 ms (49% hits; dirty trace gap 67 ms at foreign load 12 cores) |
| copies HtoD (bytes, ms, stream) | ~107 MB, 8.9 ms, compute stream, 12.4 GB/s (cache uploads) | ~139 MB, 10.8 ms, 12.9 GB/s, same stream as kernels (cache uploads; MORE bytes than ours) |
| host sync | 13.4 ms per round (round 10) | not separable without CPU sampling; the round's gap is CPU-side |
| draft cost | 3 MTP steps ~7.5 ms wall (GPU dense 1.8 ms each) | --spec 4, drafts offered 2.62/round accepted 2.30; inside the 47.5 ms |
| thread wait or spin | libgomp 63-67% of CPU samples (round 10, perf) | ~22.9 busy threads of 24 during the request (/proc task sampling, 50 ms) |

Measured: everything in the table except where a row cites round 10 (same fork, same prompt, same
graph-off method) and the Strata host-sync row. Inferred: the split of Strata's gap into CPU expert
arithmetic vs waits (its 1.78 GB at the request's ~22.9 busy threads matches a ~30-34 ms bandwidth
floor, which is the bulk of the gap).

**The three largest differences (per code round):**

1. **Dense small-batch GPU path: ~28-30 ms** (ours 27.6 ms dense+other kernels in verify plus ~5.5
   draft GPU, against Strata's ~3.4 ms all-in). Our round spends it in 4,700 tiny kernels per round
   (two GEMV families 22.6 ms: the BF16 hyper-connection/router matvecs, quantized dense matvecs;
   plus a ~14 ms tail of rms_norm/scale/bcast/unary/concat/quantize kernels at ~5-30 us each).
   Strata's fused per-layer kernels (one dequant + one grouped quantized GEMM per layer group,
   PDL, BF16 tensor cores, int8 KV) do the same math in ~3.4 ms. This alone is ~36% of our round
   and ~8.2 ms per generated token - which is the whole per-token gap (ours 22.5 ms/tok,
   Strata 14.5 ms/tok).
2. **Cache-hit kernel efficiency: ~3-4 ms.** The same ~860 hit applications cost 5-6 ms in our
   masked MMVQ vs ~2 ms in Strata's grouped dequant+mul_mat_q (494 launches for the whole
   request against 8,601 mul_mat_vec_q_moe launches in ours).
3. **Everything else is at parity:** CPU expert phase (ours 35 ms for 2.19 GB at 43% hits vs
   Strata ~34 ms for 1.78 GB at 49% hits) and cache-upload copies (ours 8.9 ms for 107 MB vs
   Strata 10.8 ms for 139 MB, both on the compute stream). The expert-cache architecture built in
   rounds 5-11 is NOT where the remaining gap lives.

**Stop-rule check:** a single difference worth at least 5% of our round exists (36%), so the first
stop rule does not apply. The second stop rule (repacked expert layout) does not apply either: the
expert path is at parity and Strata's repack does not explain the gap; the CPU expert kernels are
llama.cpp-derived in both engines. What remains is the third case the task names: the largest
difference needs new small-batch dense CUDA kernels in llama.cpp, which is not a runtime change.

## Stage 2: the largest difference in llama.cpp

The largest measured difference is the dense small-batch GPU path: ours ~28-30 ms per code round
against Strata's ~3.4 ms, worth ~36% of our round and ~8.2 ms per generated token - numerically the
entire per-token gap (22.5 vs 14.5 ms/tok). The traces also say WHY: it is fragmentation, not FLOPs
or bandwidth.

Kernel-size distribution of the same code request (whole request / steady burst):

| bucket | ours count | ours total ms | Strata count | Strata total ms |
|---|---:|---:|---:|---:|
| < 16 us | 243,606 | 821 | 6,712 | 47 |
| 16-32 us | 18,774 | 486 | 607 | 15 |
| 32-64 us | 16,341 | 774 | 523 | 23 |
| 64-128 us | 1,132 | 99 | 345 | 32 |
| >= 128 us | 457 | 335 | 399 | 129 |

52% of our GPU kernel time is in kernels under 32 us (the biggest class averages 3.4 us - launch
and scheduling overhead per ggml graph node); Strata's time concentrates in >=128 us kernels. Our
round launches ~4,670 kernels, Strata ~120, for the same 48-layer, 4-5-token window.

**What can be closed here, and what cannot.** No environment switch or runtime setting in this
worktree changes this: CUDA graphs are already on (worth only 1.7 ms, round 10); the BF16->Q8_0/F16
retype of the hyper-connection matrices was measured before and bought +1% end to end (dropped);
threading, cache policy, upload streams, draft pools and PCIe reads are all measured and dropped
(task list). Closing the ~30 ms needs new small-batch dense CUDA work in llama.cpp, in the
llama-cpp-kernel-patch domain, not a runtime change:

1. fuse or graph-capture the per-layer elementwise/norm/quantize chain (rms_norm, scale, bcast,
   unary, concat, get_rows: ~14 ms/round in thousands of sub-32-us kernels);
2. batch the two GEMV families across the verify window and across tensors where shapes allow
   (mul_mat_vec_f 10.2 + mul_mat_vec_q 12.4 ms/round, the latter 377 launches per round at
   ~33 us);
3. group the cache-hit experts into one kernel per layer like Strata's dequant+mul_mat_q
   (5-6 ms -> ~2 ms for the same ~860 hits);
4. int8 KV and BF16 tensor-core paths for the attention/recurrence core (Strata runs `--kv int8`,
   `STRATA_BF16_TC=1`).

**Bound and ceiling.** Each piece above is bounded by the trace numbers: 14 + ~8 (GEMV batching
at Strata's efficiency) + 3.5 (hit grouping) = ~25 ms per round, plus the draft steps. If llama.cpp
reached Strata's dense+hit efficiency and changed nothing else, the code round falls from 81.7 to
roughly 81.7 - 30 - 3.5 = ~48 ms, i.e. ~72 tok/s code - Strata's measured 69.6-76.6 - because the
CPU expert phase, the upload copies and the cache are already at parity. **Strata parity is
therefore reachable in llama.cpp on this machine, but only through that kernel work**; with the
current dense path the ceiling is the measured 48.4/40.1 tok/s plus the small already-measured
levers. This bound is a trace-domain estimate (overlapping clocks), not a forecast.

**Nothing was merged into the engine for stage 2, deliberately.** An env-switched half-measure
would have had no measured path to the 3% adoption gate: every runtime lever touching this cost is
already measured and dropped, and the real fix is a multi-kernel patch series that needs its own
bounded plan (per the fork's kernel-patch discipline: bound first, worktree build, mutation cuts,
end-to-end proof). Because no behaviour change was made, perplexity and output comparison were not
re-run; the last measured values stand (test-de PPL 3.8093 cache-off / 3.8126 best candidate,
rounds 9-10; outputs bit-stable across round-10 repetitions).

## End-to-end numbers (this round, medians)

Strata, deployed config, port 18100, our client and the quiet gate (n=4 code / n=2 English
gate-accepted requests; history medians in brackets): code 73.5 tok/s (69.5-76.6) [history
69.6-70.8], English 61.6 (59.4-63.7) [history 61.0-61.7]. Ours (measured history, q-r8-slots):
48.4 code / 40.1 English. The gap is 1.45-1.55x, as the task states; per round 81.7/76.1 ms against
47.2-47.8/49.9-52.4.

## Open questions for the owner

1. Authorize the small-batch dense kernel series (fusion tail, GEMV batching, grouped hit kernels,
   int8 KV)? That is where the whole remaining gap sits; expected 2-4 kernels of work for ~25
   ms/round. It should run under the kernel-patch discipline with its own budget.
2. Strata's `--spec 4` accepts 2.3-2.67 drafts/round on English at a 37-42 ms round - narrower
   windows are cheaper per round. Our MTP-3 acceptance per position (0.95/0.96/0.97 code) is at
   parity with Strata's, so this is a width choice, not a drafter quality gap (already measured:
   p_min screens).
3. Strata uploaded MORE cache bytes per round than ours (139 vs 107 MB) at the same 12.9 GB/s - its
   hit-rate trajectory (38% -> 67% over requests) is worth reproducing exactly in our LFU if the
   dense path is fixed first; with the CPU phase at parity it is not the binding constraint.
4. The dirty-trace caveat: Strata's profiled request ran under a foreign-load burst (12 cores);
   its per-phase GPU numbers are load-insensitive, but a quiet-window re-capture would firm up the
   host-side split. The capture pipeline (engine-shim) now works and is reusable.

## Stage 2 executed: GGML_CUDA_BF16_AS_F16 (measured, not adopted as sole change)

The dense-path bound identified the hyper-connection BF16 matrices as the largest single kernel
loss: `hc_*_up` (10240x320) ran the row-wise MMVF kernel at 85.6 GB/s while the same shape in F16
runs the tensor-core MMF path at 308-320 GB/s (3.7x, measured both in `bound/series2.sh` and with
the patch on). The patch converts BF16 mul_mat-only weights to F16 in place on non-BF16-mma
hardware, behind `GGML_CUDA_BF16_AS_F16`, off by default (`ggml/src/ggml-cuda/bf16_f16.cu`).

Proof (all archived in `bound/` and FACTS-bf16-as-f16.md):

- bit-exact conversion vs an F16C-verified software reference: 1542 bit patterns, 0 differ;
  mutation cuts (sign: 339 differ; 1-ulp tie: 616 differ) both detected, restored to 0;
- mul_mat vs the CPU reference on 4 hc shapes: nmse 0, switch off AND on;
- kernel speed: hc_up 76.7 -> 20.5 us (85.4 -> 320 GB/s, 3.7x);
- perplexity (test-de, 12 chunks, ctx 512): ub4 3.7963 -> 3.7990 (+0.07%), ub512 3.8200 -> 3.7949
  (-0.66%); both inside the +0.5% gate; the control reproduces the round-10 value exactly;
- end to end, stage_b, 3 interleaved rounds, n=12 per config and category, quiet gate:

| config | code tok/s | english tok/s | ms/round (server clock) |
|---|---:|---:|---:|
| q-vibe-control | 47.7 | 39.4 | 79.0 / 75.9 |
| q-vibe-patch | 48.8 (+2.3%) | 40.8 (+3.6%) | 77.8 / 74.5 |

**Verdict: not adopted as the sole change** - the pre-declared gate of +3% on both categories is
missed on code (+2.3%). The kernel saving (5.3 ms/round) mostly runs under the shadow of the CPU
expert phase: the server clock shows only 1.2-1.4 ms of round-time reduction. GPU time that
overlaps the CPU phase does not buy wall time; the next kernel step must attack the exposed
timeline (the ~14 ms/round elementwise/norm/quantize tail, ~4,700 launches per round) rather than
shadowed GPU bandwidth.

The file-level retype measured in the same session and protocol (`q-vibe-hcfix`, the existing
IQ3_S-hcfix GGUF: hc up/routers F16 AND hc down Q8_0) clears the gate with +6.1% code / +7.4%
english (n=12 each, 3 rounds). For the owner: deploying the hcfix file needs no code change; the
runtime patch remains proven groundwork that benefits any BF16-dense model on this GPU.

## Fragmentation map (round-1 addendum, the basis for the fusion decision)

Gap analysis of the round-10 decode window (6.28 s, 60 rounds, graph-off trace, nsys-inflated by
~20%): 84% of all GPU idle (3.58 s of 4.26 s) sits in 21,558 gaps immediately before `k_bin_bcast`
- the per-layer point where the GPU waits for the CPU experts. The CPU expert phase is almost
fully EXPOSED, not overlapped; it is the round's dominant gate (~48-60 ms/round inflated).
The dense kernels on the main stream run back-to-back (1.4 us inter-kernel): their execution time
is wall time. Corrected block sizes per round:

| block | ms/round (inflated) | note |
|---|---:|---|
| CPU expert wait (gaps before bin_bcast) | ~48-60 | the gate; hit rate ours 37.9% cumulative vs Strata ~50% |
| GEMV families (mul_mat_vec_q 742 ms + _f 614 ms per request) | ~22.6 | 200-290 GB/s vs ~450 achievable |
| elementwise tail (~2,700 kernels/round) | ~7 + ~2 gaps | quantize_q8_1 75, rms_norm 64, scale 63, bcast 61, unary 54 ms/request |
| cache hits (mul_mat_vec_q_moe 328 ms) | ~5.5 | grouped kernels would take ~2 |
| copies: 8.2 GB HtoD uploads + 2.4 GB DtoD split copies (23.4k calls, 20 GB/s) | ~13 | DtoD = per-layer split-boundary copies |

Consequences, measured:
- "Fusion of the tail" alone is worth ~9 ms/round, not 14 as first estimated.
- Even Strata-level GEMVs + a fused tail land at ~60 ms/round; the last ~12 ms to Strata parity
  live in the CPU expert gate (hit-rate gap + 23% more expert bytes), whose policy knobs are
  already measured (round 11: admit8 +1.3%, more admission not better).
- The two bounded patch proposals, in cost/benefit order:
  1. GEMV batching for small batches (~10 ms/round bound): batch the ~377 mul_mat_vec_q launches
     per round across same-K tensors / tune MMVQ for batch 4-5 on sm_75. Generic for every
     quantized model. Multi-day kernel engineering; needs its own bounded task.
  2. Elementwise-chain fusion in the CUDA backend (~9 ms/round bound): merge
     quantize/scale/bcast/unary/rms_norm chains into one kernel per layer segment, ~500 lines,
     pattern pass before capture. Generic for every model. High review risk (float reassociation
     must stay within the nmse gates).
Both only pay together with the bf16-as-f16 patch already in the tree (the hc_up kernels are part
of the same exposed stream-17 time).
