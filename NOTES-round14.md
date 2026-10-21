# Round 14: the final stack re-baselined - where the 84 ms round actually goes

Phase 0 of TASK-round14.md: a fresh nsys capture of `q-vibe-async` (the round-1 final
stack: bf16-as-f16, MMVQ nwarps 2, int8 KV, 1779 slots, the two fusions, async uploads)
on the same `code-duration` prompt, same methodology as round 1 (graphs disabled, one
steady request, quiet gate, GPU held). This replaces the round-1 stage-1 table, which
was measured on the pre-patch baseline and no longer describes where time goes.

## How the capture was made (the round-1 lesson repeated)

Wrapping llama-server with plain `nsys profile` still loses the report (round 1 hit this
too: `ours-nsys/` contains only a 412 KB torso and the analysis fell back to round 10's
trace). The working recipe is round 10's: `--capture-range=nvtx --nvtx-capture=round14
--capture-range-end=stop --env-var=NSYS_NVTX_PROFILER_REGISTER_ONLY=0`, an NVTX range
pushed from the driver around the steady request, and the sqlite export run manually
with the QdstrmImporter binary (nsys's own importer lookup fails from the wrapped env).
Artifacts: `task-logs/vibe-round14/{driver.py,profile2.py,trace.qdstrm,trace.sqlite}`,
report `round14-report.json`. The request: 218 tokens, client 43.8 tok/s decode, hash
`6085cbab...` - bit-identical to round 1's same-prompt output.

## The decision table (decode window, 60 rounds, graphs OFF)

Request wall 84.1 ms/round client-side (5045 ms / 60 rounds). Trace decode window 5.62 s.

| bucket | ms/round | share | round-1 baseline | note |
|---|---:|---:|---:|---|
| GPU kernels, total | 34.9 | 42% | ~33 (27.6 dense + 5-6 hits) | flat overall - see the split |
| - mul_mat_vec_q | 11.5 | | 12.4 | MMVQ nwarps patch; round 13 closed further gains |
| - mul_mat_vec_q_moe (cache hits) | 5.6 | | 5-6 | unchanged |
| - mul_mat_f (f16 tensor-core path) | 5.4 | | (inside the dense blob) | the bf16-as-f16 patch routes the hc matvecs here |
| - mul_mat_vec_f | 1.5 | | 10.2 | the bf16-as-f16 patch: -8.7 ms/round |
| - elementwise/norm/quantize tail | ~9.5 | | ~14 | quantize_q8_1 1.2, gated_delta_net 1.1, rms_norm 1.0, k_bin_bcast 0.97, concat 0.84, unary 0.83, get_rows 0.78, dsv4_hc 0.65, scale 0.33, ... |
| - attention + rest | ~1.4 | | | flash_attn_ext 0.27, topk 0.29, rope 0.16, ... |
| GPU copies (HtoD) | 9.3 | 11% | 8.9 | 103 MB/round at 12.4 GB/s; now spread over streams 13/18/21 (async upload works: 5.1 ms/round on the separate stream, overlapped) |
| GPU idle | ~49 | 52% | (35 CPU expert + 13.4 host sync, similar) | the CPU expert wait is still the round |
| - idle in gaps > 0.5 ms | 31.9 | | | 36 gaps/round, avg 0.9 ms; NOT covered by CUDA API calls - the host threads are in CPU expert compute (the round-1 "21,558 gaps before k_bin_bcast" shape is gone; async uploads consolidated it) |
| - small gaps | ~17 | | | launch latency, round boundaries, async-stream fences |
| host cudaStreamSynchronize | 13.5 (host-blocked, mostly while the GPU is busy) | | 13.4 | 556 syncs/round, 21 us average - unchanged by the async-upload patch |

Two headline facts:

1. **The final stack's improvements are real but redistributed**: the graphs-off client
   rate went 35.3 -> 43.8 tok/s (+24%) vs the round-1 capture, and the bf16-as-f16 patch
   alone moved 8.7 ms/round out of mul_mat_vec_f; but mul_mat_f (the tensor-core path the
   patch routes to) now carries 5.4 ms/round, and total GPU kernel time is back at ~35
   ms/round. The GPU work was never the wall.
2. **52% of the round is still GPU idle, and it is still the CPU expert phase**: the
   >0.5 ms gaps are host-side compute (no CUDA API coverage), ~32 ms/round of exposure.
   The CPU phase is DDR4-bandwidth-bound (round 1: 2.19 GB at 55-66 GB/s) with a real
   data dependency per layer (the routing for layer L's tokens is only known when layer
   L runs). Async uploads consolidated the wait shape (359 small waits/round -> 36 bigger
   ones) but did not and cannot hide the compute.

## The candidates, dispositioned by this table

- **Slots bump (config)**: DEAD. `nvidia-smi` during the steady decode: 10570/10820 MiB
  used, **252 MiB free**. The hit-rate-vs-slots curve (sim: +3.3 points for 2048 slots)
  needs ~600 MB. Nothing to take.
- **Elementwise tail fusion**: ~9.5 ms/round remains after round 1's fusions took ~4.5.
  Sharper estimate from the launch counts: the tail is launch/latency-bound, not
  bandwidth-bound - quantize_q8_1 averages 2.5 us per launch (30,013 launches, 1.25
  ms/round), rms_norm ~3 us (17,024, 1.0 ms/round), k_bin_bcast ~2.4 us (23,927, 0.97).
  Fusing a producer+quantize pair eliminates one kernel invocation per site (the
  activations are 4-5 x 2560 f32, ~50 kB - the memory saved is nothing; what is saved is
  the ~2-3 us stream slot per launch). At ~500 quantize sites/round the full win is
  ~1.0-1.5 ms/round (~1.5% e2e), and it survives graphs-on (graphs hide launch cost but
  not stream serialization). **Below the +3% gate on its own** - worth doing only if it
  comes free with other work.
- **mul_mat_vec_f / mul_mat_f**: already on the optimized path (10.2 -> 1.5+5.4). The
  5.4 ms of mul_mat_f at 264 launches/round is tensor-core work at the measured shape
  speeds; nothing further without the graph surgery round 1 named.
- **Host syncs (556/round)**: mostly overlapped with GPU work; reducible only by deeper
  scheduler surgery in the moe-cache patch, uncertain payoff.
- **CPU core reservation**: does not raise the quiet-machine ceiling (the CPU phase is
  bandwidth-bound, not core-starved - 22.9 of 24 threads busy in round 1); it protects
  the under-load number only. The owner asked about it; it remains available as a
  deployment choice if npm load hurts the live gateway.
- **The CPU expert exposure itself (32 ms/round)**: structural. The experts' routing is
  known only per layer during execution; hiding the compute needs speculative routing
  (research), not a patch. On this machine, with the policy space closed (rounds 5-12),
  the slots capped by VRAM, and the mma route dead (round 13), this is the honest
  end of the line for the decode path on this GPU.

## Bottom line

The final stack is at its structural ceiling on this hardware: ~35 ms of GPU kernels
(measured at their per-family limits), ~9 ms of overlapped uploads, and ~32 ms of
CPU-expert wait that no measured knob removes. The remaining patch-worthy item is the
2-3 ms of fusable elementwise tail; everything else in the round-14 candidate list is
closed by a number in this table.
