# Round 13: int8-MMA for the small-batch Q6_K matvec - measured dead, the feed is the wall

Result up front: **the stop rule fired.** No prototype structure reached 350 GB/s at batch 4
(best 222 GB/s, both structures lose to the current dp4a path's 300 GB/s). The phase
isolation shows why, and it is a structural answer, not a tuning one: the INT8 tensor
cores are fast enough (the mma phase alone runs at 690-750 GB/s equivalent), but **every
way of feeding the m16n8k16 tile with Q6_K data costs more than the entire current
dp4a kernel**, because the tile needs 16 rows per warp while the fast load patterns
deliver one row per lane. The round-1 hypothesis "the batch-4 path is dp4a-ALU-bound,
only int8 tensor cores lift it" is disproven for this machine and tile shape.

All numbers below: RTX 2080 Ti, SM clock pinned at 1350 MHz during the runs (both paths
measured interleaved in the same minutes, `nvidia-smi` confirmed the clock), 200 iters,
best of 3 rounds, kernel-only CUDA-event timing for the prototype, full-graph timing for
the ggml path (includes the q8_1 quantize kernel and CUDA-graph overhead, i.e. the
comparison is generous TO the prototype). ncu was unavailable (ERR_NVGPUCTRPERM, no sudo
on this machine), so the prototype carries its own phase probes instead (bmode argument).

## The two prototype structures (bound/mma8_bench.cu)

Both use the mma.cuh tile API, which on Turing issues 2x `mma.sync.aligned.m8n8k16.
row.col.s32.s8.s8.s32` per mma() call, and the MMQ scale trick: one mma covers 16 k
values = exactly one Q6_K scale group, so the int8 scale is applied to the int32 C
fragment after the mma, the fp16 super-block d and the q8_1 d8 around it. The fragment
layouts were verified first with a standalone probe (bound/mma8_probe.cu, LAYOUT 1:
128/128 elements correct): lane l holds A row l/4 (+8 for x[1]) at k-bytes 4*(l%4)..+3,
B token l/4 at k-int32 l%4, C (i,j) = (8*(l/2)+l/4, 2*(l%4)+l%2).

- **v1 shared-staged** (the MMQ-heritage shape): warp owns 16 rows x full K; per 256-value
  super-block it dequantizes to int8 in shared (sA 16x68 int32, natural k order, row-per-pass
  so the ql loads are perfectly coalesced), ldmatrix A, load_generic B from shared y,
  next super-block prefetched into registers so the loads fly during the mma phase.
  255 registers, 7808 B shared, 6 warps/SM.
- **v2 fragment-perfect** (no shared staging at all): the lane assignment makes every lane
  unpack exactly the registers its own fragments consume - lane l owns rows l/4 and l/4+8
  at ql int32s c+4j (c = l%4), which produce precisely the slots the A fragments need
  (every slot a lane produces is == c mod 4, and the m8n8k16 fragment asks lane l for
  slots == l%4). A and B are built in registers, no ldmatrix, no sA, one small shared
  yd table, no barriers except one syncwarp for the 64-byte yd.
  193 registers, 256 B shared, 10 warps/SM.

## The measurements (Q6_K, n_tok=4, quiet windows, interleaved)

| what | 10240x4x2560 | 2560x4x6144 |
|---|---|---|
| current ggml MMVQ path (mmv_bench, full graph) | **71.4-71.5 us, 300-306 GB/s** | **48.7-49.4 us, 261-265 GB/s** |
| mma8 v1 full kernel | 97.5-97.8 us, 222 GB/s | 73.6-74.5 us, 174-176 GB/s |
| mma8 v2 full kernel | 101.9-102.1 us, 213 GB/s | 85.5 us, 152 GB/s |
| v1 mma phase alone (one dequant, 10x mma) | 31.4-31.7 us, ~700 GB/s | - |
| v1 dequant alone (loads+unpack+STS+syncwarp) | 81.8-82.1 us, 265 GB/s | - |
| v1 loads+ALU without the STS (no-store probe) | **44.2 us, 490 GB/s** | - |
| v2 mma phase alone | 28.9 us, 749 GB/s | - |
| v2 dequant alone (scattered self-loads, no stores) | 93.8 us, 231 GB/s | - |

The phase arithmetic closes: v1 full = dequant(82) overlapped with mma(31) by the prefetch
= 98; v2 = 94 + 29, worse overlap. The **dequant feed** is the whole kernel, in every
structure:

- Coalesced loads + unpack alone: 44 us (490 GB/s) - this is the floor of any structure
  that loads Q6_K the way MMVQ does (same 2-byte `get_int_b2`-style loads, the 210-byte
  block_q6_K is 2 mod 4 and cannot take wider vector loads).
- Cross-lane delivery via shared staging (STS + syncwarp + ldmatrix): **+38 us** (44 -> 82).
- Cross-lane delivery via scattered self-loads (v2's register approach): **+50 us**
  (44 -> 94, and it also breaks coalescing).
- Cross-lane delivery via shfl: counted out on paper, ~256 shfl per lane-super-block
  (the fragment needs 16 values per pass-pair across all 16 k-tiles), strictly worse
  than the STS path.

The current MMVQ kernel needs **no** cross-lane movement - each lane streams its own row
and dp4a consumes it directly - which is why its whole kernel (71.5 us) is faster than
our dequant feed alone (82-94 us). The m16n8k16 tile fundamentally requires 16 rows in
one warp; on this memory system that requirement costs more than dp4a saves.

Clock-state note: the phase-0 numbers in TASK-round13 (71.66 us / 300.1 GB/s) reproduce
exactly (71.4-71.5). An early 80-us v1 reading was taken at a higher boost state and did
not reproduce; all numbers here are from the interleaved runs at 1350 MHz.

## Correctness work (all of it passes; kept in bound/ for reuse)

- Full check vs a double-precision reference with the same integer semantics (raw
  int8 weight values x quantized activations, scale group by scale group): PASS on
  64x256x4, 1024x512x4, 1024x6144x5, 1024x2560x8, 48x512x1, 256x2560x4, 10240x2560x4.
  Typical max_rel 1e-5..1e-4; worst case 3.3e-4 relative at 1.2e-6 absolute on a
  near-zero accumulator (fp32 rounding of ~500-magnitude group terms, gate is
  rel > 2e-4 AND abs > 2e-6).
- Mutation cuts, all FAIL loudly as required: MUT_OFFBY1 (rel 55), MUT_SIGNHI (rel 1351),
  MUT_NO_SCALE (rel 10.6), MUT_QH_IDX (rel 1061).
- The bug hunt is documented because it generalizes: the kernel writes the q8_1 d8
  chunk table with the lane phase 2*c instead of the chunk index 2*cc (every chunk got
  the d of block 6) - found with a register dump mode; the dump itself initially
  reported garbage because all 16 blocks wrote the same dump buffer. Two lessons kept
  for the next kernel: single-block dump launches, and per-lane dump slots.
- block_q6_K field order is ql, qh, scales, d (210 bytes) - the harness and kernel use
  the struct fields and are unaffected, but hardcoded-offset replications get it wrong
  (cost one debugging round).

## What this means for the remaining round-13 plan

- Phase 1-3 (integration behind GGML_CUDA_MVQ_MMA8, stage_b gate +3%, PPL gate) are
  **not reached**: there is no kernel to integrate that beats the current path.
- The masked expert-hit path (mul_mat_vec_q_moe, 5.5 ms/round, "the same int8-mma tiles
  with the expert-id gather") inherits the same verdict: the feed problem is identical,
  the gather makes it strictly worse. Recommendation: do not build it.
- The realistic conclusion for the decode path on this machine: the dp4a MMVQ at
  ~300 GB/s (batch 4) is at its structural ceiling for this tile shape and block layout;
  the remaining headroom on this GPU is in the ~50% CPU-shadowing and the cache/schedule
  side (rounds 1-12), not in the quantized kernel family.

## Artifacts

- `bound/mma8_bench.cu` - both structures, the check/bench/dump modes, the mutation cuts.
- `bound/mma8_probe.cu` - the fragment-layout probe (LAYOUT 1..5; 1 is correct, 128/128).
- Binaries are gitignored (bound/.gitignore).
