# FACTS: GGML_CUDA_BF16_AS_F16 and the MMVQ Turing table tuning

For the owner and for upstreaming. All numbers with their source. State: 2026-10-19/20, worktree
`llama-vibe`, branch `vibe-parity` (llama.cpp `l5-cache` at e3b15507c). Machine: RTX 2080 Ti
(sm_75), Threadripper 3960X, shared.

## Why (stage 1/2 measurements, task-logs/vibe-round1/)

- Our server round is 81.7 ms (code) against Strata's 47.2-47.8 ms. The entire per-token gap
  (22.5 vs 14.5 ms/tok) sits in the dense small-batch GPU path, not in the expert path: CPU
  experts, upload copies and cache hit rates are at parity (NOTES-vibe-round1.md, gap analysis).
- The hyper-connection matrices are BF16; on Turing `ggml_cuda_should_use_mmvq` keeps every BF16
  tensor in the row-wise `mul_mat_vec_f` kernel for batch <= 8, so `hc_*_up` (m=10240, k=320,
  batch 4) runs at **85.6 GB/s / 76.6 us** while the same shape in F16 takes the fused tensor-core
  path (MMF) at **308-320 GB/s / 20.5-21.3 us** - 3.6-3.7x (`bound/series2.sh`, 3 rounds, quiet).
- The quantized dense matvecs (attn_qkv/ssm_out Q6_K, attn_gate Q4_K, shexp IQ4_XS/NL) run the
  MMVQ kernel at ~200-290 GB/s at batch 4-5; the launch table for batch 2-4 used 4 warps per
  block. A 6-configuration sweep (`bound/sweep.sh`) on the real shapes selects **2 warps**;
  the advantage grows from ~5-7% (quiet) to 30-60% under foreign CPU load (attn_qkv 300-341 vs
  182-207 GB/s at load 15-22), which matters because the server always runs its dense kernels
  alongside the CPU expert phase.

## What was built

1. `GGML_CUDA_BF16_AS_F16=1` (default off, off = bit-for-bit unchanged):
   `ggml/src/ggml-cuda/bf16_f16.cu/.cuh` converts BF16 weight tensors in place to F16 once (same
   element size, no extra VRAM) when every node of the graph consumes the tensor through
   GGML_OP_MUL_MAT; any other consumer (get_rows, mul, add, mul_mat_id, 3D experts, non-weights
   buffers, other devices) vetoes the conversion and keeps the old path. `ggml_cuda_mul_mat`
   reinterprets a registered src0 as F16 before the dispatch, so the F16 heuristics (MMF tensor
   cores on Turing) pick the fast kernels. The conversion runs during the graph warmup call, so
   it is never captured into a CUDA graph. MUL_MAT_ID (3D experts) is never touched.
   The conversion kernel is written as integer arithmetic because the hardware `cvt` path (and
   the fused bf16->f16 chain the compiler emits for `__float2half(__bfloat162float(v))`) flushes
   f16 subnormal results on sm_75 (measured: 2^-24 -> 0x0000 instead of 0x0001).
2. MMVQ launch table for Turing, batch 2-8: `calc_nwarps` 4 -> 2 (batch 2-4 joins 5-8). A pure
   launch-configuration tuning change in the same spirit as the existing "tuned on RTX 2080 Ti"
   entries. `GGML_CUDA_FORCE_MMQ=1` (off by default, measurement only) forces the MMQ path for
   bound experiments.

## Proof

1. Bit-exact conversion: `bound/check_bf16_f16` covers 1542 bit patterns (every exponent,
   subnormals, overflow, NaN) against an independent software reference that was verified against
   F16C on the host: **0 differences, PASS**. Mutation cuts, both detected: sign dropped in the
   subnormal branch -> 339 differences (FAIL, exit 1); tie rounds up instead of to even -> 616
   differences (FAIL); restored -> 0 differences.
2. mul_mat on the 4 hc shapes vs the CPU reference (`bound/check_mul_mat_bf16`): nmse 0
   (bitwise equal), with the switch off AND on.
3. Kernel speed (quiet, interleaved, 3 rounds): hc_up 76.7 -> 20.5 us (85.4 -> 320 GB/s).
4. MMVQ table correctness: 2,444 test-backend-ops MUL_MAT cases OK, 0 failures, identical count
   with the baseline library (the full suite OOMs on an 11 GB card in both libraries; the huge
   synthetic cases n>8M are excluded for both).
5. End to end (stage_b, 3 interleaved rounds, n=12 per config and category, 0 requeues,
   `vibe-tuned-e2e/`; sustained foreign load 7-11, so interleaved ratios are valid and all
   absolute numbers are ~4-5% below the quiet baseline):

   | config | code tok/s | english tok/s |
   |---|---:|---:|
   | q-vibe-control | 45.7 | 37.3 |
   | q-vibe-tuned (table only) | 47.7 (+4.4%) | 38.5 (+3.2%) |
   | **q-vibe-plus (table + bf16-as-f16)** | **48.9 (+7.0%)** | **40.5 (+8.6%)** |

   Both clear the pre-declared +3% gate; the package is the recommendation. A quiet confirmation
   run (`vibe-plus-confirm/`) is recorded separately.
6. Perplexity (test-de, 12 chunks, ctx 512, deterministic, reproduced twice in back-to-back
   pairs): **-ub 4 (the verify window): control 3.8181 -> tuned table 3.8363 (+0.48%, inside the
   +0.5% gate with no margin)**; -ub 512: 3.8200 -> 3.8200 (identical - the MMVQ path is not
   used at large batches, which confirms the mechanism). The bf16-as-f16 switch adds +0.07%
   (3.7963 -> 3.7990) at ub 4 on its own.

## Negative results (kept, with numbers)

- The MMQ path is not the answer for batch 4-5: `GGML_CUDA_FORCE_MMQ` A/B on the real shapes
  shows MMVQ already wins at our batch sizes; MMQ only wins at batch 8 (Q6_K 10240x2560:
  281.7 vs 211.6 GB/s).
- `rows_per_block` 1 instead of 2 for batch 2-8 (the exact batch-1 launch shape): no measurable
  gain under load (<2%), reverted.
- Cache policy knobs cannot close the hit-rate gap to Strata: replaying the real bench-8 routing
  trace (`bound/sim_policy.py`, 3 rounds) through a faithful model of our adaptive LFU shows
  our current decay=128/admit=32 is already the best of the swept family (43.7% cumulative,
  43.5% steady state); no-decay is WORSE (31.7% - fast topic changes need fast decay, and
  cross-round reuse is minimal because the 8 prompts are different topics). Strata's adaptive
  tier must be structurally different (their static profile scored only 10% on our trace in the
  round-4 analysis), so matching their 50-57% trajectory is a policy redesign, not a knob.

## Open items

- Deploy: the package needs the tuned libggml-cuda plus `GGML_CUDA_BF16_AS_F16=1` in the model's
  runtime environment. Re-check the ub4 perplexity pair on a quiet machine before flipping the
  live gateway (the +0.48% leaves no margin). No live setting was changed in this round.
- The remaining round-time blocks, in measured order: the CPU expert gate (~48-60 ms/round
  exposed; hit-rate policy research), the multi-token MMVQ kernel efficiency (300 vs 440 GB/s
  at batch 1, ~4 ms/round bound), the ~9 ms/round elementwise tail (2-3 targeted fusions along
  the existing dsv4-hc.cu pattern), and the 2.4 GB/request DtoD split copies (~2 ms/round).
