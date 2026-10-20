# Round 13: int8-MMA grouped matmul for the multi-token quantized path

Machine: RTX 2080 Ti (sm_75, 68 SMs, INT8/FP16 tensor cores, no BF16), CUDA 12.0. Worktree
`/opt/llm-gateway/data/build/llama-vibe`, branch `vibe-parity` (round-1 final stack deployed
in configs). Read first: `NOTES-vibe-round1.md`, `bound/FACTS-bf16-as-f16.md`, the c-cpp-cuda
skill references (`/opt/llm-gateway/.agents/skills/c-cpp-cuda-expert/references/cuda-turing.md`),
and the upstream AGENTS.md rules (no AI-written PR descriptions; ASCII only; one-line comments).

**Budget: 2-3 days wall clock. About 60% is coding/CPU work that runs under foreign load;
the timing measurements (~1 h per cycle) need quiet windows - stage_b and the bound harness
gate themselves.**

## Why (the measured basis, cite it)

- The dense-path bound (`bound/mmq-bound.sh`, `bound/ab-libs.sh`, quiet, 3 rounds): the
  MMVQ batch-4 path runs at ~300 GB/s where batch-1 runs at 440-466 GB/s, and forcing the
  MMQ path (`GGML_CUDA_FORCE_MMQ`, already in the tree) is not better at batch 4-5. The
  kernel is INT-ALU saturated at multi-token batches (4 dots per weight byte), not
  bandwidth-bound: the 2080 Ti's `dp4a` throughput is the ceiling. Only the INT8 tensor
  cores (`mma.sync.aligned.m16n8k8`, ~4x the dp4a rate) lift it.
- Strata's trace: their whole quantized matmul family (dense + grouped expert hits) runs in
  2.1 ms per round (dequant 1.29 + mul_mat_q 0.81) where ours runs ~15-17 ms. The difference
  is the `mul_mat_q` approach: grouped tiles through tensor cores.
- The bound: `mul_mat_vec_q` (742 ms/request) + `mul_mat_vec_q_moe` (328 ms, the cache hits)
  at a realistic 2x -> **~9 ms/round** (nsys-inflated numbers, ~7.5 plain). This clears the
  5% build threshold with margin IF the kernel reaches >= 400 GB/s at batch 4.

## The target shapes (measure them again first, `bound/mmv_bench`)

| tensor | type | m x n x k | current (tuned table) |
|---|---|---|---|
| attn_qkv | Q6_K | 10240 x 4..5 x 2560 | ~76 us, 282-300 GB/s |
| ssm_out | Q6_K | 2560 x 4 x 6144 | ~47 us, 274 GB/s |
| attn_gate | Q4_K | 6144 x 4 x 2560 | ~41 us, 216-223 GB/s |
| shexp gate/up | Q4_K / IQ4_XS | 640 x 4 x 2560 | ~12-14 us, 61-75 GB/s (latency-bound) |
| expert hits | IQ2_S/IQ4_NL masked | ~860 applications/round in mul_mat_vec_q_moe | 5.5 ms/round |

## Phase 0 - bound and prototype standalone (day 1, CPU + 1 quiet hour)

1. Re-measure the shapes above with `bound/mmv_bench` (3 rounds, quiet) on the current tree -
   these numbers move with every round; the table above is from round 1.
2. Write a STANDALONE prototype harness `bound/mma8_bench.cu` (skeleton: allocate q6_k weights
   + F32 activations, run the prototype kernel, verify against `ggml_cpu` reference on the
   host, then time with CUDA events; 50 iters, 3 rounds, interleaved with the current path
   via LD_LIBRARY_PATH). Copy the check style from `bound/check_mul_mat_bf16.cu`.
3. Prototype: dequantize Q6_K blocks to int8 on the fly (the q6_k quant values are 0..63 with
   per-block scales; int8 needs the scales handled OUTSIDE the mma - see the existing
   `ggml/src/ggml-cuda/mmq.cu` which already does exactly this for its int8-mma path on
   newer archs; on sm_75 the same `mma.sync.aligned.m16n8k8.s32` works). Fragment layout:
   16 rows x 8 cols x 8 k per mma; batch-4 tokens fit one 8-col tile. The scale handling:
   dequant to int8 with a per-block or per-64 scale folded into the activation quantization
   (the q8_1 activations are already quantized per 32 elements - reuse their scales).
   TARGET: >= 400 GB/s on 10240x4x2560 in the standalone, bitwise-verified.

## Phase 1 - correctness (day 1-2)

- Exact-tolerance check against `ggml_cpu` (nmse < 1e-6 for the SAME accumulation order is
  NOT possible with mma - the tile order differs. State the accumulation-order argument,
  then prove by: full-range random blocks, all lengths 1..10 tails, scale corner cases
  (Q6_K has a 6-bit value grid; enumerate the 64 values x extremes of the d/min scales),
  and the greedy-decode hash comparison end to end).
- Mutation cuts that must fail the check: a sign flip in the int8 conversion, an off-by-one
  in the 64-value grid, a dropped scale, a single 1-ulp leak (the round-1 check programs
  in `bound/` are the template).
- `test-backend-ops test -b CUDA0 -o MUL_MAT` with the switch on: full matrix must pass.

## Phase 2 - integrate (day 2)

Behind `GGML_CUDA_MVQ_MMA8=1` (off by default; off = the existing MMVQ path untouched):
- dispatch in `ggml_cuda_should_use_mmvq` (or beside it): use the new kernel for Q6_K/Q4_K
  at ne11 2..8 on cc >= TURING when the switch is on; fall back for every other shape/type;
- the masked expert-hit path (`mul_mat_vec_q_moe`) is a SECOND, larger win (5.5 ms/round):
  the same int8-mma tiles with the expert-id gather in front. Only do it after the dense
  path is proven; it shares the conversion kernels.

## Phase 3 - measure (day 3, 2 quiet hours total)

- `bound/mmv_bench` A/B (libs side by side as in round 1: `bound/libs-*`), 3 rounds quiet.
- stage_b: `q-vibe-async` (control) vs `q-vibe-mma8` (same stack + the switch). Three
  interleaved rounds, quiet gate. Hashes and draft counts compared (cache-state effects on
  near-ties are known - explain every difference).
- Perplexity: test-de 12 chunks, ctx 512, `-ub 4` and `-ub 512`, within +0.5% of control.
  The int8-mma accumulation order differs - if PPL moves at all, report the direction; the
  gate stays +0.5%.

**Gate: +3% on both code and english.** Expected from the bound: +4-5%.

### Stop rules

- Stop after phase 0 if the prototype cannot reach 350 GB/s at batch 4 on the target shapes
  (then the ALU ceiling is NOT the binding constraint after all - profile it with `ncu
  --set full -k <kernel> -c 5` and report what is).
- Stop if the int8 conversion overhead (reading q6_k, expanding to int8, scales) eats the
  mma gain in the standalone - the prototype must include the conversion, not just the mma.
- Do not touch the MMQ path for larger batches (it wins there already, measured).

## Deliverables

`NOTES-round13.md` (English, every number with its source), the patch, the prototype and
check programs in `bound/`, the ncu profile files named, no push, no live settings change.
German one-liners in `/opt/llm-gateway/data/moe-quant/NOTE.log` per phase.

## Upstream note

The kernel change is self-contained in ggml-cuda (new file + dispatch), generic for Q6_K/Q4_K
on sm_75+ - this is upstreamable as "cuda: int8-mma matvec for small multi-token batches on
Turing" once the owner has reviewed and rewritten the PR description themselves (AGENTS.md:
AI must not write the PR text).
