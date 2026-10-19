# Patch: BF16 weights as F16 on devices without a fast BF16 path (Turing)

## Why (measured)

The dense-path bound (this directory, `bound/`): the qwen4exp hyper-connection matrices are BF16;
on the RTX 2080 Ti `hc_*_up` (m=10240, k=320, batch 4) runs the row-per-block `mul_mat_vec_f`
kernel at **85.6 GB/s** because `ggml_cuda_should_use_mmvq` keeps every BF16 tensor in the vector
kernel for batch <= 8, while the fused tensor-core path (mmf) supports only F16 on this hardware.
The same shape in F16 runs at **308 GB/s** (3.6x). Per layer the four hc matrices cost 229 us of the
453 us dense budget; x48 layers that is 11.0 ms of the 81.7 ms round. Stage_b with the existing
IQ3_S-hcfix file (hc up/routers F16, hc down Q8_0) measures the effect end to end
(`q-vibe-hcfix` vs `q-vibe-control`, this round).

## What

`GGML_CUDA_BF16_AS_F16=1` (off by default, bit-for-bit unchanged when off):

- `ggml/src/ggml-cuda/bf16_f16.cu/.cuh`: an in-place bf16 -> f16 conversion kernel (same element
  size, no extra VRAM), a registry of converted data pointers, and a graph survey. The survey runs
  before every graph compute and converts a BF16 2D contiguous weight tensor of this device only
  when **every node of the graph consumes it via MUL_MAT**; any other consumer (get_rows, mul,
  add, mul_mat_id, anything else) vetoes the conversion - those tensors keep the old path.
- `ggml-cuda.cu`: `ggml_cuda_mul_mat` reinterprets a registered BF16 src0 as F16 (a scratch tensor
  with type F16, same ne/nb/data) before the dispatch, so the F16 heuristics (mmf/mmvf/mmq/cublas)
  pick the fast tensor-core kernels. MUL_MAT_ID is untouched (3D experts are never registered).
- Conversion happens during the graph warmup call, which always executes directly, so the
  conversion kernel is never captured into a CUDA graph; replays see the registry hit only.

## Numerics

bf16 -> f16 is exact for f16-normal-range values (mantissa 7 -> 10 bits); values below 2^-14 round
to f16 subnormals, |x| > 65504 becomes inf. Model weights are far inside that range; the previous
offline retype measured perplexity within 0.25% (moe-cache NOTES section 6). Proven by the
standalone check (`bound/check_bf16_f16.cu`), test-backend-ops and perplexity below.

## Verification plan (skill discipline)

1. `check_bf16_f16`: full bit-pattern coverage vs an F16C host reference (exact in-range, inf
   overflow, subnormal rounding); mutation cuts: sign drop and a 1-ulp leak must fail the check.
2. `test-backend-ops test -b CUDA -o MUL_MAT` BF16 cases with the switch on: reference equality.
3. stage_b 3 interleaved rounds `q-vibe-control` vs `q-vibe-patch` (same original GGUF, the switch
   on only in the patch config), quiet gate, output hashes and draft counts compared.
4. Perplexity on test-de (12 chunks, ctx 512, -ub 4) within +0.5% of the control.
