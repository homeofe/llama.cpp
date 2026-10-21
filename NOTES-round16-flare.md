# FLARE and the sm_75 hybrid stack

Inspected 2026-10-21 at NVIDIA/flare-local-inference commit `47581ccac82660953a5b59cb04b2750d64edc97d`. Source review only: FLARE was not installed or benchmarked on this host.

## Decision

Transfer the accepted-history MTP catch-up design first. A full FLARE port is a different backend project, and does not by itself solve our expert-weight capacity problem. There is no measured speedup from FLARE on this machine yet.

FLARE's [reference memory layout](https://github.com/NVIDIA/flare-local-inference/blob/47581ccac82660953a5b59cb04b2750d64edc97d/docs/core-design.md#L655-L668) places 59.94 GiB of routed experts on the GPU; all GPU-bound stored tensors total 66.82 GiB before MTP, workspaces or caches. Its 26.82 GiB host/disk payload is PLE embeddings, not a CPU-computed MoE expert cache. Our GPU has 11 GiB. FLARE's recipe therefore cannot be our deployment configuration even after kernel compilation succeeds.

## Concrete source comparison

| Mechanism | FLARE | Our branch | Next useful step |
|---|---|---|---|
| Accepted-history absorption | `PreparedMTP(cache_only=True)` runs embedding, feedback and attention state writes; returns before attention output, experts and vocabulary projection | `common/speculative.cpp:1660` builds catch-up batches with no requested logits, but `src/models/qwen4exp.cpp:615` constructs the attention/FFN residual path and expands `h_nextn` | Measure attention-output/FFN work, then add a cache-only MTP graph if its removable time clears 3% end-to-end |
| QSA state maintenance | Cache-only attention still writes KV and runs QSA in cache-only mode | Raw and pooled indexer keys are persistent graph side effects | Preserve raw keys, pooled keys, positions and rollback; skipping the entire attention builder would be wrong |
| Shared/routed experts overlap | Separate CUDA stream and disjoint scratch regions | Shared-expert overlap already exists in our stack | Check the actual trace before proposing another overlap patch |
| Prepared graphs and scratch | Python prepares graphs/layouts; Rust serves them | llama scheduler creates many splits when GPU experts wait for CPU misses | Trace graph boundaries and host waits; FLARE's all-GPU graph cannot remove true CPU routing dependencies by itself |
| PLE host delivery | Produce lookup IDs early and consume delivered rows later | Host PLE and overlap are already present in our/Strata paths | Verify remaining critical-path waits rather than counting the same optimization twice |

Primary sources: [MTP lines 130-235](https://github.com/NVIDIA/flare-local-inference/blob/47581ccac82660953a5b59cb04b2750d64edc97d/ignition/recipes/qwen38/mtp.py#L130-L235), [attention lines 322-340](https://github.com/NVIDIA/flare-local-inference/blob/47581ccac82660953a5b59cb04b2750d64edc97d/ignition/recipes/qwen38/attention.py#L322-L340), [expert overlap](https://github.com/NVIDIA/flare-local-inference/blob/47581ccac82660953a5b59cb04b2750d64edc97d/ignition/components/expert_overlap.py).

The current draft context enables masked next-token embeddings (`common/speculative.cpp:1555`). Catch-up requests zero output rows, so the vocabulary projection can already be an empty operation. Do not credit the full head cost as a new saving. The upstream graph visitor still traverses the nonempty parents of empty gathers (`ggml/src/ggml.c:7277`); the unmasked FFN computation before the output-row gather remains the concrete candidate. QSA selection is also explicitly expanded (`qwen4exp.cpp:857`) even though cache-only absorption only needs state maintenance.

## Small port that can answer the performance question

1. Instrument catch-up separately from the recursive proposal and target verification. Include CUDA graph nodes in nsys. Compute an upper bound from the removable MTP attention-output/FFN and selector work; do not use the invalid old Strata trace that omitted graph nodes.
2. Add an opt-in Qwen MTP cache-only graph for absorption batches. Keep token/target-hidden alignment, feedback normalization, KV writes and QSA indexer state. Full proposal rows still run the ordinary graph. Do not change target-model math or expert-cache policy.
3. Compare the following full draft step's logits/residuals and final token IDs against the existing graph. Cover accepted lengths 0 through 5, partial rejection, rewind, context boundaries and prompt absorption. Deliberately corrupt a KV/indexer write to prove the check detects it.
4. Measure base/candidate interleaved on the same packed expert profile and exact Strata draft weights. Adopt only with a repeatable gain and matching output, or the affected quality panels when numerics change. The completed native Strata reference remains the target; the interrupted aligned-reference block is invalid.

This is an implementation plan, not a claim that the cache-only patch has been implemented. It does not predict future-layer expert IDs and therefore does not directly solve the entire CPU-expert wait.

## What a complete FLARE sm_75 port would require

- An alternative to the `b12x`/Blackwell expert and projection backends. NVFP4/MXFP8 native arithmetic is not available on Turing. Use existing ggml weight-only quantized kernels, or dequantize tiles to FP16 for Turing Tensor Cores; do not expand the complete model to FP16. Turing INT4 tensor operations are not FP4 floating-point operations.
- BF16-aware numerical handling. BF16 storage and software conversion can work on sm_75, but native BF16 Tensor Core arithmetic requires newer hardware. Blind BF16-to-FP16 conversion changes range and rounding. [NVIDIA capability tables](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html).
- Compatible QSA, GDN, PLE and gated-residual kernels, scratch plans and graph capture. The code also contains an SM103 path, but no Turing backend was found.
- A compatible toolchain/dependency matrix: current `pyproject.toml` pins Torch 2.14.1+cu132, CUDA Python 13.3.1 and a b12x-enabled FlashInfer revision. Our checked build environment is CUDA 12.0/sm_75. Changing a capability check does not supply missing kernels.
- A CPU/GPU expert execution and cache system for models larger than VRAM. FLARE's documented GPU-resident experts do not supply that part of our workload.

Retain FLARE's Apache-2.0 license and notices for copied code. No external implementation has been copied into this branch by this review.
