# Round 16: reproduce Strata's smaller MTP draft in llama.cpp

2026-10-21. Work in progress; the screening below is not an adoption result. The live gateway configuration is unchanged.

## Question and implementation

The existing external MTP GGUF used Q8_0 experts and a full Q8_0 output head. That was the available draft configuration, not the result of a controlled comparison proving Q8_0 best. Strata uses 675 MiB of Q2_0 routed experts, all resident on the GPU, plus a 106,299-row subset of the native Q6_K target head. The verifier and target weights remain unchanged in this experiment.

`task-logs/round16/make_draft.py` copies Strata's actual packed Q2_0 weights; it does not approximate its quantizer. It splits the packed gate/up rows into llama.cpp's separate tensors and checks reconstruction, a swapped gate/up negative control, every tensor's readback SHA256 and unchanged metadata. Other draft tensors initially remain as in the existing Q8_0 GGUF. This is therefore parity for experts and head, not yet parity for every dense weight or normalization operation.

The source patch adds optional `d2t` support to the Qwen MTP graph using the existing DFlash pattern. The external file must be MTP-only, contain one MTP layer and be loaded with MTP enabled. Its reduced head projects only selected rows; a scatter restores full-vocabulary logits with negative infinity elsewhere. Loading rejects invalid or duplicate IDs. No-map files keep their original graph.

Models are local artifacts under `data/moe-quant/qwen38/round16-models/`; manifests and the converter are tracked instead of model payloads:

| Variant | File MiB | Difference |
|---|---:|---|
| q2 | 2070.76 | Only three routed-expert tensors changed from Q8_0 to Strata Q2_0 |
| q2-identity | 2072.65 | Same plus full identity `d2t`; original head unchanged |
| q2-q8subset | 1703.17 | Q2 experts plus selected original Q8_0 head rows |
| q2-q6subset | 1640.32 | Q2 experts plus selected native target Q6_K head rows |

The Q8 subset saves 368.40 MiB of head storage; Q6 saves 431.25 MiB. Mapping and graph overhead are additional. The original draft's dense weights and full token embedding are still present; the filename does not mean every tensor is Q2_0.

## Verification

- CPU and CUDA standalone projection/scatter checks: full identity and nontrivial reordered subset, 1/2/4/7 outputs, exact reference values and negative-infinity excluded logits. Clean and restored cases have zero differences; swapped IDs and a 1e-6 weight perturbation fail in all eight cases on both backends. CPU ASAN/UBSAN also passes. Sources/results: `check_head.cpp`, `head-cpu.txt`, `head-cuda.txt`, `head-sanitizers.txt`.
- Actual model loader: valid and restored maps load; negative, out-of-vocabulary and duplicate IDs return an error. `check_mapping.py`, `mapping-checks.json`, `mapping-*.txt`.
- The old runtime rejects the subset file with its original full-head shape requirement: `load-old-negative.txt`.
- Actual end-to-end identity-map control: q2 and q2-identity have identical text hashes, token counts and draft generated/accepted counts on code-duration and en-1. `identity-e2e.json`.
- All model tensor readbacks and the gate/up packing controls pass. `*-manifest.json`, `conversion.txt`.

## Measurement protocol

`run_drafts.py` uses the existing stage_b target configuration and prompts, temperature zero, seed 1, thinking off, no prompt-cache reuse, 256 output tokens. A short warmup precedes measured prompts. Every request waits for load1 < 2 and less than one foreign CPU core; foreign CPU use is measured again during generation. The runner holds/frees the gateway GPU and releases it on completion/error, with a bounded deadline. It never kills preflight jobs. Model loading, shutdown and the decay of our own inference load make wall time much longer than inference time.

The first screen isolates Q2 experts, the map, the head subset/type and residency, all at the existing target cache size of 3640 MiB (1779 slots). The original runtime was copied before building. Exact commands/environment and complete outputs are in each run directory. Record both client streaming rate and server decode rate: first-chunk batching can bias the former, as seen on the first baseline code request.

| Sequential screen, n=1 each | Code stream tok/s | English stream tok/s | GPU MiB after code |
|---|---:|---:|---:|
| Original Q8 draft | 47.34 | 41.40 | 10506 |
| Q2 experts, original full head | 46.77 | 40.90 | 10508 |
| Q2 + identity mapping | 46.05 | 40.82 | 10510 |
| Q2 + Q8 subset head | 47.17 | 42.38 | 10140 |
| Q2 + Q6 subset head | 48.80 | 44.24 | 10076 |
| Q2 + Q6 subset, all experts pinned in hybrid cache | 48.96 | 44.76 | 10240 |
| Q2 + Q8 subset, GPU-only draft | 48.56 | 43.28 | 10304 |
| Q2 + Q6 subset, GPU-only draft | 49.68 | 45.46 | 10242 |
| Q2 + Q6 subset, GPU-only draft, +256 MiB target cache | 49.60 | 45.01 | 10488 |

These are **screening values**, not medians, not directly comparable to the historical 51.3/42.7 aggregate across four prompts per category. Code output is identical across this screen. English outputs change with draft quantization/head type, so quality panels are required before selecting a candidate. Q2 alone did not improve end-to-end speed: reduced draft misses did not compensate for changed acceptance and the remaining execution overhead.

The GPU-only isolation removes `-cmoed`, adds `--spec-draft-ngl 99`, and sets `GGML_MOE_HYBRID_DRAFT_MIB=0`. This makes the draft experts regular GPU weights, eliminating the draft's masked CPU/hybrid graph rather than merely pinning its cache. Spending another 256 MiB on target slots (1779 -> 1905) did not improve this screen and changed English output again. Select `q6gpu`, keeping the existing target cache and more memory headroom. Raw results: `gpu-screen/`.

The final comparison uses two copies of the original runtime with **only libllama replaced** in the candidate, checked by SHA256 for every file and matching symlink targets (`isolated-binaries.json`). The initial full server build also refreshed build-version strings and automatically fetched/embedded UI assets through CMake's UI target (`build.txt`); those files are excluded from the isolated runtime. The final build uses `--target llama` only (`build-final.txt`). There are no source differences in src/ggml/common/tools between the original binary's source commit ada2198d6 and the pre-patch worktree 6c8bfda89. The final map-validation guard applies only to Qwen4exp, preserving other model loaders.

Additional source findings while the confirmation runs:

- Strata's `--spec 4` means a four-token **MTP** window (anchor plus three drafts), matching `--spec-draft-n-max 3`. Its default-on suffix/prompt-lookup drafter extends the maximum verifier window to six (`src/program/generate.cpp:2190-2195`, startup log `strata-bench/engine.log:40`). Thus the full speculation policy is not equivalent to our MTP-only setup. The old code-duration log reports 2-3 suffix windows per request, not a replacement of every MTP window. llama.cpp has ngram-first/MTP-fallback composition, but that is not Strata's selection policy; longer windows also require checking recurrent rollback capacity, not only changing a CLI argument.
- Strata's deployed `--spec-min-p 0.7` can shorten the chain, while our baseline has p_min=0. The probability definitions also differ (full subset probability versus llama.cpp's top-10 candidate normalization), so copying the numeric threshold would not make them equivalent.
- Draft backend sampling is already enabled by default in llama.cpp; CPU sampling is not an unenabled optimization.
- Strata's default whole-hidden RMS normalization and its `--mtp-hnorm stream` option remain distinct from our per-stream graph; do not silently change trained-model semantics to match an engine setting.
- Strata's catch-up computes the K/V-producing front for the verifier window but the rest of the draft only for the selected row (`src/core/mtp.cpp:1047-1085`). Our graph should be traced for redundant catch-up work before implementing a KV-only cut. No saving is measured here.

The old Strata throughput comparison is explicitly a warm-request reference, not a matched cold-request benchmark. Its harness retried rejected measurements without reloading; the server reused conversation prefixes. For example, the code-duration engine log has 98 new prompt tokens initially and then 91 reused plus 7 new. The first request was also foreign-load contaminated, so its 59.8 tok/s **must not be used as a clean cold baseline**. The reported clean 69.5+ tok/s values are valid for warm repeats, but cannot isolate the engine-only gap to our cache_prompt=false sequence. A future cross-engine comparison must match this state policy. This does not invalidate the new llama.cpp A/B comparison.

The first confirmation attempt (`confirm/`) was stopped after an en-1 foreign-load measurement of 1.37 cores. All of its partial results are excluded. The corrected runner records block attempts, discards the entire configuration block on contamination, reloads the model and replays the same warmup/prompt sequence. `analyze.py` admits only completed clean blocks from `confirm-clean/`; it never merges screening, partial or discarded blocks into the final result.

## Remaining validation and next decision

Pending: select the resident GPU candidate from the screen, confirm against the original runtime in three interleaved rounds over all eight stage_b prompts (n=12 per category), compare output hashes/counts and run affected code/tools/German/format panels against both versions. Require the existing +3% gate in both categories. No new deployment or Strata-parity throughput claim is justified before those results.

The remaining Strata verifier-graph/CPU synchronization difference is a separate measured design question. The old graph-node-incomplete trace cannot provide its saving. Closed target quantized kernels and cache-policy sweeps remain closed.
