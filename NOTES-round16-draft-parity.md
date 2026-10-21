# Round 16: MTP draft footprint, normalization and selection policy

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

The initial Q2/Q6/GPU candidate failed to generalize. `confirm-clean/summary.json` contains a clean A B B A comparison, two rounds, **n=8 per category/configuration**. Code server medians are 50.5371 -> 50.6865 tok/s (+0.2955%); English +1.1188%. Streaming rates give -0.7863% code / +1.1179% English. All eight code output pairs match; all eight English pairs differ. The experiment was stopped after those four complete blocks because code gain was below 1%, well below the 3% adoption gate. This is an early negative selection result, not a completed three-round adoption proof. The supervisor therefore did not launch quality panels. It supersedes the encouraging two-prompt screen.

The acceptance loss has a measurable potential cost: for code-intervals the first round changes 110/120 accepted/generated to 107/129; code-roman changes 189/196 to 186/205. Restoring those accepted-token rates while retaining cheaper draft execution could exceed the 3% gate. This motivates a direct comparison with Strata's whole-hidden draft RMS norm; it does not establish that this normalization is the trained model's reference behavior.

An experimental, default-off `LLAMA_MTP_HNORM_GLOBAL=1` switch applies the existing RMS operation over the full 4*2560 hidden row before reshaping back to streams. It only changes the MTP graph. `check_norm.cpp` checks the full 2560/4 shape at 1/2/4/7 tokens, varying stream scales and weights, against a double-precision formula. CPU and CUDA pass; maximum CUDA absolute error is 3.27e-7. Selecting the wrong normalization mode produces 9,963 to 69,743 differing entries, so the check distinguishes the two formulas. The original isolated runtime remains frozen; this experiment uses `e2e/global/`, again with only libllama replaced. Raw eight-prompt pilot: `global-screen/`.

Two fallback files retain all original Q8 experts and only change the head: q8exp-q8subset (3578.17 MiB), q8exp-q6subset (3515.32 MiB). Every tensor/metadata readback passes. A first conversion was bounded too tightly at 30 seconds and stopped before validation; its unverified output was renamed and is never used by a benchmark. The converter now publishes a final GGUF name only after all readback checks pass. The successful conversion is `q8-expert-conversion-final.txt`. Configurations can spend the head savings on target cache in a separate test; none is adopted yet.

Pending: select a candidate with a positive full-prompt pilot, confirm it against the original runtime in three interleaved rounds over all eight stage_b prompts (n=12 per category), compare output hashes/counts and run affected code/tools/German/format panels against both versions. Require the existing +3% gate in both categories. No new deployment or Strata-parity throughput claim is justified before those results.

The remaining Strata verifier-graph/CPU synchronization difference is a separate measured design question. The old graph-node-incomplete trace cannot provide its saving. Closed target quantized kernels and cache-policy sweeps remain closed.


## Complete Strata reference configuration (2026-10-21)

The component screens above were not the requested full engine-configuration comparison. The new `strata-reference` experiment combines the actual Strata choices instead of inferring parity from Q2 experts alone:

- `strata-q6subset.gguf` (1331.928 MiB) copies all 34 source draft tensors into the llama representation, plus the draft-to-target map. This includes the native IQ4_XS embedding, Q8 dense projections, BF16 small matrices, converted norm weights, Q2_0 routed experts and native Q6_K head subset. `strata-q6subset-manifest.json` and `strata-weight-conversion.txt` verify all types, shapes, bytes, split/concatenated projections and saved tensors.
- `LLAMA_MTP_HNORM_GLOBAL=1` selects the same whole-hidden norm. The standalone norm checks remain valid, but neither the earlier norm-only pilot nor the single abandoned complete-weight request establishes a gain.
- `LLAMA_MTP_STRATA_REFERENCE=1` uses full-vocabulary softmax confidence (excluded head rows are -infinity), threshold 0.7, up to three MTP drafts, and the actual MIT-licensed Strata suffix drafter/policy with up to five lookup drafts. CLI n_max=5 reserves six-token verifier/rollback capacity; the MTP chain itself remains capped at three. The policy receives the preceding verifier/commit/catch-up/next-draft elapsed time and accepted-token outcome.
- `LLAMA_MTP_DENSE=1` uses dense causal attention in the MTP block, as this Strata version does. It requires context <=32768, so every test token lies inside Strata's default dense draft window. The target still uses QSA. Qwen's proposed reuse of draft QSA selections is a separate long-context experiment, documented in `NOTES-round16-qwen-architecture.md`.
- Target cache is frozen to the **same 1815 expert IDs** from the first ranks of Strata's profile. Their weight bytes total 3574.19 MiB. llama's allocation across separate quantization groups needs a 4032 MiB budget for those IDs (computed lower bound 4028.18 MiB); Strata packs them without this group-allocation slack. `make_profile.py` reproduces the translation and budget calculation. No adaptive replacement is enabled.

`e2e/reference-dense/` preserves every original runtime file except `libllama` and `libllama-common`, with the latter carrying the experimental sampling/policy. `reference-dense-binaries.json` checks every file and symlink. Build only `llama-common` and `llama`, never the server/UI target. This is a measurement branch, not a generic upstream-ready sampler or policy API.

The imported policy/suffix implementation retains the original MIT license and source hashes in `common/strata-reference/PROVENANCE.json`; only local include paths differ. Its original policy and suffix unit tests pass. `check_confidence.cpp` passes 12 cases on CPU and CUDA for 17, 106299 and masked 248320 vocabulary sizes, with double-precision probability references around 0.7 and elsewhere. CPU ASAN/UBSAN passes. Independently compiled wrong top-10 normalization and +1e-6 probability mutations return 1; restored source returns 0 (`confidence-cuts.json`). This proves the confidence calculation, not yet the complete server integration or model quality.

The matched native-engine harness `native_reference.py` sends the same chat requests with temperature 0, seed 1, thinking off, 256 tokens and prompt-cache reuse enabled. Each of eight prompts is sent once and immediately repeated. "Novel" means first use of that prompt in this engine process; it does not mean no shared prefix or an empty expert cache. Startup, source/config, full text, hashes, cache counts, draft counts and timing are archived. A contaminated block is reset and replayed in full. This is still a sequential pilot, not the three-round interleaved adoption test.

Actual Strata completed a clean block (attempt 1, after discarding attempt 0 for foreign CPU load). Server-rate medians and ranges, **four distinct prompts/category, one execution per phase**:

| Phase | Code tok/s median [min,max] | English tok/s median [min,max] |
|---|---:|---:|
| Novel | 69.15 [62.8,70.6] | 60.70 [57.5,64.0] |
| Immediate repeat | 73.65 [68.7,76.8] | 63.30 [63.0,65.4] |

Source: `native-reference/runs.jsonl`, only the completed Strata attempt 1. A long foreign-load interval required 1177.9 seconds of quiet waiting before its final request. The overall runner was stopped after this complete Strata block because its original deadline was consumed; the llama blocks must run separately. No cross-engine gain can yet be derived from this table. The old runner's shutdown POST returned HTTP 415; its owned-engine PID fallback stopped it and released the GPU. The harness now sends JSON with the required Content-Type.

Remaining engine differences include arithmetic/reduction order, draft K/V formats, dense projection concatenation versus separate sums, KV-only versus full-graph draft catch-up, first-window handling, prefix-cache implementation, and target scheduler/CUDA-graph boundaries. These are explicitly not made equivalent by sharing draft weights and policy.

## Completed aligned-engine pilot

The aligned llama.cpp engine completed all 16 novel/repeat requests in `aligned-clean-v2/`, attempt 1. Attempt 0 was discarded in full after foreign CPU averaged 1.208 cores during its second request. The successful block starts with a fresh process and the same warmup. Each result below is the median [minimum, maximum] across four distinct prompts, one execution per phase. Output SHA256 values, token counts and accepted/offered drafts are retained in the summary and raw rows.

| Phase | Code tok/s | English tok/s |
|---|---:|---:|
| novel | 41.68 [40.35, 43.84] | 37.37 [33.86, 38.15] |
| repeat | 42.30 [40.57, 45.15] | 36.93 [34.09, 37.84] |

This is a sequential engine-reference pilot, not an adoption comparison against the original stack. It does not justify deploying the complete reference configuration. The original Q2/Q6/GPU-only candidate also remains rejected on its separate interleaved evidence.

The native repeat attempt in `native-clean-v2/` is discarded: its first block was contaminated, and its restart reused the wrapper workspace after `/engine/stop` had persisted a stopped flag. The wrapper therefore started without an engine. The harness now creates a fresh workspace under `workspace/attempt-N` for each native start. This changes only benchmark isolation, not the engine or gateway.

A sample taken after the first aligned discard found `kcompactd0` at one CPU core during model reloading. This is an observation, not proof that it caused every earlier outlier. The original load gates remain intact. Before the next native attempt, `POSIX_FADV_DONTNEED` released cached pages of the now-unused stock-control Qwen3.6 and Phi model files; memory snapshots are archived. File contents and system settings were not changed.

The new native startup log explicitly selects **PROFILE, ranked by routing frequency, no eviction**, with 1815 entries pre-filled. The older generic first-touch explanation does not describe this measured profile configuration. Native prefill temporarily borrows 1600 cache slots and refills them afterwards (`generate.cpp:515`); identical selected IDs do not make the two prefill/refill schedules or routed-token sequences identical. The native head log also hard-codes the text "Q5_K" (`generate.cpp:4021`); tensor metadata and the 521,472,000-byte payload identify the actual Q6_K head. The draft-subset converter uses tensor metadata and readback hashes, not that log label.

## Fresh native block and remaining runtime gap

`native-clean-v3/` completed all 16 requests in clean attempt 0. Same four distinct prompts/category and novel/repeat protocol as the aligned block:

| Phase | Code tok/s median [min,max] | English tok/s median [min,max] |
|---|---:|---:|
| novel | 68.60 [62.4, 71.6] | 60.95 [59.3, 64.1] |
| repeat | 69.65 [68.1, 74.1] | 62.30 [61.8, 66.2] |

Novel code acceptance is 519/573 (90.58%) native versus 511/567 (90.12%) aligned; English is 617/766 (80.55%) versus 614/763 (80.47%). This points away from acceptance rate alone as an explanation of the large remaining runtime gap, but does not isolate an individual scheduler or kernel cause. Only 7 of 16 full outputs are byte-identical, despite matching prompt-token counts in all cases. No quality-equivalence claim is made.

`aligned-native-summary.json` retains medians, ranges, generated-token counts and acceptance totals. `aligned-native-output-comparison.json` retains pairwise token counts and hashes. These are sequential clean blocks, not interleaved adoption evidence. No new runtime is selected for the gateway.

The prepared `profile_engine.py` captures one warmed repeated request with `--cuda-graph-trace=node` and explicit NVTX capture. It labels profiler rates invalid for benchmark comparison. Analysis must distinguish arithmetic/copy intervals, wait-only kernels and true device idle; summing overlapping streams or interpreting a spinning wait kernel as arithmetic would misstate the bottleneck. REA's potential supporting role and release limits are documented separately in `NOTES-round16-diagnostics-tools.md`.
