# Round 15: compare the executed Strata paths before selecting the next patch

2026-10-21. This is a source/trace audit and a small n5 screening, not a new throughput result. The round-14 stack remains the measured winner. No engine source, live model settings or runtime was changed.

## Result and priority

The statement that only more VRAM or speculative expert routing can close the remaining gap was too strong. Strata has relevant software differences that the previous comparison did not isolate: a smaller draft output head, fully resident compressed draft experts, and one verifier CUDA graph spanning CPU expert service through mapped flags. The old Strata GPU trace also omitted graph-node activity. Its reported 3.44 ms/round is not a total GPU-work measurement.

The first bounded candidate is a reduced draft vocabulary. It has a measured cost in our existing trace, saves memory at the same time, and can reuse llama.cpp's existing DFlash draft-to-target mapping pattern. A whole-verifier scheduler rewrite is a later, larger experiment whose exposed time must first be measured correctly. The closed CPU kernels, target quantized-matvec tile experiments and cache-policy sweeps stay closed.

## Versions and evidence

- Ours: branch `vibe-parity`, base `6c8bfda89`; final config `q-vibe-async`. The last commits `6c8bfda89`, `7954e8c53`, `3438498f5` are handoff/measurement records, not additional kernel gains. The active source stack includes `dda498af9` (async uploads), `91d3674b1` (elementwise fusions), `a9334d269` (Turing MMVQ launch), `18d7a133b` and `692c789e4` (BF16/F16 path).
- Measured Strata: `/opt/llm-gateway/data/runtimes/strata/0.1.41-local-1cb045cb`. Source references below refer to this directory, which was only read.
- Local newer Strata checkout: `data/build/strata-src` at `f57c0adcb31c00f3f216550c004dac4e713a9f70`. Its existing modified `src/qwen35/q36_engine.cu` was left untouched.
- Upstream main inspected through GitHub: [9de04cd](https://github.com/Niko1221/Strata/commit/9de04cd44e927a9a3df6bbc0e94fc1be8b844ba4), a Control 0.1.46.3 version bump with the engine unchanged. Do not attribute newer engine features to the older measured binary.
- Reproducible numbers and source hashes: `task-logs/round15/compare_strata.py`, `strata-comparison.json`. The script reads GGUF metadata, the actual subset IDs and both archived SQLite traces. It never reads or prints an API key.

## Differences in the active paths

| Area | Strata 0.1.41, measured configuration | Our final stack | Decision |
|---|---|---|---|
| Draft output head | 106,299 selected token rows copied from the native target head; 212.9 MiB | Full 248,320-row Q8_0 head; 644.14 MiB | First candidate: retain Q8_0 and only reduce the rows, to isolate the vocabulary effect |
| Draft routed experts | All 512 in VRAM, Q2_0, 675 MiB | Q8_0 weights total 2550 MiB; 101 cache slots in a 512 MiB reservation, about 61.5% hits in this screening | Separate candidate; do not combine quantization and vocabulary changes in the first measurement |
| CPU/GPU synchronization | One graph launch per verify window; mapped doorbell and completion flags at each layer; final stream sync | ggml GPU/CPU graph splits, input copies/event waits; graphs within GPU splits | Source-confirmed architectural difference; exposed saving still unmeasured |
| Shared expert overlap | CUDA shared-expert side stream after routing doorbell | Existing GPU-tail/CPU split overlap | Both already overlap independent work; Strata does not eliminate the layer data dependency |
| Verify token groups | `--spec-split` opt-in; `split_ = false`; absent from measured args | No equivalent whole-window grouping | Inactive in the comparison; source labels it exact but slower. Do not implement based on the benchmark gap |
| Target expert cache | 1815 profile-prefilled slots, first-touch/no eviction | 1779 adaptive slots at 3640 MiB | Already studied; repeat-request hit rate is not novel-request policy superiority |
| CPU expert kernels | Persistent 23 workers plus host, 72 tasks/phase; native multi-token kernels where useful | Existing optimized CPU path; earlier multiplicity/thread-pool tests negative | No new evidence to reopen those axes |

Relevant Strata locations: `src/core/mtp.cpp:299` (all draft experts), `:452` (full-head diagnostic), `:604-650` (subset gathering), `tools/mtp_rt.py:1-30` (Q2_0 layout and dimensions), `tools/draft_vocab.py:1-23` (subset and language coverage); `src/core/verify.cpp:1907-2052` (launch, CPU service, final sync), `:1274-1473` (shared stream/flags), `include/strata/core/verify.hpp:437` (split disabled); `src/core/expert_cache.cpp:653` (no eviction).

Relevant llama.cpp locations: `src/models/qwen4exp.cpp:190` and `:613` (full output head and MTP projection); `src/models/dflash.cpp:128` and `:1078` (existing reduced-vocabulary load and scatter to full target logits); `ggml/src/ggml-backend.cpp:1862` and `:1942` (input synchronization and overlap split).

The native Strata head's 521,472,000-byte payload and 212.9 MiB subset correspond to Q6_K (210 bytes/256 weights). The old log's phrase "experimental native Q5_K head" is a misleading label; do not derive its type from that phrase. Keeping our Q8_0 head in the first candidate avoids conflating head quantization with the row subset.

## Bound for the reduced draft head

The round-14 graph-off trace contains exactly 180 one-token Q8_0 MMVQ launches with 248,320 output rows: 209.952965 ms over 60 complete decode rounds, or **3.499216 ms/round**. The target's head is a different type and four-token launch shape and is excluded from this count. `compare_strata.py` checks the full demangled kernel specialization, grid and count.

The actual `draft_vocab.bin` holds 106,299 unique valid IDs, 42.8073% of the vocabulary. Applying that selection to our existing Q8_0 rows would reduce head storage from 675,430,400 to 289,133,280 bytes: **368.40 MiB saved before mapping/graph overhead**. At linear row scaling the head-time saving is **2.0013 ms/round**. This is an estimate; it is not an observed end-to-end gain. Reproducing Strata's Q6_K subset as well would save 431.25 MiB relative to our current Q8_0 head, but would change another variable.

Extra target-cache capacity is a second possible benefit. Do not add a cache-speed estimate to the head estimate and present it as a measurement. Reserve the freed memory first, measure the head-only change, then explicitly spend it on target slots in a separate comparison.

Minimal implementation direction, not implemented in this audit:

1. Give an external MTP GGUF an optional `d2t` tensor and a row-selected `output.weight`; preserve the tokenizer and the other tensors. Reuse DFlash's load/scatter pattern for qwen4exp MTP only, with unique/in-range IDs validated. Keep the no-mapping path identical.
2. Validate a full identity mapping against the old full head, plus a nontrivial subset: gathered rows must match the reference rows and excluded full-vocabulary logits must be negative infinity. A swapped-ID mutation must fail. Keep the target model and verifier unchanged.
3. Test the stored Strata subset without retuning cache budgets. Compare code, English and German acceptance, outputs and quality; this subset was trained around language coverage and is not automatically adequate for every language. Rejected drafts must still fall back to target generation.
4. Spend the measured head-memory saving on target-cache slots only after the first comparison. Adoption requires the existing +3% gate in BOTH code and English, three interleaved rounds/n=12 per category, output/count comparison and affected quality panels if outputs differ.

This is independent of the closed round-13 quantized-kernel tile experiments: it reduces the amount of model work and its resident bytes.

## The old cross-engine GPU comparison is invalid

`task-logs/vibe-round1/engine-shim.sh` captured CUDA graphs at the default graph level. Local `nsys profile --help` says this does not collect individual graph nodes. The old Strata SQLite has 356 `cudaGraphLaunch` calls, 8586 visible kernels (245.846 ms total), and zero graph-node IDs. The round-14 llama trace deliberately disabled graphs: 266,109 kernels, 2256.542 ms, zero graph launches. These kernel sums cannot be compared as complete work totals.

The old Strata 3.44 ms/round and 120 kernels/round claims in `NOTES-vibe-round1.md` omit graph work. Conclusions derived from the size of that GPU gap must be remeasured. Our own round-14 measurements and the separately measured final-stack throughput remain useful; they do not prove all software approaches have reached a ceiling.

`task-logs/round15/profile_strata.py` prepares an NVTX-bounded capture with `--cuda-graph-trace=node`, the measured Strata binary/configuration, a temporary server key, GPU hold/release, and the same code-duration prompt. The first attempt timed out at its 180-second quiet gate before acquiring the GPU or loading a model. Concurrent unrelated CI/browser processes drove load to 12-16; no new Strata trace or throughput result was produced. The script's capture path has therefore not yet been validated. No profiling process remains running.

When rerunning, separate wait/spin kernels from arithmetic and use interval unions rather than summing concurrent streams. A wait kernel represents CPU dependency time even though the GPU profiler marks it active. Compare graph-on node captures from both engines and record accepted tokens/window and prompt/decode boundaries. Profiler throughput itself is not the release benchmark.

## n5 correction and screening

The archived failure `data/moe-quant/qwen38/vibe-n45-screen/q-vibe-kv8n5.server.log:27-29` is **CUDA out of memory in cudaGraphInstantiate**, not a draft-depth assertion. With 256 MiB more headroom (target reservation 3640 -> 3384 MiB), n5 completed both short requests. No C++ assert fix was needed for this failure.

| Configuration | Code tok/s | English tok/s | Target slots |
|---|---:|---:|---:|
| n3, 3640 MiB | 47.37 | 41.26 | 1779 |
| n5, 3384 MiB | 50.01 | 38.82 | 1654 |

These are **n=1 per category/configuration**, sequential screening runs, not medians or an adoption result. Output hashes differ (code 218 versus 217 tokens; English 256 each). English regresses about 5.9%, so no general performance improvement is established. Runs began at load 1.87-1.96; measured foreign CPU consumption was 0.27-0.40 cores. The machine's load rose during our own inference. Duration: 07:13:13-07:19:30 UTC (6m17s), including starts, quiet waits and shutdown. Scripts, exact configurations, text outputs and raw rows are in `task-logs/round15/screen/` and `probe.py`.

## Routing speculation: narrow negative result only

`routing_bound.py` replays a causal predictor: the previous target verify window's expert union at the same layer. On 24,816 layer pairs (517 windows, warmup plus eight requests), recall is 41.60%, precision 41.50%, byte precision 41.59%. Identity/oracle and shifted-ID controls pass. The trace has no cache residency, so these are not miss-only rates and do not imply a speedup. This weak predictor does not justify a precompute implementation; it does not close better routing-prediction designs.

## Recent upstream changes: applicability matters

The Oct 17 verifier history includes [ba2c2d5](https://github.com/Niko1221/Strata/commit/ba2c2d5fb7666797840666acb62ed8a046ee9f13), which enables four measured switches specifically for sm_120 with fully resident experts. Those are not RTX 2080 Ti results. [75fabbd](https://github.com/Niko1221/Strata/commit/75fabbd94152f62ff787abcc7d861d0cc60b18f9) extends interleaved MMVQ to 5-8 columns; our n3 target verification normally has four rows. Neither commit explains the older measured binary's advantage or justifies reopening the closed Turing tile study by itself.

## Operational status

The Codex daemon repair completed separately: package selected, `update --from-cli --yes`, managed stop/start and repeated version/start checks succeeded; CLI, package and app server are all 0.163.0, backend `pid`. The gateway checkout and live settings are unchanged. GPU hold was released after the n5 screen; the later profile never acquired one. New analysis/scripts are local worktree changes and have not been committed or pushed.
