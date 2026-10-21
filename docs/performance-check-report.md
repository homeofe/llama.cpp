# MoE performance audit

Date: 2026-10-21. Audited release baseline: `9a773611b80e5fe88a2bf7630c73f1b01fe721bb` from `homeofe/llama.cpp`, worktree `llama-performance-check`, branch `performance-check`. Requested upstream comparator: `def4d406ae2c2f39573120d68730fbb7760b24bf`, separately built in `llama-audit-upstream`.

This is not the gateway's deployed build, nor the `vibe-parity` experimental hybrid cache. The release uses `src/llama-moecache.cpp`; the vibe branch uses a different implementation in `src/llama-moe-cache.cpp`. Results and knobs must not be transferred between them without checking.

Hardware: RTX 2080 Ti, sm_75, 11 GiB; Threadripper 3960X, 24 cores / 48 threads; 128 GB RAM. CUDA 12.0. CPU and CUDA builds were made locally; no dependencies or models were downloaded except the documented GLM MTP tensor ranges. External source/PR pages were read for verification.

PASS means the stated, scoped behavior is supported by code and the named test. FAIL is an observed contradiction, including documentation claims stronger than the evidence. N/A means unverified here or unavailable hardware, not a successful test. Line references point to this audit worktree after the fixes. Raw evidence is under `tools/bench/performance-check/`.

## 1. Cache selection and off switches

| Check | Status | Evidence and limits |
|---|---|---|
| Automatic cache when an MoE does not fit | PASS, scoped | `common/common.cpp:1481` selects CPU experts plus automatic GPU cache when combined model file size exceeds free VRAM minus 1 GiB per GPU. Real Qwen IQ3_S: 77.9 GiB model, 10.4 GiB free, cache selected automatically, 48 layers with 56 slots each (`mtp-integration/qwen-plain.txt`). Dense models, CPU-only, explicit placement and prompt-heavy placement comparison are exceptions to the README's formerly absolute wording. |
| Model fits: cache stays off | PASS tested behavior; global guarantee unproved | Size check at `common/common.cpp:1526`; tiny MoE actual GPU response has `active=false` in `fixture-after-off-fix/fits.json`. Dense CPU control also keeps it off. The selector is a file-size-plus-reserve heuristic, not an exact proof that every borderline model cannot fit; lazy/offloaded tables also affect that distinction. A runtime ratio of exactly 1.0 is not established by a placement test. |
| `--moe cache=0` and `--moe-expert-cache 0`: no cache/tuner/placement trials | FAIL -> PASS after fix | `options.cpp` tests both flag orders including later `-at on`. Fixed `common_autotune_on` (`common/common.cpp:1405`), stock early return, speculative auto settings, predictor/prefetch parameters, thread-pool expansion and server depth tuner. Real dense model previously loaded saved 2/2 threads instead of 24/24; now remains 24/24. Fixed cache-zero GPU control passes. |
| Literally all fork code becomes historical upstream | FAIL | A flag cannot remove compiled CPU/ggml kernel changes. README itself reports a CPU-only Q2_0 kernel speedup. The documented guarantee is now cache/tuning off and stock placement, not binary or numerical identity to `def4d406a`. No global kernel revert was attempted. |
| `LLAMA_MOE_AUTO_MODE=stock\|cache` | PASS by inspection | `common/common.cpp:1534` acts only in automatic oversized-MoE placement. It does not override explicit `-ngl`/tensor placement or force a cache on fitting/dense/CPU-only models. |
| `LLAMA_MOE_STATE=0` | PASS | `src/llama-moestate.cpp:24` returns before directory creation or I/O. Fresh-process C-API check and strace find zero state-file syscalls; existing file bytes and mtime unchanged. `check_state.py`, `state.txt`. |
| `LLAMA_AUTOTUNE=0`, `LLAMA_ARG_AUTOTUNE=off`, `LLAMA_ARG_MOE=cache=0` | FAIL -> PASS | Legacy environment previously beat explicit `-at on`. `autotune_set` now tracks explicit parser settings; `env-cli` tests precedence. Cache zero remains dominant. `common/arg.cpp:2552`, `common/common.cpp:1405`. |

## 2. State and warm-up

| Check | Status | Evidence and limits |
|---|---|---|
| Default INI path and persistence across processes | PASS | `src/llama-moestate.cpp:24` uses XDG cache root or `$HOME/.cache/llama.cpp/moe-state.ini`; explicit `LLAMA_MOE_STATE` path supported. C-API fresh-process tests preserve hot counts, tuned values and placement choice. Actual tiny MoE writes hot expert counts after 80 output tokens and loads them on restart. |
| Hot experts, tuned settings, placement choice are consumed | PASS, scoped | `src/llama-moecache.cpp:562`, `:683`, `:2498`; `common/common.cpp:1342`, `:1534`. Placement records are only produced for workloads that need comparison; a plain oversized short-chat cache selection need not write a placement trial. |
| Second run reproduces 25.2 -> 31.5 t/s | N/A | These are machine D's 4x3090 / EPYC numbers. Its raw logs were not retained; this host has one 2080 Ti. State persistence is proved; the advertised magnitude is not a guarantee and was not reproduced here. |
| Deleting the INI resets learned state | PASS | Fresh process after deletion cannot retrieve the saved keys (`check_state.py`). This resets learned state, not OS page caches or explicitly set options. |
| State off never reads/writes the INI | PASS | Real C API under strace, above. All model tests use private state files or state off; the owner's INI is untouched. |

## 3. Autotuner

| Check | Status | Evidence and limits |
|---|---|---|
| Start from cores minus one per GPU | PASS, scoped | `common/common.cpp:1610` subtracts GPU count from automatically selected CPU counts, floor 2, only on the automatic oversized-MoE path. The subsequent bandwidth probe can lower the starting count to measured RAM saturation (`src/llama-context.cpp:1346`). |
| Step thread counts and reject non-improving settings | FAIL as literal speed-only guarantee | Paired three-slice comparison, noise rejection and step candidates exist (`src/llama-context.cpp:1433`). More threads must improve by a noise/core-cost threshold. Fewer threads are accepted within 1% slowdown (`:1497`). This is an intentional resource tradeoff, not strict throughput maximization. README now states it; changing the tuning policy requires a controlled performance experiment. |
| Explicit `-t N`, `-tb N` never auto-tuned | PASS | Parser preserves `auto_threads=false` and `threads_batch_set`; tuning enable checks at `common/common.cpp:1846`. Real dense run with poisoned saved state remains 7/9. Parser also preserves explicit `-ub 256` and FA. |
| `-at off`: cache runs, no tuning state | FAIL -> PASS | Before: real GPU cache loaded saved settings and rewrote the INI. Fix gates profile access and automatic placement-state reads by tuning state. After: four-slot cache works, no saved-settings load, INI bytes/mtime unchanged. `fixture-before-off-fix/autotune-off.txt`, `fixture-after-off-fix/summary.json`. |
| `-at off` performs no measurement whatsoever | FAIL documentation claim, clarified | Cache hit statistics and startup bandwidth probes support the operational cache and still run (`src/llama-moecache.cpp:1210`, `:2555`). Tuning trials and learned state are disabled. README/help now describe this distinction. |
| First-run cache versus stock for prompt-heavy work | PASS implementation; N/A speed reproduction | `common/common.cpp:1364`, `:1534` estimates prompt/output lengths at startup, alternates placements across starts and normally measures two of each. A 3% estimated cache win can decide after one each (`:1393`). Server defaults use estimated 2000 prompt / 600 generation tokens, not future HTTP request contents. It does not load and time both modes in one startup. |

## 4. Flags and manual precedence

| Check | Status | Evidence and limits |
|---|---|---|
| `--moe-expert-cache -1` sizes from free VRAM | PASS | `src/llama-moecache.cpp:2019` subtracts margin and auxiliary allocations; actual fixture chooses 7 of 8 experts per layer. Auto sizing is called after context/compute buffers, not simply file bytes divided by VRAM. |
| `--moe-expert-cache N` fixed slots per layer | PASS | `src/llama-moecache.cpp:2023`; actual `N=4` gives 4..4 slots. Only host-resident eligible layers are cached; it does not move GPU-resident experts back to CPU. |
| Compound keys and tuning knobs | PASS | `common/arg.cpp:2560` maps cache, prefetch-slots, inserts, window, predict, train; `src/llama-moecache.cpp:466` includes MARGIN, GATE, WAIT, BIG, SWAP_FRAC. Known numeric keys use `atoi`; unknown knobs are passed to the engine, so this is not strict general input validation. |
| Explicit knobs across model reloads | FAIL -> PASS | The self-tuner cached its first model's candidate list in a process-static variable. Reloading after tuning off kept it disabled; reloading with explicit MARGIN=123 could reset it to zero. A harness invoking the actual production `self_tune` reproduces both failures. Candidate lists now belong to each cache instance; both controls pass. |
| Uppercase explicit tuning value wins | FAIL -> PASS | Automatic margin formerly appended lowercase `margin_mb` after explicit `MARGIN_MB=501`, overriding it. The auto-wrapper now checks case-insensitively; actual model initialization preserves the sole explicit value (`margin-before.txt`, `margin-after.txt`). |
| `--load-mode pin\|mmap` and server default | PASS flags; FAIL former scope claim | Parser maps pin to NONE, mmap to MMAP (`common/arg.cpp:2754`). Automatic pinning is limited to the server's automatic oversized-MoE cache path when weights fit available RAM (`common/common.cpp:1598`). Fits/dense/manual-placement paths keep stock defaults. README/help corrected. |
| README settings table matches parser | PASS after corrections | Table now describes cache-zero scope, pin scope and environment precedence. `options.cpp` checks concrete parsed fields, not string matches of documentation. |

## 5. MTP

| Check | Status | Evidence and limits |
|---|---|---|
| Upstream PRs #28243 and #27917 merged | FAIL literal statement | GitHub inspection: #28243 closed without upstream merge; #27917 open/draft. Their work is integrated into this fork: Qwen `de4b7e1ac`, GLM ports `720a2f067` / `dc1d42b84` are ancestors of the release baseline. README says integrated, not upstream merged. |
| Qwen external `-md ... --spec-type draft-mtp` | PASS | Actual Qwen IQ3_S plus existing shared Q8_0 head, explicit depth 3, 64 output tokens: 45/53 drafted tokens accepted (84.9%). Text SHA256 `2106db7d2024ac2ddd328517bf31e26b88b227541940781b09c5390c0809650b` matches plain decoding. CPU draft placement is a correctness control, not a throughput recommendation. |
| GLM Q4_K head with max depth 3 | PASS load/request; FAIL universal output identity | Live extracted 4.26 GiB head loads and generates 64 tokens; 43/59 accepted (72.9%). Output differs from plain decoding. The second no-draft control also differs from the first, so the test does not isolate MTP as the cause. Dynamic CPU/GPU cache placement and batch arithmetic remain possible causes, not proved explanations. |
| Oversized model skips external draft without explicit max | PASS Qwen and GLM | `common/common.cpp:1280`: with tuning enabled, >2x free VRAM, no explicit max, external draft is disabled before load. Both auto runs log "not loading it". Qwen returns identical 64-token output; GLM does not. Explicit cache-zero/tuning-off paths preserve the user's draft request. |
| Draft acceptance and identical output | PASS Qwen; FAIL GLM identity in this test | Qwen 84.9% and identical text; GLM initially 72.9%, then 46/51 (90.2%) after removing unused embedded weights, both with different text from plain decoding. These loaded-machine functional tests do not establish speed, a universal 89%, or semantic equivalence. Full JSON and text are archived. |
| HTTP-range GLM MTP extraction | FAIL -> PASS | Fixed unchecked full-file 200 responses/wrong ranges, leaked `split.*` metadata, and counting an already-present NextN layer twice. Seven local HTTP boundary tests and two real-GGUF fixture tests pass. Live run fetches only `blk.45.*` from documented unsloth UD-Q4_K_XL source and writes a standalone head, about 4.3 GiB. No full trunk download. |

## 6. Diagnostics

| Check | Status | Evidence and limits |
|---|---|---|
| CLI `-lv 3` after each response | PASS by code | `tools/cli/cli-context.cpp:379` formats cache/hit/knobs/threads/batches; `:677` emits at INFO verbosity. Interactive terminal rendering was not separately automated. |
| Server status and hit rate | PASS real responses | `tools/server/server-common.cpp:100`, `tools/server/server-context.cpp:2226`; actual fixture and Qwen responses carry `timings.tuning.moe_cache`, server logs print cache on/off, request/total hit rates and settings. |
| Hit rate above VRAM fraction | PASS plain Qwen/GLM; FAIL GLM external-draft case | Qwen plain request: 47.6% request hits with 77.9 GiB model on an 11 GiB GPU (48 layers, 56 slots/layer). This is a short functional request, not a warmed benchmark. The tiny random fixture's >90% is not representative of real models. GLM plain reaches 20.4%, but external MTP drops to 3.6% and 1 slot/layer: the target needlessly loads its embedded MTP weights as well. The target now skips those unused embedded weights when an external draft is supplied on the cache path: 42 instead of 43 eligible layers, 2 instead of 1 slot/layer, 5.8% request hits. The head/verification still consumes memory; the low-hit condition is not fully resolved. This is capacity evidence, not an isolated speed comparison. D's 87.9-94.9% cannot be reproduced on this hardware; total file VRAM fraction and eligible expert-slot fraction are different denominators. |

## 7. Benchmarks and regression

| Check | Status | Evidence and limits |
|---|---|---|
| Run history includes every documented run with full provenance | FAIL | At baseline: 608 rows; 15 lack commit/build number, 182 lack full command. 469 blank args can represent defaults and are not all invalid. Several headline rounded values have no matching model/test row (`history-audit.json`). D explicitly has no raw logs/CSV. No missing historical rows were invented. |
| Upstream def4d406a vs current llama-bench: short/12k | PASS measured protocol | Complete clean A B B A A B cases, n=3 per build, identical Qwen3.6 Q4_K_M model/placement/flags. Short-context decode: 53.776 -> 55.359 t/s (+2.95%); 12k-context decode: 52.496 -> 53.381 (+1.69%); 12k prompt processing: 381.195 -> 380.878 (-0.08%). These are stock-placement regression controls, not cache gains. Raw v2/v3 and `stock-bench-summary.json` retain every sample. |
| Does llama-bench exercise the auto cache? | FAIL if assumed | It has its own parser and initializes model/context directly (`tools/llama-bench/llama-bench.cpp:1305`, `:2326`), default expert cache zero. Its comparison is a stock-placement regression control, not the README cache speedup. Real server tests exercise the cache. |
| Long-prompt 0.4-0.7x and >RAM mmap 20-30% tradeoffs | N/A on this host | No controlled >RAM model workload was created. Available Qwen/GLM models fit RAM; the no-download constraint precludes adding a larger model. README numbers remain historical observations, not reproduced current-branch guarantees. |
| Slow second GPU link | N/A | One GPU installed. No inference about a second PCIe slot from this machine. |
| Dense GPU and CPU-only exactly 1.0x | PASS near-parity on tested model; global exact guarantee unproved | Phi-4-mini Q4_K_M, three runs/build in ABBAAB order: GPU 163.057 -> 162.968 t/s (-0.05%); CPU 29.074 -> 29.102 (+0.10%). CPU control uses both `-ngl 0` and `-nopo 1`. Real dense stock model no longer consumes tuning state. These results do not establish exact parity for every dense model or explain a different historical Qwen workload. |
| README CPU-only 1.7 -> 1.6 explanation | FAIL as evidence of exact parity | Rounded values imply about -5.9%, not exactly 1.0. Missing repetitions/ranges/clock/load data prevent distinguishing noise from regression. Cache cannot explain it when off; do not attribute a cause without a controlled trace. |

## 8. Additional controls

| Check | Status | Evidence and limits |
|---|---|---|
| Flash Attention `-fa 1` with cache | PASS | Parser mapping and real tiny MoE / Qwen cache requests run with FA enabled. No claim that all model shapes/quantized KV combinations are supported. |
| Larger `-ub`: PP gain with <10% decode loss | N/A | Explicit `-ub` precedence is proved. Throughput requires paired long-prompt and decode measurements; no gain inferred from the auto-ubatch model. Larger buffers can consume expert-cache VRAM and hurt decode. |
| `--n-cpu-moe`, `-ot` precedence | PASS by code and CPU-MoE fixture | `common/common.cpp:1489` keeps automatic cache off when placement is explicit. Explicit cache N/-1 can opt back in; only CPU expert tensors are eligible. The fixture exercises `--cpu-moe` plus cache=4. |

## Controlled stock-placement results

Each entry has three independent process runs per build in A B B A A B order. Rates are median [minimum, maximum] tok/s; all accepted runs start at load1 < 2 and remain below one average foreign CPU core. The model, synthetic tokens, thread count, FA, placement and batch settings match. No semantic output equivalence is inferred from llama-bench.

| Workload | Upstream | Fork | Fork / upstream |
|---|---:|---:|---:|
| Qwen3.6 short-context decode | 53.776 [53.561, 54.023] | 55.359 [55.267, 55.394] | 1.0295x |
| Qwen3.6 12k-context decode | 52.496 [50.016, 53.007] | 53.381 [53.146, 53.555] | 1.0169x |
| Phi mini CPU decode | 29.074 [28.967, 29.391] | 29.102 [28.939, 29.318] | 1.0010x |
| Phi mini GPU decode | 163.057 [163.005, 163.095] | 162.968 [162.690, 163.061] | 0.9995x |
| Qwen3.6 12k prompt processing | 381.195 [380.917, 381.462] | 380.878 [380.676, 381.108] | 0.9992x |

The contaminated v1 attempt is excluded completely. V2 completed the six decode processes, then exhausted its bounded deadline during the separate prompt-processing case; only the complete decode case is used. V3 completes prompt-processing and both dense controls. Partial/interrupted rows remain in CSV with `ok=0` and blank rates. Source commits and exact binaries are recorded in `build-identities.json`; stale embedded fork version text is not used as source provenance.

## Fix commits and validation

| Commit | Minimal source change | Validation |
|---|---|---|
| `467286d48` | Verify HTTP byte-range responses before copying | Old code fails four boundary cases; all seven pass after |
| `a19f7db36` | Strip shard metadata from standalone MTP head; reject unsafe combined sharded trunk | Real GGUF fixture fails before, passes after |
| `c057441c7` | Keep automatic tuning/placement changes out of stock path | Real saved-state dense model 24/24 -> 2/2 before; 24/24 after; explicit 7/9 preserved |
| `468a22f81` | Explicit autotune parser values beat legacy environment | Precedence test fails before, passes after |
| `aec2a5362` | Tuning off does not consume/write learned state | Before/after real GPU fixture and file bytes/mtime |
| `fd96003e2` | Uppercase user memory margin is not overridden | Real initialization preserves MARGIN_MB=501 |
| `9c164555f` | Use existing portable BF16-pair cast in optional MMVF path | Original sm_75 build fails on unavailable intrinsic; full build succeeds; all 65,536 BF16 patterns match FP32 expansion (NaNs compared by class) |
| `839353168` | Exclude existing NextN count when locating GLM head | Before local fixture and live run fail at blk.46; after fixture passes and live blk.45 extraction succeeds |

The external-head fix is `08c52b62f` (`common/common.cpp:1716`). Qwen output and 45/53 acceptance remain identical; GLM external-head loading remains functional.

The BF16 checker also rejects deliberate pair swapping (32,642 mismatches) and a 1e-6 perturbation (16,897), then returns to zero failures.

The server ordering fix is `d19629860`: an actual `server_context::load_model` control preserves the stock draft limit at 3. Restoring the removed early auto-call as a mutation changes it to 5 and fails. The oversized Qwen auto run still correctly disables the external draft after placement.

The per-cache tuner candidate fix is `44f0ee12e`; both actual-function reload controls and the final real GPU fixture pass. `options-final.json` records the final parser/model-init regression checks.

`run-history.csv` includes 24 additional functional audit rows with full command/environment/hardware and deliberately empty throughput fields. They are not performance samples. It also includes the new stock-control runs and excluded attempts; only complete clean cases have throughput fields. `build-identities.json` records exact final binary/library SHA256 values and embedded versions for both comparator builds.

Build compatibility is separate from performance. The optional MMVF fast path was not enabled to infer a speedup. No gateway settings were changed or runtime deployed.

## Measurement integrity

The initial dense-state test driver used the server parser's unresolved `n_parallel=-1` without the normal server resolution step. This caused a large test allocation and OOM-killed the driver. It was a harness error, corrected to `-np 1`; it is not counted as a fork model failure. An in-progress vibe comparison potentially affected by its memory pressure was discarded entirely. The older, already completed native Strata reference was unaffected.

Functional fixture/MTP runs intentionally have `benchmark_valid=false`. Their raw timing fields are retained for traceability but must not be used as speedup evidence. Historical output files retain their original whitespace because hashes refer to the original bytes.
