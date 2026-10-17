# Round 8: cache policy, PCIe bound and draft reservation

Base: `7a0b887761dfb9c29e9c3bcf14767853cba3ee55`, branch `l5-cache`, worktree `/opt/llm-gateway/data/build/llama-l5`. Work started 2026-10-17 17:28 UTC, with a three-hour limit.

## Result and execution boundary

The offline work is complete. Byte-value admission and soft per-layer budgets are opt-in, the production policy now supports multi-group replay with byte metrics, speculative verdicts are recorded, and the standalone mapped-host PCIe harness builds. The existing draft-specific ubatch switch supplies the requested reservation experiment. No new policy default is promoted: real traces from both models and the required GPU comparisons are still missing.

The gateway is reachable at 127.0.0.1:8790, but this sandbox has no usable NVIDIA device nodes (`/dev/nvidia0` and `/dev/nvidiactl` are absent). `nvidia-smi` exits 9 and cannot communicate with the driver. `task-logs/round8/preflight.json` and the final device check record that result; a `/dev/nvidia-caps` directory alone does not provide GPU access. The attempted trace recorder and GPU-step wrapper stopped at device preflight, before freeing or holding the GPU. There are no real route JSONL files in `/opt/llm-gateway/data/moe-quant/traces/`; its benchmark log records the failed attempt. No GPU reservation was taken by Codex, and no foreign process was stopped.

An external Claude attempt appeared in `task-logs/round8/pcie-run/claude-gpu.err`: it reached the RTX 2080 Ti, then failed on a JSON-to-pair conversion before timing. The ambiguous initializer was replaced with explicit integer extraction and `emplace`; the rebuilt harness passes its new host-only manifest check. That external attempt supplies no usable timing or correctness result. A fresh GPU run is required.

The first authorized WIP commit failed because Git could not create `/opt/llm-gateway/data/build/llama-upstream/.git/worktrees/llama-l5/index.lock`: read-only filesystem. No commit, push or PR is claimed. Stage patches replace the unavailable WIP commits. Existing untracked round-5/6/7 files were preserved. German milestones were appended to `moe-quant/NOTE.log`.

## Switches and unchanged defaults

| Setting | Default | Effect |
| --- | --- | --- |
| `GGML_MOE_HYBRID_BYTE_VALUE` | `0` | Rank admission proposals across layout groups by marginal saved bytes per uploaded byte. |
| `GGML_MOE_HYBRID_LAYER_SOFT_PCT` | `0` | Penalize counts in layers exceeding this percentage of their group's mean slot allocation; zero disables it. Valid range 0..1000. |
| `GGML_MOE_REPLAY_ACCEPTED_ONLY` | `0` | Replay only: defer speculative learning until a verdict and discard rejected routes. |
| `GGML_MOE_REPLAY_WARM` | `0` | Replay only: load the supplied profile in file order as the ordinary unpinned warm start. |
| `--spec-draft-n-ubatch 32` | Existing default `0`, inherit target | Existing switch used by the new measurement configs; only the draft ubatch changes. |

The existing runtime decay default remains 512. The measured round-7 benchmark controls explicitly use 128, admission 32 and prefill cap 1024. The new runtime flags default to zero; with them unset, the old admission path and graph construction are retained. The master cache switch still defaults off. Trace version 2 is emitted only when tracing is requested; replay accepts versions 1 and 2.

The LFU planner was separated into proposal and application phases. The old `plan()` calls both, preserving its ordering. Byte-value mode gathers proposals from all groups before applying the common upload cap. Its score is `(new_score - victim_score) * expert_bytes / (expert_bytes + map_write_bytes)`. The victim score is zero for an empty slot. Because each layout group stores equal-sized expert triplets, weight size mostly cancels in this ratio; the useful change is comparing marginal value across groups instead of letting the first group exhaust the allowance. Multiplying frequency by bytes alone would not implement the requested saved-bytes-per-uploaded-byte objective. This score remains a traffic proxy, not a measured CPU-time model.

The optional soft budget divides a layer's score by `max(1, resident_slots / soft_limit)`, with `soft_limit = group_slots / group_layer_count * percentage / 100`. It uses the residency snapshot before planning, has no hard quota, and never evicts pinned slots. The existing queued weight/map upload protocol is unchanged; the mixed-layout host test verifies it without adding synchronization.

## Trace and replay semantics

Version 2 adds actual slot counts per layer, the total group count including dropped groups, and `spec_begin` / `spec_accept` records with context, sequence, position, ubatch watermark and retained input-prefix length. The target span includes the anchor and draft inputs. Draft-MTP spans begin before draft evaluation. Verdicts are emitted before the checkpoint-restore early return, so rejected verification work can be identified even when accepted tokens are replayed. These hooks do nothing when tracing is closed.

`--replay-log TRACE 0 ADMIT PREFILL STATIC_PCT DECAY [PROFILE|-] [CONTEXT_ID]` uses recorded group slot counts. A positive slot argument retains the old single-group override. Groups without cache slots contribute misses and bytes to the denominator; their positions in the production round-robin order are preserved. A trace-only context with no recorded slots can be replayed as entirely uncached. Heterogeneous trace-only layouts need actual grouped cache traces to simulate a cache budget.

Accepted-only replay learns nothing from a marked speculative span until its verdict arrives. Retained votes are discounted by every decay boundary crossed while waiting; rejected votes are discarded. It does not use future verdicts during earlier admissions. Verdicts share the remaining upload allowance of the last ubatch, so several sequence verdicts do not create extra upload budgets. The production policy object implements learning and admission; verdict-dependent learning is an offline experiment, not a new runtime switch. Real GPU traces are needed before wiring that policy into inference.

Metrics use all evaluated tokens, including later-rejected speculative work. Decode byte hits are saved expert-triplet bytes divided by routed expert-triplet bytes during decode. First-32 byte hits cover the first 32 evaluated decode tokens after each recorded prefill. MiB/token includes prompt bursts and map writes, excludes initial profile uploads, and divides by all evaluated tokens. These are not accepted-output-token metrics and cannot predict tok/s on their own.

Hand-calculated fixtures cover unequal layout sizes, causal verdict handling, delayed votes crossing decay, multiple verdicts sharing a cap, dropped-group rotation, and a fully uncached context. The two-group fixture gives 83.3333% weighted decode hits without filtering and 75.0000% with filtering, with two rejected route votes. It deliberately shows that filtering can reduce hits; the replay is not an optimistic acceptance oracle.

## Real-trace results

| Required trace | Decode bytes hit | Upload MiB/token | First-32 bytes hit | Status |
| --- | --- | --- | --- | --- |
| Qwen3.8 eight-prompt server session | Unmeasured | Unmeasured | Unmeasured | Device preflight failed; no route file. |
| Ling eight-prompt server session | Unmeasured | Unmeasured | Unmeasured | Device preflight failed; no route file. |
| Long German real session | Unmeasured | Unmeasured | Unmeasured | Trace recorder stopped before this step. |

Consequently there is no evidence for a default that wins on both models' real routes. Keep the round-7 control settings. The first candidates to measure are byte-value alone and byte-value plus decay 64; soft budgets and smaller prefill caps are not favored by the available regression data.

## Old-trace regression and tuning

The old workloads are the round-7 synthetic prompt boundaries over the earlier routing trace: long has 4096 evaluated tokens in four topic segments; short has eight 40-token prompts plus 256 evaluated decode tokens each. They are not fresh server traces. All 24 original replay rows (two workloads, three decays, four warming/static combinations) match the saved round-7 binary's original metrics exactly.

Both tuning matrices start empty. They compare policy parameters, not `q-warm-draft`'s preloaded profile. The original one-group, 1501-slot matrix uses 2,050,000 bytes per expert. All 62 rows are in [replay-table.md](task-logs/round8/replay-table.md); raw metrics are in `replay-matrix.json`. Selected rows:

| Workload | Decay / admit / prefill | Decode bytes hit | First-32 bytes hit | MiB/token |
| --- | --- | ---: | ---: | ---: |
| Long, round 7 | 128 / 32 / 1024 | 48.6876% | 45.6396% | 16.0790 |
| Long | 64 / 32 / 1024 | 49.4121% | 46.0254% | 16.9682 |
| Long | 64 / 32 / 512 | 48.9049% | 39.7591% | 15.9997 |
| Long | 128 / 16 / 512 | 46.1314% | 37.8564% | 8.4903 |
| Short, round 7 | 128 / 32 / 1024 | 47.4323% | 44.4963% | 20.0887 |
| Short | 64 / 32 / 1024 | 49.0104% | 45.3426% | 20.2901 |
| Short | 64 / 32 / 512 | 46.9379% | 38.2544% | 16.9085 |
| Short | 128 / 16 / 512 | 41.2550% | 35.2376% | 10.1451 |

A second regression reuses those same routes with the actual Qwen GGUF tensor layouts and the production 2560 MiB slot calculation. It has 1248 slots across five live groups, with two dropped groups still represented in the round-robin schedule and byte denominators. This is a layout-corrected surrogate, not a real Qwen server trace. No Ling routing workload is fabricated.

| Old workload, actual Qwen layouts | Policy | Decode bytes hit | First-32 bytes hit | MiB/token |
| --- | --- | ---: | ---: | ---: |
| long | r7 | 43.9394% | 41.4018% | 14.9244 |
| long | decay64 | 44.2286% | 41.6418% | 16.7128 |
| long | prefill512 | 43.1251% | 30.8750% | 14.0282 |
| long | admit16 | 42.1593% | 40.2956% | 9.2792 |
| long | admit64 | 44.3368% | 42.8324% | 17.4276 |
| long | byte-value | 44.1655% | 43.1946% | 14.8698 |
| long | byte-decay64 | 44.9967% | 43.4715% | 16.6656 |
| long | byte-soft100 | 44.1193% | 43.2894% | 15.8917 |
| long | byte-soft150 | 44.1637% | 43.1870% | 14.8684 |
| long | byte-prefill512 | 43.6565% | 37.2293% | 13.9713 |
| short | r7 | 43.0482% | 40.3315% | 19.6193 |
| short | decay64 | 43.9755% | 40.7985% | 20.1034 |
| short | prefill512 | 39.5356% | 28.2917% | 16.2386 |
| short | admit16 | 40.8812% | 39.4699% | 13.3803 |
| short | admit64 | 43.9291% | 41.9151% | 25.0855 |
| short | byte-value | 43.5942% | 41.9733% | 19.5751 |
| short | byte-decay64 | 45.0775% | 42.5396% | 20.1366 |
| short | byte-soft100 | 43.5072% | 41.9362% | 19.9409 |
| short | byte-soft150 | 43.5955% | 41.9717% | 19.5761 |
| short | byte-prefill512 | 41.1958% | 34.6360% | 16.1322 |

Byte-value alone raises weighted decode hits from 43.9394% to 44.1655% on long and from 43.0482% to 43.5942% on short, while slightly reducing uploads in both. With decay 64 it reaches 44.9967% / 45.0775%, at 16.6656 / 20.1366 MiB/token. A 100% soft budget adds churn and does not improve total hits; 150% is nearly neutral. Smaller prefill caps reduce upload volume but lose substantial first-32 coverage. Admission 64 buys some extra hits at a large short-workload upload cost. These findings justify screening candidates, not changing defaults.

## PCIe harness

`pcie-misses.cpp`, `prepare_pcie.py` and `build-pcie.sh` are standalone artifacts under `task-logs/round8/`; no inference graph path was added. They reuse the existing CUDA masked MMVQ and CPU masked `MUL_MAT_ID` implementation. Expert slices come from the actual GGUF offsets, are held in `cudaHostAllocMapped` storage, and are accessed through `cudaHostGetDevicePointer`. Weight uploads to VRAM are bypassed. Input copies, the GPU kernel and result copies use the same backend stream. CPU and GPU masks are disjoint and their outputs are summed after joining.

The four-token window contains 48 layers with ten routed experts per token. Up/gate shapes are `[2560,640,512]`, down is `[640,2560,512]`; actual quant types include IQ4_NL, IQ2_S, IQ3_XXS, IQ3_S, IQ4_XS and Q2_0. The deterministic surrogate miss mask covers 2335.693 MiB of nominal routed weight bytes, or 1455.640 MiB of distinct expert weights. Split fractions count bytes per missed route, including repeat visits to an expert. They are 9.97073%, 19.95923% and 29.99582%, selecting 63, 134 and 206 distinct expert triplets respectively.

The harness performs 144 independent matmuls, with deterministic F32 activations. It measures CPU-only versus concurrent 10/20/30% splits in three interleaved rounds. Input/mask/ID copies, GPU synchronization, output copy and merge are timed; weight allocation and loading are outside the interval. It checks merged-output NMSE against CPU-only output, with a 5e-4 limit. Start and end load must both be below 3. This is a miss-compute prototype: it omits the full FFN dependency chain, attention, real GPU-hit work and runtime cache-upload contention, so a win would still require end-to-end validation.

| Split | CPU-only / split milliseconds per four-token window | Correctness on GPU | Status |
| --- | --- | --- | --- |
| 0% | Unmeasured | Unmeasured | No sandbox CUDA device. |
| 10% | Unmeasured | Unmeasured | No sandbox CUDA device. |
| 20% | Unmeasured | Unmeasured | No sandbox CUDA device. |
| 30% | Unmeasured | Unmeasured | No sandbox CUDA device. |

The >3 ms saving criterion has not been established. Implementing an inference graph path would violate the requested experiment gate, so none is included. The external failed attempt is retained separately; it is not a zero-copy performance result.

## Draft compute reservation and slots

The existing allocation log `moe-quant/qwen38/vram-breakdown.log` shows Qwen's draft CUDA compute buffer at 978.13 MiB with ubatch 512, one maximum output, and graph reservation `pp bs=512, tg bs=1`. Its draft model allocation is 741.18 MiB and KV allocation is separate. In `src/llama-context.cpp::sched_reserve`, prompt reservation uses `min(n_ctx,n_ubatch)`, followed by a generation reservation and another prompt reservation. It reserves for prompt evaluation, not just the three-token draft loop. This is consistent with the owner's observation that lowering ubatch reduces the buffer.

`draft-reserve.cpp` loads only metadata and builds MTP graphs on CPU without allocating model weights. At context 8192 and one output, it reports:

| Draft ubatch | Qwen CPU graph reservation MiB | Ling CPU graph reservation MiB |
| ---: | ---: | ---: |
| 512 | 137.99 | 102.01 |
| 256 | 69.07 | 51.00 |
| 64 | 17.37 | 12.75 |
| 32 | 8.89 | 6.38 |
| 8 | 6.33 | 1.59 |

Qwen's MoE down/weighted intermediates are `[2560,10,ubatch]` F32: 50 MiB each at 512 and 3.125 MiB at 32. Its hyper-connection intermediates also scale with ubatch; a 4 MiB indexer gather creates a floor at small batches. Ling's corresponding `[2560,8,ubatch]` tensors are 40 MiB and 2.5 MiB. The CPU graph does not reproduce the CUDA scheduler's buffer types, copies, fused graph choices or expanded allocations, so these values do not explain every byte of the historical 978.13 MiB or measure CUDA savings.

The safe experiment uses the already implemented `--spec-draft-n-ubatch 32`, leaving target ubatch 256 for Qwen and 512 for Ling. `common_base_params_to_speculative` changes only the draft setting; the ordinary decoder splits the draft prompt pass into bounded ubatches. No allocator is given a false size and no generation-only reservation is substituted for prompt storage. A test confirms explicit draft 32/target 256 and default inheritance. Smaller prompt chunks can change TTFT, kernel rounding and draft acceptance, which still need GPU testing.

`slot_budget.py` uses actual GGUF layouts, 128-byte alignment, map storage, the extra padding expert and the production minimum-slot rule. It reproduces the known control slot counts:

| Model / cache | Budget MiB | Slots by group | Total slots | Allocated MiB | Extra slots |
| --- | ---: | --- | ---: | ---: | ---: |
| Qwen target control | 2560 | 381,135,408,80,244,0,0 | 1248 | 2434.392 | 0 |
| Qwen target candidate | 2944 | 438,156,470,93,281,0,0 | 1438 | 2802.946 | 190 |
| Qwen target reference | 3072 | 458,162,490,97,294,0,0 | 1501 | 2926.115 | 253 |
| Qwen draft, unchanged | 512 | 101 | 101 | 508.010 | 0 |
| Ling target control | 5120 | 1668,0 | 1668 | 4987.520 | 0 |
| Ling target candidate | 5376 | 1751,0 | 1751 | 5235.547 | 83 |

The candidate budgets require approximately 368.554 MiB more actual Qwen cache storage and 248.027 MiB more Ling storage. Whether ubatch 32 releases that much usable VRAM is unmeasured. The slot counts are allocation calculations, not a claim that those configurations fit or run on the GPU.

## End-to-end measurement status

Existing configurations in `moe-quant/qwen38/stage_b.py` are preserved. Added candidates are `q-r8-byte`, `q-r8-policy` (byte plus decay 64), `q-r8-ub32`, `q-r8-slots`, `q-r8-combined`, and the corresponding `ling-r8-*` names. `slots` changes only draft ubatch and target budget; `combined` also enables byte-value and decay 64. None has been validated as the best configuration.

The new `--load-gate` mode checks load1 <3 at both request boundaries, guards against foreign GPU allocations above 300 MiB, clears inherited cache/draft knobs from control environments, and releases the lease on every exit. It checks GPU ownership during startup and periodically while streaming. A contaminated prompt pass is restarted from a fresh server; partial-pass successes do not enter the summary. The default foreign-process gate remains available for Claude's final comparison. The standalone `gpu_step.py` uses free/hold with reason `codex: round 8`, ttl 900, renewal and finally-release, and stops only its own process group.

Historical results below come from Claude's earlier `run-r7/runs.jsonl`, summarized by the existing `summary.py`; they are not new round-8 measurements. Each class has 12 accepted requests across three rounds.

| Configuration | Code tok/s | English tok/s | Code / English draft acceptance | Round-8 candidate comparison |
| --- | ---: | ---: | --- | --- |
| `q-warm-draft`, historical round 7 | 47.0 | 39.8 | 92% / 69% | Unmeasured |
| `ling-cache-mtp3`, historical round 7 | 50.8 | 47.5 | 68% / 53% | Unmeasured |

The requested best-candidate versus control comparison, with three interleaved rounds on both models, remains blocked by GPU access. No throughput gain is claimed for this round.

## Correctness and checks

| Check | Result / artifact |
| --- | --- |
| CPU cache-off logits against saved round-7 libraries | Byte-identical across baseline, new binary and new knobs set with master off. `off-parity.log`. |
| Trace-only versus cache-off CPU logits | Byte-identical. `contexts.log`. |
| Perplexity, `test-de.txt`, 12 chunks, ctx 512, ubatch 4 | GPU execution blocked; no new perplexity result for either model. |
| Production LFU, global byte value, soft budgets, pinning | Pass. `ctest.log`. |
| Queued host bank/map ordering, mixed layouts, no added sync | Pass. `ordering.log`. |
| Old replay metric parity | All 24 rows exact. `replay-checks.log`. |
| Causal replay, decay, verdict caps, dropped/uncached groups | Pass. `replay-checks.log`. |
| CPU server target-verdict integration | 44 closed spans, 126 rejected input positions, 504 rejected route votes, zero unresolved spans. `server-trace-check.log`. |
| Original and layout-corrected matrices | 62 + 20 rows, all completed. |
| CPU masked `MUL_MAT_ID` | 1314/1314 passed. `masked-cpu.out`. |
| CPU ordinary `MUL_MAT_ID` subset | 427/427 passed. `unmasked-cpu.out`. |
| CUDA Release server, perplexity and cache test build | Pass, compile only. `cuda-build.log`. |
| PCIe harness compile and JSON manifest check | Pass; GPU execution not validated. `pcie-build.log`, `pcie-manifest-check.log`. |
| Gateway/load/cleanup tests with mocks | Pass, including startup failure, busy end load, foreign VRAM and full-pass retry. `gate-checks.log`. |
| Python type checking | `ty` 0.0.88 passes on all ten round-8 Python scripts and the modified `stage_b.py`. `ty-check.log`. |
| Local style, script syntax and model naming | Pass. `style-checks.log`. |

The CPU cache-off comparison covers a small two-layer MoE fixture, a 20-token prompt and three four-token decode windows, 16,384 raw F32-logit bytes. SHA256: `c7b6f3986c8c60f65ca5f762760bce44a15615faebf06c9876745532ceb7f2cc`. This is a real bitwise regression check, but it does not substitute for GPU bitwise parity or full-model perplexity.

The CPU server integration uses two different small qwen3moe fixtures with `draft-simple` and two requests. It validates target begin/verdict ordering and replay of actual rejected rows. It does not exercise the MTP-specific draft hook or establish real-model quality.

The pinned CI version of `ty` was installed only under `task-logs/round8/check-tools`. The full repository dependency/type-check job was not run. `editorconfig-checker` is unavailable; local newline, whitespace, tab-indentation, ASCII and syntax checks plus the workflow's 157 model-name mappings passed.

## Reproduction and remaining work

Builds and reads must occur before acquiring a GPU lease. From this worktree:

```sh
cmake --build build-cpu --target test-moe-cache test-backend-ops -j 8
cmake --build build-cuda --target llama-server llama-perplexity test-moe-cache -j 8
bash task-logs/round8/build-pcie.sh
python3 task-logs/round8/model_inventory.py
python3 -S task-logs/round8/slot_budget.py
python3 -S task-logs/round8/prepare_pcie.py
task-logs/round8/pcie-misses task-logs/round8/pcie-manifest.json --check-manifest
python3 -S task-logs/round8/replay_checks.py
python3 -S task-logs/round8/replay_matrix.py
python3 -S task-logs/round8/legacy_layout_replay.py
```

The saved baseline libraries and old generated workloads are local artifacts, not embedded in the patch. `replay_checks.py` needs those baseline artifacts to compare against round 7.

After GPU access is available, record the real routes using the updated recorder, then replay each context with its recorded slots and its actual warm profile where applicable. Use `GGML_MOE_REPLAY_WARM=1` only for a profile that was actually loaded at trace start. Check for zero unresolved spans before interpreting accepted-only results.

```sh
bash /opt/llm-gateway/data/moe-quant/record-traces.sh
python3 -S task-logs/round8/gpu_step.py task-logs/round8/pcie-run-valid -- task-logs/round8/pcie-misses task-logs/round8/pcie-manifest.json
```

Screen byte-only, decay 64 and ubatch 32 separately before trying the larger budgets. Then choose the measured winner for each model and run `stage_b.py OUT DEADLINE_HHMM_UTC --load-gate --rounds 3 --only q-warm-draft QWINNER ling-cache-mtp3 LWINNER`; its outer loop interleaves and rotates configurations. Use `summary.py OUT/runs.jsonl` for medians. Claude repeats the selected pair with the full foreign-process gate by omitting `--load-gate` in a fresh output directory. Tracing must be off for timing.

For each model, run `llama-perplexity` through `gpu_step.py` with `-f /opt/llm-gateway/data/build/texts/test-de.txt --chunks 12 -c 512 -b 512 -ub 4 -lm none -fa on -ngl 99 -t 24`, the corresponding model and host-expert placement, and the control/candidate environments. Run cache-off with the new flags both zero and set while the master switch is zero; compare raw logits as well as perplexity. No new switch should become default until these checks and both-model real-trace comparisons pass.

Remaining risks: real GPU output and stream behavior are untested; the PCIe harness is a prototype using surrogate misses and independent matmuls; byte scores ignore quant-specific CPU cost and actual PCIe contention; accepted-only replay currently targets single-sequence route rows and non-shared-memory MTP, and unresolved/cancelled spans require investigation; prefill/request-boundary heuristics are inherited from round 7; smaller draft ubatches can affect TTFT and rounding; slot calculations do not establish VRAM fit. Initial profiles and dropped groups must be represented correctly when comparing real traces.

Final source and reproducibility changes are saved in `task-logs/round8/round8.patch`; stage patches are cumulative snapshots against the starting commit. The external harness edits are also captured in `task-logs/round8/measurement.patch` with paths relative to `/opt/llm-gateway/data`; that patch is embedded as an artifact in the worktree patch. Generated models, large traces, binaries, downloaded checking tools and raw logs remain local and are not embedded.
