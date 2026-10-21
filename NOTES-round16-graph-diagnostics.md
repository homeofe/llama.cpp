# CUDA graph execution diagnostics

2026-10-21. Both engines were captured sequentially with CUDA graph-node tracing. These are diagnostic captures, not benchmark/adoption measurements. The separate clean uninstrumented engine pilots remain in `NOTES-round16-draft-parity.md`.

## Recorded setup and limits

`task-logs/round16/profile_engine.py` uses the same aligned/native configuration and prompt as the engine pilot, warms the model and prompt, then captures an immediate repeated request inside NVTX `decode-audit`. Each process starts fresh. GPU ownership is acquired and released through the gateway. The GPU is empty after both runs. The capture includes reused-prefix processing and generation, not generation alone.

Aligned llama.cpp generated 217 tokens with 94 cached prompt tokens and four newly processed tokens; native generated 218 with 91 cached and seven newly processed. Outputs differ. Their hashes, draft counts, exact configs and request metadata are retained. The native warmup response also differs from its repeated response. Do not treat these captures as an identical-computation speed comparison.

Load1 at capture start was 1.918 aligned / 1.879 native. The residual CPU counter was 0.991 / 1.097 cores; it subtracts the engine and wrapper, but includes Nsight processes. Thus native fails the existing one-core gate. The contribution of profiler and unrelated processes cannot be separated retrospectively. Both runs already declare `benchmark_valid=false`; native timings support no clean cross-engine performance claim. Instrumentation can change execution timing and overlap.

## Structural observations

The old native capture omitted CUDA graph nodes. This one records 160,449 native kernels, all with nonzero graph-node IDs; aligned records 276,335 kernels, 183,708 belonging to graphs. This closes the earlier trace-coverage gap.

| Recorded calls in this request window | Aligned llama.cpp | Native |
|---|---:|---:|
| CUDA graph launches | 4,579 | 314 |
| Stream synchronizations | 33,047 | 153 |
| Event synchronizations | 9,936 | 19 |
| Graph instantiations | 385 | 0 |
| Graph exec updates | 1,166 | 0 |
| Recorded memcpy activities | 54,757 | 1,680 |
| Explicit GPU wait kernels | 0 | 9,792 |

These counts demonstrate different execution structures for these two near-equal-length responses. They do not by themselves assign a speedup to removing a call. In particular, native uses GPU wait-flag kernels while the host produces results; counting them as arithmetic would hide waiting rather than eliminate it.

The diagnostic interval unions below describe only the recorded, instrumented windows. Native's failed CPU gate and differing outputs preclude a reliable comparative time ratio:

| Recorded interval union, ms | Aligned | Native |
|---|---:|---:|
| Complete NVTX request window | 6430.389 | 3599.956 |
| Non-wait kernels, copies and memset | 2287.065 | 2135.039 |
| GPU wait activity with no overlapping non-wait work | 0 | 1081.545 |
| No recorded GPU activity | 4143.324 | 383.372 |

Each column partitions its window. Kernel and API duration sums are retained separately and are never added to idle time or to one another. Native's exposed wait plus idle is 1464.917 ms in this trace; calling its 383.372 ms idle the whole CPU wait would be wrong. The shape suggests that graph/scheduler boundaries deserve investigation; it does not establish that every idle interval is CPU expert computation or that a particular fraction can be removed.

## Consequences for optimization

- The clean pilots already show similar MTP acceptance with a large runtime gap. The graph-node traces now provide direct evidence of much finer execution boundaries and repeated graph construction in the aligned llama.cpp path. Inspect which shapes/partitions trigger the 385 instantiations and 1,166 updates before attempting a general graph-cache or scheduler change. This is new evidence for a targeted scheduling investigation; it does not reopen closed quantized-kernel tile sweeps.
- FLARE's cache-only MTP absorption remains a concrete source-level candidate, but these traces do not isolate its removable cost. Q2_0 is used by both draft and target tensors; kernel names alone cannot separate them. Do not claim the whole Q2_0 or vocabulary-head time as an available saving. The zero-output head is already empty during catch-up.
- All recorded CUDA matmul kernel names identifying Q2_0/type 42 sum to only 24.775 ms (0.385% of this instrumented aligned request window), including both target and draft. `profile-aligned-v1/q2-kernel-scope.json` preserves the five matching families. Eliminating just those recorded kernels cannot explain a multi-second gap. This does not bound shared-expert/attention work or host scheduling avoided by a cache-only graph, so it is not a rejection of that larger graph change.
- A follow-up timing comparison needs explicit accounting of the owned profiler processes and a clean rerun, plus equivalent phase/output accounting. The present captures are sufficient to reject the old claim that native graph execution was measured as mostly idle; they are not sufficient to establish a software ceiling or a deployment gain.
- REA can help inspect artifacts and package selected instrumentation, but has no built-in CUDA timeline or host-launch correlation. Its review is in `NOTES-round16-diagnostics-tools.md`; nothing was installed.

## Reproduction and artifacts

The capture command is archived per engine. Relevant options:

```sh
nsys profile --trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none \
  --cuda-graph-trace=node --capture-range=nvtx --nvtx-capture=decode-audit \
  --capture-range-end=stop --env-var=NSYS_NVTX_PROFILER_REGISTER_ONLY=0 \
  --kill=none --export=none ...
```

Nsight's automatic importer lookup still fails in the wrapped environment. The local importer and exporter work:

```sh
/usr/lib/nsight-systems/host-linux-x64/QdstrmImporter -i DIR/trace.qdstrm -o DIR/trace.nsys-rep
nsys export --quiet true --type sqlite --output DIR/trace.sqlite DIR/trace.nsys-rep
python3 task-logs/round16/summarize_profile.py DIR/trace.sqlite --out DIR/summary.json
```

`profile-aligned-v1/` and `profile-native-v1/` retain configs, logs, request results, summary JSON and exact commands. Large raw qdstrm/nsys-rep/SQLite files remain local and are git-ignored; their sizes/hashes are in `profile-artifacts.json`. The analyzer reads the complete NVTX interval, computes clipped interval unions, identifies the known wait-flag/doorbell kernels separately, and asserts that the three exclusive buckets sum to the window. Raw query results retain graph-node and copy-kind counts. No model or runtime setting was deployed.
