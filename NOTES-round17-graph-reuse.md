# Round 17: bounded CUDA graph variants

Status: graph-variant candidate rejected at the pilot gate; no runtime adopted.
The live gateway is unchanged. The final CPU sampling profile is complete; the GPU hold was released at 14:15:11 UTC.

## Bound before building

The round-16 node-level trace has a 6430.389 ms aligned-request window. Interval
unions, clipped to this window, put graph lifecycle calls over idle GPU intervals
at 155.715 ms (2.422%), and individual kernel launches at 193.285 ms (3.006%).
Synchronization calls expose only 110.976 ms (1.726%). These are diagnostic upper
bounds, not independent or additive measured speedups. See
`task-logs/round17/api-idle-bound.json`. The existing failed batched-copy experiment
is not reopened.

The property probe records 12909 checks on 293 first-node keys. Of 12105 changed
states, 10266 have an exact previously observed property fingerprint. An LRU
simulation gives 4210 changed-state hits with two entries and 9280 with four.
This does not simulate graph capture, warmup, or timed eviction. The diagnostic
FNV fingerprint is never used as a correctness key in the implementation.
Source and results: `graph_probe.py`, `property-probe/`, `property-recurrence.json`
under `task-logs/round17/`.

## Implementation

`GGML_CUDA_GRAPH_VARIANTS=N` opts into 2–8 execution states per first-node key.
Unset, invalid, zero and one retain the original single-state path. The feature
requires NVIDIA VMM scratch allocation and is disabled with `GGML_CUDA_GRAPH_OPT=1`,
whose concurrent-region events can be recreated. It does not change tensor math,
expert placement, weights, the draft model, or speculation policy.

The active object keeps its address. On a complete property match, it swaps its
captured graph, executable, property snapshot and warmup state with a retained
variant. The comparison includes all original tensor bytes, source addresses,
source shapes and strides. Unknown states still execute directly before capture.
LRU eviction bounds alternatives; the existing ten-second inactive-key eviction
also destroys their alternatives.

Implementation checkpoint: `edbd6a4ab`. Only `libggml-cuda` is replaced in the
candidate runtime; `variant-binaries.json` identifies every copied file.

## Correctness and deliberate failures

`check_graph_variants.cpp` uses an exact F32 add-and-scale oracle and changes
shapes, source addresses, source strides and scale parameters, with more shapes
than fit in the cache. Each capacity 1/2/4/8 passes 264 checks. The old CUDA library
passes the same 264 checks. A restored capacity-four run also passes.

Two deliberate source cuts are archived separately:

| Cut | Exit | Failed cases |
|---|---:|---:|
| Ignore source data addresses | 1 | 24/264 |
| Ignore operation parameters | 1 | 48/264 |

The second includes the scale change from 1.0 to 1.000001. Capacity four records
106 warm executable reuses in the small oracle. The first version of the oracle
incorrectly expected negative zero from multiplication; CUDA's scale also adds
zero bias. Using the matching FMA reference fixes this oracle error; the initial
13 signed-zero failures remain archived in `check-valid/`.

With two entries, a real traced model run completes the warmup and all three
requests with exactly matching output hashes and draft counts. It records 4210
variant hits, including 2345 warm executable reuses. This is a functional trace,
not a speed benchmark (`cap2-probe/`, `cap2-reuse.json`).

## Rejected first pilot and measurement harness correction

`aligned-pilot/` is excluded in its entirety. The capacity-four candidate runs out
of device memory in `cudaGraphInstantiate`; it completes no measured request.
The expert-cache budget was not reduced to make this prototype fit.

The crash also exposed absolute build-directory ELF RUNPATH entries in the copied
executables. The old harness cleared `LD_LIBRARY_PATH`, so copying and hashing
libraries did not prove they were actually loaded. The first pilot's control
process also used the current build libraries. `INVALID.json` prevents its use in
the summary tool.

Commit `fd32ef149` fixes the shared round-16 launcher: it explicitly sets the
runtime library directory and checks every loaded ggml/llama/mtmd library through
`/proc/PID/maps` before accepting the process. A wrong directory is a hard error.
Round-17 requests also have an absolute wall-clock deadline. Native engine
launching remains separate. The corrected real capacity-two probe records all
eight expected libraries from its isolated runtime directory.

Historical round-16 copied-runtime comparisons do not establish library isolation
unless they provide equivalent loaded-library evidence. Configuration-only and
native-engine observations are not automatically disproved by this finding, but
old/new library attribution needs revalidation. No new adoption relies on it.

## Measurement protocol

The two-entry pilot uses the unchanged aligned model configuration: packed target
cache at 3600 MiB, the same 1815 frozen expert IDs, complete resident draft weights,
and the same speculation settings. Temperature zero and seed one; code-duration
and English, each novel and repeated. Each block gets a fresh server and caches.
The GPU is held through the gateway API. Load1 must be below 2 before each request;
foreign CPU must average below one core. A contaminated block is discarded whole.
No compilation or profiler runs during timing.

The source library hashes, complete configuration, loaded-library paths, full
outputs, timings, draft counts and memory snapshots are retained. Candidate
selection still requires a positive quiet pilot and three interleaved rounds,
with exact outputs and draft counts or additional quality evaluation.

## Result: two entries do not clear the 3% gate

The corrected pilot ran 13:49:10–14:01:06 UTC on 2026-10-21, including loading,
quiet waits and shutdown. Two fresh processes, four requests each; maximum foreign
CPU 0.420 cores, every request started below load1=2. Both processes recorded all
eight model libraries from their intended runtime directory. No block was discarded.

| Prompt / phase | Original tok/s | Two variants tok/s | Difference |
|---|---:|---:|---:|
| Code / novel | 40.904 | 40.387 | -1.26% |
| Code / repeat | 39.940 | 40.677 | +1.85% |
| English / novel | 37.243 | 37.496 | +0.68% |
| English / repeat | 36.365 | 36.539 | +0.48%, different draft trajectory |

All four paired output files are byte-identical. The first three pairs also have
identical offered/accepted draft counts. English repeat has 201/152 vs 203/151.
`common/speculative.cpp:1909` supplies actual round duration to the imported policy;
`common/strata-reference/draft_policy.cpp:39` uses measured costs to select windows.
Counter differences are consistent with that timing-dependent policy and are not
proof of a math error, but the fourth pair is excluded from an isolated patch-speed
claim. No category aggregate hides this difference. Source: `aligned-cap2/summary.json`.

GPU memory after requests: 10602 vs 10644 MiB (+42 MiB). The same expert IDs and
cache budget were retained. None of the comparable pairs clears the predeclared
3% gate, so there is no three-round confirmation, deployment, or quality-panel
campaign for this candidate. This is a rejected pilot, not a precise estimate of a
small speedup or regression. The two-entry conclusion does not prove that every
possible graph-reuse design is ineffective; four retained executables already hit
the memory limit in this configuration.

A cheap causal LFU simulation with two retained fingerprints gives only 3798
changed-state hits versus LRU's 4210 on the same trace. It does not justify another
cache-policy build. The next diagnostic samples actual CPU execution instead of
reopening the closed quantized-kernel or expert-cache-policy sweeps.


## CPU profile and next decision

A 25-second `perf record -F 199 --call-graph dwarf,4096` profile of the aligned
baseline completed two real requests. All eight runtime libraries were verified.
The raw capture is retained locally; its SHA256 and size are in
`task-logs/round17/cpu-aligned/profile-summary.json`. There are no lost samples.
The exported report and library disassembly are committed beside it.

53.72% of sampled CPU cycles land at libgomp offsets 0x256c0 and 0x258a0.
Disassembly confirms both are PAUSE polling loops followed by futex fallbacks.
39.27% land in the six listed quantized dot products. These are sampled CPU-cycle
shares across all threads, including idle time before and after the requests.
They are neither wall-time shares nor a removable-latency estimate. The profile
therefore does not establish a new speedup opportunity by itself.

Do not reopen the existing OpenMP wait-policy or native-pool sweeps: round 9
already found that shorter spin waiting reduced CPU cost but lost throughput;
round 10 repaired draft-pool integration but still failed the adoption gate.
The next useful architectural check is to attribute worker waits within actual
CPU-expert phases and compare their critical path with the deployed native
engine's persistent CPU service. Separate load imbalance, dependency waits and
host scheduling before proposing another implementation. Any further pilot must
use the corrected loaded-library verification. No new experiment is started in
this round, and no completion date for native-engine speed parity is established.

REA was reviewed at source level in round 16, not installed or executed. Its
static CUDA/GGUF inspection does not replace these CPU and GPU timelines. See
`NOTES-round16-diagnostics-tools.md` for the assessed revision and release limits.
The round-17 history exports 21 rows: six eligible request timings, two clean
requests excluded from isolated comparison because their draft trajectories
differ, and thirteen diagnostic, invalid-pilot or OOM records. No diagnostic rate
is presented as a performance gain.
