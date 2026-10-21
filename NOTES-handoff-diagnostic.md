# Bounded CPU/GPU handoff investigation, 2026-10-21

Status: two captures completed, no optimization built or adopted. One aligned
capture passed the foreign-load gate; the native capture failed it. No third
capture was started. Both engines stopped and the GPU was released at 14:43:13 UTC.

## Decision

The aligned trace localizes a possible scheduling cost on the return from CPU
experts to GPU continuation. It does not establish an attainable speedup or a
valid cross-engine latency difference. No implementation is justified by this
investigation alone. Input handoff and generic OpenMP waiting are not new reasons
to reopen earlier tuning sweeps.

The immediate post-copy/pre-CPU gap is small, but it is not the complete handoff.
The analysis therefore follows the actual CUDA input event back to the last work
submitted before its record, then follows CPU completion forward to the next GPU
MoE weighted reduction. This covers all 3312 observed target CPU expert ranges.

## Scope and method

The declared budget was 20-30 minutes, at most one capture per engine, using one
warmed repeated code-duration request. Existing model configurations and frozen
expert placement were retained. Load1 < 2 and foreign CPU < 1 core were required.
The GPU was held through the gateway API; owned processes and the lease were
released on exit. No gateway settings, runtime installation or engine source were
changed. Analysis after the captures ran while preflights used the host; those
analysis timings are not measurements.

Installed bpftrace uprobes observe the native `drive_pool_multi` entry/return and
`ExpertPool::run_phase` entry/return. The seventh SysV argument is read at SP+8,
checked against the binary prologue and validated by token/layer bounds. Four
NVTX clock brackets, including request completion, give a common timestamp
offset with a 1.275 us half-width. This assumes a constant offset and does not
bound uprobe overhead or unobserved clock drift.

A diagnostic preload surrounds the original llama CPU graph function with NVTX
ranges and retains token count, node and weight names. It calls the original
function once and leaves other backends alone. Nsight records CUDA graph nodes.
The corrected round-17 launcher verifies all eight loaded llama libraries.
The aligned answer hash and offered/accepted draft counts match the previous
round-17 CPU-profile control: 217 output tokens, 175 offered, 148 accepted.
The complete output and instrumented configuration are archived.

REA was not installed or executed. Its Frida adapter would require custom hooks
for these same source-known boundaries. Existing tools provided the needed
observations without downloads or engine rebuilds.

## Valid aligned capture

Load1 at request start: 1.930; foreign CPU: 0.321 cores. The 6404.268 ms NVTX
window includes reused-prefix prompt processing and decode, not only token
generation. All durations below are instrumented observations.

| Observed section | Total | Interpretation |
|---|---:|---|
| Target CPU expert ranges | 2810.311 ms | Includes dispatch, arithmetic and worker waiting |
| Same CPU ranges without simultaneous GPU work | 2736.553 ms | GPU has no recorded kernel/copy/memset in these subintervals |
| Input readiness boundary to CPU entry | 146.236 ms | Configured part-A event completion, copies and host transition |
| Device idle inside that input interval | 23.721 ms | 0.37% of the complete capture |
| CPU return to next GPU MoE weighted reduction | 672.220 ms | Copies, host graph work, remaining GPU work and continuation |
| Device idle inside that return interval | 562.486 ms | 8.78% of the complete capture; zero-cost envelope only |

The sections overlap with GPU activity as indicated and must not be added as
independent speedups. CPU entry itself follows the last DtoH completion by a
median 4.43 us in four-row windows. Counting only that small gap would miss
most of the input and return transition.

For four-row CPU windows, median input-to-CPU time is 38.53 us; median return-to-
reduction time is 99.53 us, with 63.63 us device idle inside that return interval.
Across all token shapes, the return interval has a long tail: p95 747.61 us.
An inspected example performs output/metadata copies and graph capture before
the GPU add and weighted reduction resume. Both event readiness and reduction
pairing cover 3312/3312 ranges; no nearest-copy heuristic is used for these totals.

Within the return intervals, CUDA runtime graph lifecycle calls occupy
142.638 ms, graph launches 49.825 ms, captured kernel launch calls 89.834 ms,
copy calls 36.735 ms and synchronization calls 137.693 ms. Another 215.494 ms
lies outside recorded CUDA runtime calls. These are CPU API intervals, overlap
GPU work, and are not additive with the idle total. Time outside CUDA calls is
not automatically attributable to one framework function or removable work.

**Limit:** Nsight instruments graph capture/launch and can inflate exactly this
path. The 8.78% interval share is not a forecast of an unprofiled improvement.
The rejected round-17 graph-variant pilot remains rejected; these observations
do not overturn its end-to-end result. The input boundary is the scheduler's
existing completion event, not proof of earliest mathematically possible input
production. Moving that boundary earlier would require a separate dependency
proof and is not proposed here.

## Native capture excluded from performance comparison

The native request started at load1 1.874, then foreign CPU reached 1.278 cores.
`quiet=false` is retained in its event record. The result has 218 output tokens
and 169/151 offered/accepted drafts; the text and trajectory differ from aligned.
This is neither a quiet comparison nor identical work, and no latency ratio or
throughput gain is claimed.

Instrumentation coverage is useful independently of its speed: 3360 dispatch
ranges pair one-to-one in order with 3360 GPU doorbell-publish kernels and 3360
mapped-result-copy kernels; 6650 pool-phase ranges were retained. Every paired
dispatch has valid layer/token arguments and verified event ordering. No lost
event message was reported. Source confirms that the native pool follows the
doorbell and publishes the completion flag before the mapped-result consumer.
This proves the hooks work; it does not quantify the native scheduling advantage.

## What would be needed to proceed

Keep the return-to-GPU path as an unproven architectural hypothesis. Before any
implementation, obtain a quiet native comparison and check the aligned return
delay with lower-overhead timestamps. A credible >3% end-to-end envelope after
accounting for observation overhead is the gate, followed by the existing exact
output/quality and interleaved timing requirements. No further run is scheduled
by this report. No CPU-kernel, cache-policy or OpenMP setting sweep is reopened.

## Reproducibility

All small artifacts are in `task-logs/handoff-diagnostic/`: plan, source, capture
commands/configurations, raw host events, exact outputs, clock brackets, library
paths, interval details and summaries. `artifact-manifest.json` identifies local
raw captures and binaries. Raw profiler files and the preload binary are ignored
by Git; their hashes are retained. Four request rows are exported to
`tools/bench/run-history.csv`, all explicitly diagnostic with `ok=0`.

Validation: six interval-union/clipping checks, Python compilation, actual hook
coverage, event-order/argument checks, loaded-library verification, matching
aligned output/draft counts and final empty GPU process list. No speed claim
depends on the contaminated native capture.
