# Bounded CPU/GPU handoff diagnosis

Authorized 2026-10-21. Budget: 20-30 minutes, at most one new diagnostic capture
per engine, no optimization build or tuning sweep. Preflights may use the host.
Use the existing aligned baseline and deployed native engine, one warmed repeated
code-duration request each. Require load1 < 2 and foreign CPU < 1 core; otherwise
retain the capture as contaminated and make no comparative latency claim.

Observe CPU expert entry/return and correlate with input production and GPU
consumption. CPU ranges are elapsed time, including worker dispatch and waits.
They are not isolated arithmetic time. A gap after input production is only a
candidate for removal after its remaining dependencies are accounted for.
No summed or overlapping intervals may be presented as an additive speedup.

Use installed bpftrace uprobes for the native CPU dispatch and phase functions,
and an NVTX-only preload for llama CPU graph execution. Neither engine binary is
changed. Nsight captures CUDA graph nodes. Mark both clocks in the driver for
cross-tool alignment. All measured rates are diagnostic, never adoption evidence.

Record loaded llama libraries, binary hashes, output hashes, draft counts,
foreign load, hook coverage and clock alignment uncertainty. Native CPU dispatch
has arguments n_tok (arg3) and layer (seventh x86-64 SysV argument on the stack).
Native pool phases are identified by mode and task count. Per-row hooks are out
of scope. Uprobe/NVTX overhead is a limitation; no precision beyond that evidence.

Decision: identify a concrete avoidable transition with a credible >3% end-to-end
envelope, or stop this optimization direction. Generic OpenMP waiting and already
rejected cache/kernel/wait-policy axes do not qualify. REA is not installed: its
Frida adapter would need custom hooks for these same source-known boundaries.
