# Diagnostic tooling review: REA

Reviewed 2026-10-21, read-only. Source: [morluto/rea](https://github.com/morluto/rea), main commit `c538a843cb49f7156f86beb3b21501e03255af9a`. No installation, MCP registration, target attachment or benchmark instrumentation was performed. This is a source review, not a local acceptance test.

## Decision

Useful optional evidence adapter for static artifact inspection and targeted runtime hooks. It does not replace the CUDA graph-node timeline needed to explain the current decode gap. Both inference engines have available source, so decompilation is not the first step. Keep ordinary benchmark runs free of REA process sampling, Frida hooks, provider indexing and large artifact hashes.

| Task | Verified capability and limits | Relevance here |
| --- | --- | --- |
| Verify deployed CUDA target and kernel declarations | Static ELF/PE/fatbin/CUBIN/PTX inventory; selected register/shared-memory/parameter metadata. No GPU execution, SASS disassembly or host-launch correlation. CUDA target decoding is not restricted to newer SMs. | Can independently check that the actual library carries sm_75 and compare declared resource usage. This does not establish occupancy, runtime bandwidth or the critical path. |
| Check model and draft layouts | GGUF v2/v3 and safetensors metadata, tensor shapes/types/offsets and derived storage totals. Header-only hash by default; full-file hashing is separate. `ModelTensorTypes.ts` includes Q2_0 ID 42 (64 elements / 18 bytes), Q6_K and Q8_0. | Useful independent check of draft format and output-head dimensions. Header identity does not prove identical weight bytes. Existing converter/hash evidence remains necessary. |
| Investigate an assertion/crash | Recorded Linux ELF64 little-endian x86-64/AArch64 cores; registers, notes, optional mapping context. Requires caller-installed pwntools stack; optional GDB/pwndbg. Default core limit 32 MiB, configurable only up to 1 GiB, with fixed stage deadlines. | Limited fit for a large model process core; use our existing source-aware GDB backtrace first. REA's recorded-core operation does not itself provide a source-level stack investigation. |
| Inspect actual host behavior | Frida attach/spawn and caller-written instrumentation, retained script hash and bounded messages. Optional Frida dependency loaded on demand. | Could inspect a narrowly selected CUDA API call or host function when ordinary tracing leaves an unresolved question. Hooks execute in the target and can perturb scheduling; their timings are diagnostic, not benchmark rates. |
| Compare command behavior | PTY capture, interactions, exit/process observations and selected filesystem snapshots. Inherited environment is not recorded; snapshots are not syscall tracing. | Possible harness support, but our controlled environment, GPU hold, quiet-window checks and token/output comparisons must still be supplied. It adds no built-in MoE hit-rate or CUDA dependency analysis. |

## Availability matters

The reviewed main tree advertises package version 6.7.0, but the latest GitHub release is `rea-agents-6.7.0` from 2026-10-18, commit `27dc42d10249f0410904a75c0997842eb03b931c`. Its complete recursive tree has no `src/gpu/` or `src/model/`, and the two new guide paths return 404 at that tag. Therefore an installation of release 6.7.0 must not be represented as containing the reviewed GPU/GGUF features. They require a later release or an explicitly pinned source build. Recorded-crash support is present at that release tag. Local Node v24.21.0 satisfies main's declared Node requirement; other optional providers were not installed or validated.

## Next useful application

1. Complete the current uninstrumented microbatch series before any tooling work.
2. Capture matched warm requests with Nsight Systems `--cuda-graph-trace=node`; separate arithmetic/copies, wait-only kernels and device idle with interval unions. Similar MTP acceptance does not explain the remaining speed gap.
3. If a trace leaves a concrete host-call question, use a small source/NVTX probe or a narrowly scoped Frida hook in a separate diagnostic run. REA can package that evidence, but does not supply the hypothesis or remove instrumentation overhead.
4. Consider the static REA readers as an independent cross-check after the required features are released. There is no evidence from this review that installing REA alone increases tokens/s.

## Source references

- [GPU guide](https://github.com/morluto/rea/blob/c538a843cb49f7156f86beb3b21501e03255af9a/docs/gpu-code-objects.md), [CUDA target decoder](https://github.com/morluto/rea/blob/c538a843cb49f7156f86beb3b21501e03255af9a/src/gpu/CudaTarget.ts), [GPU provider](https://github.com/morluto/rea/blob/c538a843cb49f7156f86beb3b21501e03255af9a/src/gpu/GpuCodeObjectsProvider.ts). GPU inspection always hashes the full file before selected payload work; avoid it during timings.
- [Model guide](https://github.com/morluto/rea/blob/c538a843cb49f7156f86beb3b21501e03255af9a/docs/model-artifacts.md), [quantization layouts](https://github.com/morluto/rea/blob/c538a843cb49f7156f86beb3b21501e03255af9a/src/model/ModelTensorTypes.ts).
- [Recorded crashes](https://github.com/morluto/rea/blob/c538a843cb49f7156f86beb3b21501e03255af9a/docs/recorded-crashes.md), [Frida](https://github.com/morluto/rea/blob/c538a843cb49f7156f86beb3b21501e03255af9a/docs/frida-instrumentation.md), [process capture](https://github.com/morluto/rea/blob/c538a843cb49f7156f86beb3b21501e03255af9a/docs/process-capture.md).
- [Release 6.7.0](https://github.com/morluto/rea/releases/tag/rea-agents-6.7.0), [release source tree](https://github.com/morluto/rea/tree/27dc42d10249f0410904a75c0997842eb03b931c/src).
