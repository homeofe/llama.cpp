# Round 14: re-baseline the final stack, then attack the top remaining bucket

Machine: RTX 2080 Ti (sm_75, CUDA 12.0), Threadripper 3960X (shared, npm load comes and
goes), 11 GiB VRAM. Worktree `/opt/llm-gateway/data/build/llama-vibe`, branch `vibe-parity`.

## Why

Every time-table we have for "where does the round go" comes from the round-1 stage-1
trace, which was taken on the DEPLOYED BASELINE (q-r8-slots, pre-patch), before the final
stack landed: bf16-as-f16 (the hyper-connection matvecs went from 85 to 320 GB/s on the
affected shapes), MMVQ nwarps 2, int8 KV, 1779 slots, two elementwise fusions, async
uploads. That trace attributes ~28-30 ms/round to the dense small-batch GPU path,
~14 ms to an elementwise tail, 22.6 ms to the two GEMV families, 5-6 ms to the moe hit
kernels, ~35 ms to the CPU expert phase. All of those numbers moved with the stack, and
round 13 closed the int8-mma route for the quantized GEMV family (the feed is the wall,
NOTES-round13.md). Picking the next target on stale numbers is how effort gets wasted.

**Phase 0 of this round is a fresh nsys profile of `q-vibe-async` (the round-1 final
stack, 51.3/42.7 tok/s) on the same code prompt as round 1, with the same methodology
(graphs disabled, one steady request, quiet gate, GPU held).** Deliverable: the current
per-bucket table with the deltas vs the round-1 baseline spelled out, plus two cheap
facts collected while the server is loaded:

- free VRAM after load (nvidia-smi) - the hit rate is LINEAR in slots (sim: 1779 -> 48.6%,
  2048 -> 51.9%, 2400 -> 55.7%), so if >= ~600 MB are free, a slots bump is a candidate
  config change worth screening;
- the kernel-size distribution of the current round (is the sub-32-us tail still ~50%?).

## Then, in order of what the profile says (each is a candidate, stop after the first
that clears its gate; the task ends when the budget is spent)

1. **Elementwise tail fusion** (rms_norm / scale / bcast / unary / concat / get_rows /
   quantize chains, if they are still a big bucket): extend the ggml-cuda fusion
   infrastructure the same way round 1 did scale+silu and scale+sigmoid+scale
   (bit-exact, behind the same switch family). The most valuable single fusion is
   quantize_row_q8_1 into the preceding elementwise op if the profile shows the
   quantize kernels as a large sub-bucket - every MMVQ matvec needs the activations
   quantized first.
2. **Slots bump** (pure config, no code): q-vibe-async + GGML_MOE_HYBRID_MIB above 3640,
   sized from the measured free VRAM. Screen with the stage_b gate (+3% both categories
   to adopt). Watch for OOM during prefill and for upload-cost blowback at the higher
   admission volume - the admit-32 screen taught that more resident experts also mean
   more uploads per round.
3. **mul_mat_vec_f family** (the F16/BF16 matvecs on the hyper-connection and router
   tensors, if still >= 8 ms/round): the bf16-as-f16 conversion already lifted the
   bandwidth on the big shapes; check in the fresh trace whether the remaining time is
   fragmentation (many small tensors) or bandwidth, and only then decide.
4. **CPU core reservation** (deploy-level, the owner asked about it): pin the server's
   compute threads to a fixed core set (OMP_PLACES/PROC_BIND variants are already in
   the configs; a taskset/cpuset around llama-server is the stronger form). Measure
   stage_b under artificial CPU load with and without reservation. This does not raise
   the quiet-machine ceiling; it protects the 35 ms CPU expert phase (a 62-66 GB/s
   DDR4-bandwidth-bound phase) from npm contention. Only worth a round if the owner
   wants the under-load number improved specifically.

## Rules (unchanged)

- Write only in `/opt/llm-gateway/data/build/llama-vibe` and `/opt/llm-gateway/data/moe-quant/`.
- Hold the GPU at the gateway before every GPU step, release after; never print the API key.
- Quiet gate (1-minute load < 3, no foreign process above one core) for every timed GPU run.
- No sudo, no installs, no downloads. Pushes to homeofe/llama.cpp are authorized.
- NOTES in English, one-liners in `/opt/llm-gateway/data/moe-quant/NOTE.log` in German.

## Gates

- Phase 0 has no gate; it produces the decision table.
- Any adopted change: stage_b +3% on both code and english (3 interleaved rounds, n=12),
  perplexity within +0.5% of the control, outputs compared and differences explained
  (cache-state near-ties are the known noise source).

## Deliverables

`NOTES-round14.md` (English, every number with its source), the profile and its analysis
script under `task-logs/vibe-round14/`, any patch on vibe-parity, config changes proposed
but not applied to the live gateway without the owner's screen passing first.
