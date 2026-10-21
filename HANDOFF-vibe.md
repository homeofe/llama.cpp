# HANDOFF - the Qwen3.8-Flash-Next decode-speed effort, state after round 14

**Round 16 in progress (2026-10-21): `NOTES-round16-draft-parity.md`.**
Q2 experts and a reduced Q6_K draft head run in llama.cpp, with the draft fully on
GPU. The initial two-prompt gain DID NOT generalize: clean A B B A, n=8/category,
shows only +0.30% code / +1.12% English by server time (`confirm-clean/`). Stopped
early below the +3% gate; no adoption. The head mapping correctness controls pass.
Current GPU work: `task-logs/round16/global-screen/`, an eight-prompt check of Strata's
whole-hidden MTP norm, behind `LLAMA_MTP_HNORM_GLOBAL=1` (default off, draft only).
CPU/CUDA reference checks pass. Do not launch another GPU job while its runner holds
the GPU. Two head-only fallback model files with original Q8 experts are also ready
(`q8exp-q8subset` / `q8exp-q6subset`); not yet measured.
Live settings are unchanged. The old runtime is `task-logs/round16/e2e/original/`;
`e2e/isolated/` differs only in libllama (reduced head), `e2e/global/` additionally
has the experimental norm switch. Source checkpoint: `2bf16dd2b`, local, not yet
pushed. The abandoned `confirm/` run encountered foreign load and is excluded.
The updated runner discards/reloads a whole block after contamination to prevent
uniquely warmed retry caches. Final selection still needs 3 rounds and quality.

**Round-15 correction (2026-10-21): read `NOTES-round15-strata-comparison.md` first.**
The old Strata trace omitted CUDA graph nodes, the archived n5 crash was graph-allocation
OOM, and the software ceiling was overstated. Strata's reduced draft vocabulary and
compressed resident draft experts are concrete unisolated differences. The measured
51.3/42.7 stack remains valid; no new speedup is claimed by the audit.

For the next agent. Historical round-14 state follows, with corrections below.
Read order after the round-15 audit: this file, then `NOTES-vibe-round1.md`, `NOTES-round13.md`,
`NOTES-round14.md`, `TASK-round12-admission-tier.md`.

## The goal and where it ended up

Original goal: reach Strata's decode speed for Qwen3.8-Flash-Next-IQ3_S in llama.cpp
(Strata measured ~70/61 tok/s code/english; our deployed config was 48.4/40.1). The
owner then directed: extract ALL available performance, keep going, and push everything.

Result: **51.3/42.7 tok/s (+6.0/+6.5% over deployed), all on branch `vibe-parity`,
pushed to `homeofe/llama.cpp`**. The remaining gap includes architectural differences;
the measurements below do not establish that all software options are exhausted.

## Machine, rules, locations

- RTX 2080 Ti (sm_75, 11 GiB, CUDA 12.0, SM clock pins at 1350 MHz under load), Thread-
  ripper 3960X (24 cores, shared with npm/node jobs that come and go).
- Worktree: `/opt/llm-gateway/data/build/llama-vibe`, branch `vibe-parity`, remote
  `homeofe` = `github.com/homeofe/llama.cpp` (pushes authorized; `origin` is ggml-org).
- `/opt/llm-gateway` itself is deploy-only: change NOTHING there (an untracked file
  blocks the gateway's deploy). Write only in the worktree and
  `/opt/llm-gateway/data/moe-quant/`.
- GPU discipline: hold the GPU via `POST 127.0.0.1:8790/api/v1/gpu/hold` with
  `{"reason":"...","ttlS":900}` before every GPU step, DELETE after; the key is in
  `/opt/llm-gateway/data/api-key` - read it, never print it.
- Quiet gate for every timed GPU run: 1-minute load < 3, no foreign process above one
  core. No sudo, no installs, no downloads.
- NOTES in English, one-line German milestones appended to
  `/opt/llm-gateway/data/moe-quant/NOTE.log`.

## The current stack (all behind env switches, default = deployed behavior)

Config name in the harness: `q-vibe-async`. Components, in landing order:

1. `GGML_CUDA_BF16_AS_F16=1` - bf16_f16.cu: in-place bf16->f16 conversion so the
   hyper-connection matrices run on the f16 tensor-core path (round 14 trace:
   mul_mat_vec_f went 10.2 -> 1.5 ms/round; the work now shows up as mul_mat_f
   5.4 ms/round).
2. MMVQ Turing launch table nwarps 4 -> 2 for batch 2-8 (`ggml/src/ggml-cuda/mmvq.cu`).
3. int8 KV (`-ctk/-ctv q8_0`) + `GGML_MOE_HYBRID_MIB=3640` (1418 -> 1779 expert slots;
   perplexity IMPROVED -0.51% at ub4).
4. `scale+silu` fusion and 5. `scale+sigmoid+scale` fusion (`scale-unary-fused.cu`),
   bit-exact.
6. `GGML_MOE_HYBRID_ASYNC_UPLOAD=1` - expert uploads on a separate stream, event-gated
   publication (the round-11 patch); +1.8/+1.7%.

Benchmark protocol: stage_b, n=12 per config and category, 3 interleaved rounds, quiet
gate, hashes compared - 24/24 bit-identical vs the control.

## The round-14 re-baseline (the current truth about the decode round)

Source: `task-logs/vibe-round14/trace.sqlite` (nsys, capture-range recipe, graphs off,
`code-duration` prompt, output hash bit-identical to round 1). Decode round, 60 rounds:

| bucket | ms/round | verdict |
|---|---:|---|
| GPU kernels total | 34.9 | at per-family measured limits |
| - mul_mat_vec_q | 11.5 | round 13 closed further gains (mma feed is the wall) |
| - mul_mat_vec_q_moe (cache hits) | 5.6 | already one launch per layer-ubatch |
| - mul_mat_f (f16 tensor core, hc) | 5.4 | the optimized path from patch 1 |
| - mul_mat_vec_f | 1.5 | was 10.2 before the bf16 patch |
| - elementwise tail | ~9.5 | launch-bound (2.5-3 us/kernel); fusing saves only ~1-1.5 ms/round - below the +3% gate |
| GPU copies HtoD | 9.3 | 103 MB/round at 12.4 GB/s, async stream overlapped |
| GPU idle | ~49 | **52% of the round** |
| - CPU expert wait (>0.5 ms gaps) | 31.9 | 36 gaps/round, host threads in compute, NOT CUDA calls: a real per-layer data dependency (routing for layer L is only known when L runs) |
| host cudaStreamSynchronize | 13.5 | 556 syncs/round, mostly overlapped with GPU work |

## Closed axes - do not reopen without new evidence

- int8-MMA for the batch-4-5 quantized matvec (round 13, `NOTES-round13.md`): the mma
  phase alone runs at 690-750 GB/s equivalent, but every way of feeding the m16n8k16 tile
  costs more than the whole dp4a kernel (shared staging +38 us, scattered self-loads
  +50 us, shfl counted out). The 300 GB/s dp4a path is the ceiling for this tile shape.
- Cache policy (round 12): admit 32 measured optimal (128/256 lose 3-6% to upload
  costs); pin hybrid monotonic negative; Strata's policy is first-touch-no-eviction =
  repeat memory (10.3% on novel requests, 100% on repeats). LFU knob space flat.
- More slots (round 14): **252 MiB VRAM free** at the current config - dead. The hit
  rate is LINEAR in slots (sim: 1418->43.7%, 1779->48.6%, 2048->51.9%, 2400->55.7%),
  so this axis is pure VRAM.
- q4 KV failed MTP draft init; draft-cache removal -2/-3.7%; the archived n5 failure
  was CUDA graph-allocation OOM (round 15 runs with 256 MiB extra headroom, but loses
  English speed in a short screen); n4 trades english for code; CUDA graphs are worth
  1.7 ms/round (already on in production); prefill 1024 optimal.
- The CPU expert phase is DDR4-bandwidth-bound (2.19 GB at 55-66 GB/s) - no CPU-side
  kernel work can fix it; only fewer bytes (more cache hits = more VRAM) or overlap
  (the dependency is real) help.

## What a next agent can actually do (ranked)

**The round-15 source comparison supersedes this ordering:** first bound the reduced
MTP vocabulary/head (368.4 MiB Q8_0 head savings; about 2.0 ms/round at estimated linear
scaling), then isolate resident Q2_0 draft experts. Re-capture Strata with graph nodes
before sizing a whole-verifier synchronization change. See the new audit for checks,
cost bounds and limits. The earlier hardware/routing options remain below for context.

1. **Hardware**: a GPU with more VRAM. The expert cache is the whole game: at 3640 MiB
   the hit rate is ~41-49%; every +600 MiB is ~+3 hit points and the CPU phase shrinks
   proportionally. On a 24 GB card this model's decode likely lands near Strata.
2. **Speculative expert routing** (research): hide the 32 ms/round CPU-expert exposure
   by predicting the next layer's topk before it runs and prefetching/computing CPU
   experts ahead. Nothing in rounds 5-14 tried this. Round 15's previous-window-union
   predictor reaches only 41.5% precision; better predictors remain open.
3. **n5 memory headroom** (`--spec-draft-n-max 5`): the archived crash is OOM during
   cudaGraphInstantiate. It starts with a smaller target cache, but the short round-15
   screen loses English throughput; no adoption. Gate on BOTH categories.
4. **Elementwise fusions** only if they come free with other work (~1-1.5 ms/round,
   launch-bound tail; the round-1 fusion infra in ggml-cuda is the template).
5. **Upstreaming the stack** to ggml-org/llama.cpp (bf16-as-f16, MMVQ nwarps, fusions,
   async uploads, the moe-cache family): all self-contained; AGENTS.md says the OWNER
   writes the PR text - never generate it.
6. **Under-load robustness** (owner mentioned it): core reservation (taskset/cpuset)
   around llama-server protects the 32 ms CPU phase from npm contention; it does not
   raise the quiet ceiling. Deployment-level, measure with stage_b under load.

## Tooling (all still working)

- `stage_b.py` (`/opt/llm-gateway/data/moe-quant/qwen38/`): the bench harness. Controls:
  `q-vibe-control` (baseline), `q-vibe-async` (the final stack). Deadline arg is UTC
  HH:MM; 3 s polls; quiet gate built in. Config definitions show every env switch.
- `bound/`: `mmv_bench.c` (ggml MUL_MAT on the dense shapes), `mma8_bench.cu` (round-13
  prototype: check/bench/dump modes, 4 mutation cuts, phase probes),
  `mma8_probe.cu` (m8n8k16 fragment layouts, LAYOUT 1 = truth, 128/128),
  `sim_policy.py`, `sim_hybrid.py` (validated cache simulators).
- `task-logs/vibe-round1/stage1.py`: bench + nsys machinery (Hold, wait_quiet,
  gate_request, request); imports stage_b.
- `task-logs/vibe-round14/driver.py` + `profile2.py`: **the working nsys recipe**.
  Plain `nsys profile` around llama-server loses the report (the 83 GB shutdown gets
  SIGKILLed mid-finalization). Use: `--capture-range=nvtx --nvtx-capture=round14
  --capture-range-end=stop --env-var=NSYS_NVTX_PROFILER_REGISTER_ONLY=0`, NVTX range
  pushed from the driver around the request, then run
  `/usr/lib/nsight-systems/host-linux-x64/QdstrmImporter` and `nsys export --type
  sqlite` manually (nsys's own importer lookup fails from the wrapped env).
- `analyze_trace.py` (`task-logs/vibe-round1/`): sqlite -> per-kernel totals, burst
  detection. The round-14 gap/sync analysis snippets are in NOTES-round14.md.

## Gotchas (each cost real debugging time)

- `build-cuda/bin` IS the final stack (all env strings and symbols verified; the
  compaction note that said "baseline restored" is wrong). `bound/libs-tuned` is GONE;
  stage_b's `LD_LIBRARY_PATH` pointing there is harmless (loader falls back to rpath).
- ncu is unavailable: ERR_NVGPUCTRPERM, no sudo. Use phase probes in the harness
  (mma8_bench bmode pattern) or the nsys gap/sync attribution instead.
- Measure A/B interleaved in the same minutes: the SM clock pins at 1350 MHz under
  sustained load; non-interleaved comparisons drift 20%+.
- `pkill -f` with a self-matching pattern kills your own shell (hit twice). Match on
  `profile\.py` style patterns and check `pgrep` output before killing.
- Background shells: `nohup cmd &` in the bash tool blocks until timeout - use
  `tools.process.start`.
- `block_q6_K` field order is ql, qh, scales, d (210 bytes) - hardcoded offsets in
  offline replications get it wrong; the kernel uses struct fields.
- nsys wrapping a server: `--duration` counts from launch (includes model load);
  the capture-range recipe bounds the trace to the request instead.
- The trace field `cache` is `use_cache`, NOT a hit flag.
- `GGML_CUDA_DISABLE_FUSION=1` disables ALL fusions including pre-existing ones - it
  is not an isolation switch.
- Debug-dump kernels: launch a SINGLE block and use per-lane dump slots; multi-block
  dumps collide and lie.

## Current branch state

`vibe-parity`, pushed through `7954e8c53` (round 14). History: round-1 final stack
commits, round-12 task files and analysis, round-13 int8-MMA negative result, round-14
re-baseline. German one-liners for every phase are in `/opt/llm-gateway/data/moe-quant/
NOTE.log`. The live gateway settings were never touched; adopting `q-vibe-async` as a
deployed config is the owner's call (it passed the gate in round 1).
