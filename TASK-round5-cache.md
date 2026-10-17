# Round 5: GPU-hit / CPU-miss expert cache for verify windows (llama.cpp fork, worktree `llama-l5`)

You work in `/opt/llm-gateway/data/build/llama-l5` (branch `l5-cache`). Read `AGENTS.md`, `CONTRIBUTING.md` and
`skills/code-review/SKILL.md` first. Wall-clock budget for this round: **3 hours**. Stop at the checkpoints below if the
numbers say it is not worth it, and write what you found.

## Why

Qwen3.8-Flash-Next (qwen4exp, 48 layers x 512 experts, 10 used, about 2.05 MB per expert) writes 40.7 tok/s (code) in
this fork with MTP 3 against **69.6 in Strata** on the same GGUF, same prompts, same draft acceptance (92 vs 90 %).
Measured today on this branch's base (`/opt/llm-gateway/data/moe-quant/qwen38/qbound.txt`, llama-bench `-ncmoe 48`):

| per window | 4 tokens (verify) | 1 token |
|---|---|---|
| window time (CUDA graphs on) | 76.9 ms | 36.8 ms |
| GPU kernels + copies | 28.1 ms | 19.9 ms |
| CPU experts and handoff | **48.7 ms (63 %)** | 16.9 ms |

Strata's log for the same model: a 1805-slot expert cache (3.47 GiB) on the GPU, **53 % of decode routings served by
the GPU from VRAM**, another 15 % read by the GPU over PCIe; the CPU computes only the rest. A simulation on our
routing trace (`/opt/llm-gateway/data/moe-quant/cache/sim_lfu.py`, 4 topics x 1024 tokens) gives LFU with decay 44 %
hits at 1805 slots with 5.3 uploads per token, 56 % at 3000; the best static set only 33 / 45 %. If the GPU computes
the hits while the CPU computes the misses, the CPU part of a window shrinks by about 40 %: **bound about +39 % per
verify window** on Qwen3.8. The same machinery applies to every MoE model whose experts live in RAM.

## What exists

- **This branch** (base `l3-mmq` plus our IQ3_S/IQ3_XXS AVX2 kernels): the Bolt stack. `GGML_SCHED_OVERLAP=1` cuts a
  GPU split that precedes a CPU split into part A (up to the producer of the CPU split's inputs), an event, and part B;
  the CPU copies wait only on the event, so part B runs on the GPU while the CPU computes. `LLAMA_MOE_SHEXP_EARLY=1`
  orders the shared expert after the routing and before the CPU experts so it lands in part B. Also: Turing MMVQ/MMQ
  thresholds, flat CPU MoE rows, and measurement switches (`LLAMA_MOE_SKIP*`, leave them alone).
- **Round 4** (Oct 11-12, `/opt/llm-gateway/data/build/llama-codex2`, uncommitted; READ ONLY, do not write there): a
  masked `GGML_OP_MUL_MAT_ID_MASKED` (CPU and CUDA MMVQ), a frozen resident set (`src/llama-moe-cache.*`,
  `GGML_MOE_HYBRID*`), complementary masks, joined in expert order. Its round-4 diff is the tree's current diff minus
  `task-logs/round4-hitmiss/source-before.patch` (that patch also holds older scheduler work you do NOT need). Results
  and limits: `/opt/llm-gateway/patches/llama.cpp/HANDOFF-codex-hitmiss-results.md`. Measured only +1.3 % because:
  (1) **single-token decode only: multi-token batches fell back to the ordinary path**, so MTP verify windows never
  used it; (2) serial, no overlap; (3) slot maps copied per layer and token; (4) a profile trained on 512 tokens per
  topic (31 % predicted hits at 2088 slots).

## This round (stages; measure after each)

1. **Port** the masked op and the frozen hybrid onto this branch (new code only, upstream style, env-guarded, off by
   default, ordinary path unchanged when off). Keep slot maps on the device for the life of the context (no per-token
   copy). Tests in `tests/test-backend-ops.cpp` for the masked op (zero / all / mixed hits, duplicate ids, the types
   below), CPU and CUDA.
2. **Verify windows:** the masked CUDA path must cover batches of 1 to 8 tokens (MMVQ handles up to 8 columns; mask
   per (token, slot)) for the expert types of our models: IQ3_S, IQ3_XXS, IQ2_S, IQ4_NL, IQ4_XS, Q2_0 (Qwen3.8),
   Q4_K, Q5_K, Q6_K (Qwen3.6 UD), IQ2_XXS (Bolt). A type without support falls back for that layer group only.
3. **Overlap by graph order:** build the GPU-hit FFN after the routing and before the CPU-miss split, the same way
   `LLAMA_MOE_SHEXP_EARLY` places the shared expert, so `GGML_SCHED_OVERLAP=1` runs the hits in part B while the CPU
   computes the misses. Prove it with the scheduler debug output (`GGML_SCHED_DEBUG=2`, needs a tool that does not
   silence logs, e.g. `llama-perplexity -v` or `llama-cli`) or `GGML_SCHED_TRACE`, not by assumption.
4. **Profile:** a frozen resident set from routing counts. Record a longer trace (several thousand tokens of the four
   topics in `/opt/llm-gateway/data/moe/texts/`; a trace hook behind an env var is fine) and size the set to the VRAM
   that is free with `-ncmoe 48` and the MTP head (about 3 GB; measure it).

**Checkpoint after stage 3** (before any adaptive work): Qwen3.8, `llama-bench -ngl 99 -ncmoe 48 -fa 1 -t 24 -n 0`,
`-p 4` and `-p 1`, `-r 30`, control (hybrid off) and hybrid, interleaved control/hybrid/hybrid/control, with the
environment `OMP_PLACES=cores OMP_PROC_BIND=close GGML_SCHED_OVERLAP=1 LLAMA_MOE_SHEXP_EARLY=1`. llama-bench feeds
random tokens, which route unlike text, so also report the hit rate on real text (perplexity or completion over the
held-out half of the trace texts). Then the same on Qwen3.6
(`/opt/llm-gateway/models/Qwen3.6-35B-A3B-MTP-GGUF/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf`, `-cmoe`, and `--n-cpu-moe 32` as
the gateway runs it). Adaptive LFU admission with uploads is the NEXT round; do not start it.

## Correctness

- Hybrid on with an empty (header-only) profile: logits bit-identical to hybrid off.
- Real resident set: GPU hits round differently from the CPU, so not bit-exact. Perplexity on
  `/opt/llm-gateway/data/build/texts/test-de.txt` (12 chunks, ctx 512) with `-ub 1` AND `-ub 4` (the window path) must
  stay within +0.5 % of hybrid off; report both.

## Machine rules

- RTX 2080 Ti 11 GB (sm_75, CUDA 12.0), PCIe 3.0, Threadripper 3960X (24 cores), 125 GiB RAM. Other agents use the
  machine: measure only when the 1-minute load is below 3 and no foreign process uses more than half a core
  (`/opt/llm-gateway/data/moe-quant/qwen38/qbound.sh` has a `clean`/`watch` pattern; copy it). Repeat a step that saw
  foreign load.
- GPU: free and hold it at the gateway before GPU work and release it after each step (`curl -X POST
  http://127.0.0.1:8790/api/v1/gpu/free`, `.../gpu/hold` with `{"reason":"codex: round 5 cache","ttlS":900}`,
  `DELETE .../gpu/hold`; header `Authorization: Bearer $(cat /opt/llm-gateway/data/api-key)`; never print the key).
  Never hold it while building or reading code.
- Builds: `nice -n 10 cmake --build build-cuda -j 16` (configure `build-cuda` like
  `/opt/llm-gateway/data/build/llama-l4/build-cuda`: `-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75
  -DCMAKE_CUDA_COMPILER=/usr/bin/nvcc -DCMAKE_BUILD_TYPE=Release`).
- Commit WIP to `l5-cache` in this worktree at each stage (no push, no PR, no `Co-Authored-By`). Do not touch other
  worktrees, `/opt/agent-tools`, Strata, or the gateway's configuration. Stop only your own processes.

## Deliverable

`NOTES-round5.md` in this directory: what was ported and changed, the overlap proof, hit rates (trace and real text),
the interleaved numbers (medians, spread), perplexity, what did not work, and what the adaptive round needs.
