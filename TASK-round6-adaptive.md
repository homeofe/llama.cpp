# Round 6: adaptive admission for the GPU-hit / CPU-miss expert cache (worktree `llama-l5`)

You continue your own round 5 in `/opt/llm-gateway/data/build/llama-l5` (branch `l5-cache`). Read `NOTES-round5.md`,
`AGENTS.md`, `CONTRIBUTING.md`, `skills/code-review/SKILL.md`. Wall-clock budget: **3 hours**.

## Environment (read first, it saves you an hour)

Your sandbox has no network (the gateway's GPU hold is unreachable), no view of other processes, and cannot write the
worktree's git metadata. That is intended: **do not ask for access and do not attempt GPU runs.** Build the CUDA
tree, run CPU tests and host-side checks; Claude runs every GPU test and measurement afterwards and commits. Save your
diff as `task-logs/round6/round6.patch` when you finish.

## Round 5 results (measured by Claude on the GPU, 2026-10-17)

- CUDA masked tests 716 / 716, ordinary `MUL_MAT_ID` regression clean.
- Empty profile: perplexity identical to hybrid off (3.8093). Round-4 4096 MiB profile: 3.8081 (`-ub 4`) and 3.8175
  (`-ub 1`) against 3.8093 / 3.8257: quality unchanged.
- Overlap proven: `GGML_SCHED_DEBUG=2` shows part A of every MoE split ending at `ffn_moe_miss_mask-N`; the GPU hits
  run in part B beside the CPU misses.
- Speed, Qwen3.8 `-ncmoe 48`: llama-bench pp4 50.2 / 52.9 -> 58.1 / 57.9 tok/s (+12.7 %), pp1 +4.5 %; perplexity
  with real text and 4-token batches 123.7 -> 103.7 s (+19 %), 1-token batches +10 %.
- Fixed by Claude: the MTP draft model reads the same environment and its context threw "profile references an
  uncached layer"; a profile that names layers without routed experts in a model now leaves the cache off for that
  context with a warning (`f8a3142a7`).
- The server with MTP has room for only about 2 GB of cache next to the draft head, and the frozen round-4 profile is
  weak (trained on 512 tokens per topic): a static set reaches at best 33 % hits at 1805 slots on our trace.

## Goal of this round

A simulation on the round-4 routing trace (`/opt/llm-gateway/data/moe-quant/cache/sim_lfu.py`, Qwen3.8, 4 topics x
1024 tokens) gives **LFU with decay 44 % hits at 1805 slots with 5.3 uploads per token**, 56 % at 3000; the best
static set only 33 / 45 %. Strata's adaptive tier reaches 53 % in the gateway. Build that policy:

1. **Count routing** per (layer, expert) for every evaluated token, on the host, without a sync on the critical path.
   The upstream upload-on-miss cache in `src/llama-moe-cache.cpp` already sees the selected ids on the host through its
   copy callback (`copy()`, `plan()`); reuse that point or an equally cheap one. Counting must not break the overlap
   (part A must still end at the miss mask).
2. **LFU with decay:** halve all counts every 512 evaluated tokens; a non-resident expert enters only when its count
   exceeds the coldest resident's in its group (ties stay out); evict that coldest one. Never a group smaller than one
   window's working set.
3. **Throttled uploads between graph evaluations:** at most N experts per ubatch (env `GGML_MOE_HYBRID_ADMIT`, default
   8), issued asynchronously on the cache backend's stream after the graph that counted them and before the next one,
   together with the device slot-map update. A slot is never rewritten while a graph that reads it is in flight; prove
   the ordering in a comment and a test. Misses are still computed on the CPU in the same step; no synchronous upload
   on a miss.
4. **Warm start:** the frozen profile (if given) fills the cache at initialization as in round 5; adaptation starts from
   there. Without a profile the cache starts empty and fills by admission.
5. **Statistics** printed at context teardown at WARN level (the server hides INFO): hits, misses, hit rate, uploads,
   MiB uploaded, per token.

Off by default behind `GGML_MOE_HYBRID_ADAPTIVE=1`; with it off, round 5 behaviour is unchanged bit for bit.

## Checks you can do in the sandbox

- A host-side replay test: feed the round-4 trace
  (`/opt/llm-gateway/data/build/llama-codex2/task-logs/round4-hitmiss/qwen-routes-r1.routes.jsonl`, rows
  `{topic, token, layer, ids}`) through the C++ policy object (one ubatch per token, slots 1805 in one group, no
  device) and compare hit rate and uploads per token with `sim_lfu.py` (44.2 %, 5.3). Small differences from group
  boundaries are fine; explain any above one point.
- CPU `test-backend-ops` for the touched ops, a CUDA build, `git diff --check`.

## What Claude will run afterwards (so leave the switches and logs ready)

llama-bench pp4 / pp1 (off / on / on / off), perplexity `-ub 4` and `-ub 1` on test-de, and the end-to-end server
comparison with MTP 3 against Strata (`/opt/llm-gateway/data/moe-quant/qwen38/stage_b.py`).

## Deliverable

`NOTES-round6.md`: design, the ordering argument for uploads, the replay result against the simulation, test results,
open risks. Do not touch other worktrees, `/opt/agent-tools`, Strata or the gateway's configuration.
