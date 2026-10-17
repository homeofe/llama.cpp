# Round 7: warm the expert cache from the prompt, a static base set, and the MTP draft's experts (worktree `llama-l5`)

You continue rounds 5 and 6 in `/opt/llm-gateway/data/build/llama-l5` (branch `l5-cache`). Read `NOTES-round6.md`,
`AGENTS.md`, `CONTRIBUTING.md`, `skills/code-review/SKILL.md`. Wall-clock budget: **3 hours**. Your sandbox has no
network, no view of other processes and read-only git metadata: do not ask for access, do not run GPU work; build,
run CPU tests and host-side replays. Claude runs the GPU tests and commits. Save your diff as
`task-logs/round7/round7.patch`.

## Where we are (measured by Claude, 2026-10-17)

- Round 6 fixed and measured (commit 36a79b641, Claude fixed two runtime bugs: the routing collector asserted one
  thread, the CPU pool runs custom ops on every thread; and profile entries of groups dropped in adaptive mode aborted).
  Long German text, 2004 slots: **59-61 % hits, 4-token windows +50 %, single tokens +27 %**, perplexity +0.38-0.44 %
  with admit 32.
- **Server with MTP 3** (Qwen3.8, `-ub 256` so 1501 slots / 2.9 GB fit beside the draft head; VRAM is full at
  10.7 GB): code 40.5 -> 43.7 tok/s, English 33.1 -> 37.3 (round 1 of 3), but only **20-24 % hits**: the benchmark is
  8 short prompts of changing topic, 256 tokens each, and admission (32 per ubatch, decay every 512 tokens) cannot
  fill 1500 slots in time. Strata writes 67 / 61 here with 53 % GPU hits plus 15 % read over PCIe.
- The MTP draft round costs ~15-20 ms of a ~92 ms round; the draft model's routed experts (3.2 GB) all run on the CPU
  (`-cmoed`), and `common/speculative.cpp` sets `moe_cache_size = 0` for the draft context.
- Four reviewers (`/opt/llm-gateway/patches/llama.cpp/moe-cache/reviews-round2/`) agree on the order below.

## This round

1. **Route log + replay first (decides the rest).** A switch `GGML_MOE_HYBRID_TRACE=<file>` that appends every
   evaluated ubatch's routing (context id, ubatch size, token positions, layer, selected ids, whether it was prefill or
   a verify/decode window) in a compact binary or JSONL form, without a sync on the critical path (reuse the round-6
   collector). Extend the host replay (`test-moe-cache`) to replay such a log with: current LFU; **prompt warming**
   (after a prefill ubatch, admit up to `GGML_MOE_HYBRID_ADMIT_PREFILL` experts, default 1024, hottest of the prompt
   first); **a static base set** (a fraction of the slots, `GGML_MOE_HYBRID_STATIC_PCT`, filled from a multi-topic
   profile and never evicted) plus the LFU remainder; and decay 64/128/512. Report hits, uploads and upload MiB per
   token, and hits in the first 32 decode tokens after each prompt.
2. **Prompt warming** in the runtime: prefill ubatches (more than 8 tokens, on the ordinary path today) still count
   routes; after the last prefill ubatch of a request, admit up to `GGML_MOE_HYBRID_ADMIT_PREFILL` experts in one
   stream-ordered burst (cost about 1024 x 2.05 MB / 12 GB/s = 175 ms of time-to-first-token; make it configurable and
   log the time). Same ordering rules as round 6.
3. **Static base set** in the runtime: `GGML_MOE_HYBRID_STATIC_PCT` of the slots are filled from the profile at start
   and excluded from eviction.
4. **The draft's experts:** let the draft context have its own cache (its own budget `GGML_MOE_HYBRID_DRAFT_MIB`,
   default 0 = off; its own profile or adaptive cold start), so the MTP head's routed experts can be computed on the
   GPU too. Plumbing in `common/speculative.cpp` and the context parameters; the target's cache must not be disturbed.
   Today's draft-profile mismatch guard (a profile naming layers without experts leaves the cache off) must keep
   working for configurations without a draft budget.

Everything off by default; with the new switches off, round 6 behaviour unchanged bit for bit.

## Checks in the sandbox

Replays of the round-4 trace (as in round 6) and of a synthetic log of 8 short prompts with topic changes (generate it
from the round-4 trace: 8 segments of 40 prompt tokens + 256 decode tokens, cycling the four topics) for every policy
above; CPU `test-backend-ops` for the touched ops; the host ordering test extended to a prefill burst; CUDA build;
`git diff --check`.

## Deliverable

`NOTES-round7.md`: the replay table (policies x workloads: hits, first-32-token hits, uploads per token), design and
ordering argument for the burst, the draft-cache plumbing, test results, the switches Claude should run, open risks.
