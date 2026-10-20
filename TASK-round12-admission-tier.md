# Round 12: the admission tier - closing Strata's hit-rate edge on the CPU expert gate

Machine: RTX 2080 Ti (sm_75), Threadripper 3960X, shared. Worktree
`/opt/llm-gateway/data/build/llama-vibe`, branch `vibe-parity` (start from the round-1 final
stack: bf16-as-F16 + MMVQ nwarps 2 + int8 KV + 1779-slot cache + two bit-exact fusions + async
uploads, 51.3 / 42.7 tok/s against the deployed 48.4 / 40.1).

Read first: `NOTES-vibe-round1.md` (whole file, in English), `FACTS-bf16-as-f16.md`, and
`/opt/llm-gateway/patches/llama.cpp/moe-cache/NOTES.md` sections 5-13.

**Budget: 1 day wall clock. Almost all of it is CPU-only work that runs fine under foreign
load (npm runs). Only the final confirmation (~1 h) needs a quiet machine; stage_b's quiet
gate waits automatically (3 s polls).**

## Why (the measured basis, cite it)

- The gap analysis: 84% of GPU idle in the decode window sits in 21,558 gaps before
  `k_bin_bcast` - the per-layer wait for the CPU experts. The CPU expert phase (~35-45 ms of a
  ~70 ms round) is the round's dominant gate, almost fully exposed.
- The slot sweep in `bound/sim_policy.py` on the real bench-8 routing trace: hit rate is
  LINEAR in resident slots (1418 -> 43.7%, 1779 -> 48.6%, 2048 -> 51.9%, 2400 -> 55.7%
  cumulative in the sim). Strata's measured 50-57% is a cache-size effect at their 1805 slots -
  but at OUR 1779 slots we realize only ~40-48% in the server, and their steady-state log goes
  to 67% over request history. The sim (LFU with decay, our exact policy) reproduces our
  trajectory, so the LFU knob space is closed (decay faster/slower, admit 16-256, prefill
  256-4096: all measured, all flat or negative). The remaining difference to Strata's
  long-run 67% is STRUCTURAL: their adaptive tier keeps experts differently across requests.

## Measured ceilings (do these three computations first; they are already in the repo's history)

- The per-request ADAPTIVE oracle at 1779 slots (keep the top-1779 experts of each request with
  perfect foresight) on the bench-8 trace: **55.5%**. Our LFU simulates 48.6% cumulative. The
  maximum any policy can gain on THIS benchmark is therefore ~7 hit points, not 20 - Strata's
  67% is the DEPLOYED log (multi-turn conversations), unreachable on 8 discrete prompts.
  Strata's same-bench measurement was 50-57%, i.e. AT the oracle ceiling.
- The within-request sequential reuse with an INFINITE cache is 91.1% - the binding constraint
  is the slot count, not the policy. One request touches up to 15,286 distinct experts against
  1779 slots.
- TRACE SEMANTICS WARNING: the `cache` field in `traces/qwen38-bench8.jsonl` is
  `l.use_cache` (whether the layer's ubatch ran through the cache path at all), NOT a hit
  flag (`src/llama-moe-cache.cpp:1112`). The trace does not record per-expert hits; a fresh
  trace with per-request hit accounting needs a small logging addition or the `slots` field
  decoded. Do not parse it as hits (round 1 did and got a meaningless 95.8%).
- Our real server realizes ~40.6% cumulative against the sim's 48.6% - an 8-point
  implementation gap (warm-up passes, multi-context effects, per-ubatch admission timing).
  Closing SIM-TO-SERVER may be cheaper than closing policy-to-oracle: instrument WHERE the
  8 points go before writing any new policy (the `adaptive:` statistics lines plus the
  per-request trajectories in the server logs are the data).

## The task

Find, by simulation on the real traces, an admission policy that materially beats our LFU on
the benchmark's request history, then implement it behind an environment switch in
`src/llama-moe-cache.cpp` (the LFU class there: `llama_moe_cache_lfu`), measure it end to end,
and adopt it if it clears the gate.

### Phase 1 - reconstruct Strata's behavior (CPU-only, 3-4 h)

1. Extract Strata's per-request hit-rate trajectory and upload volumes from
   - `/opt/llm-gateway/data/build/llama-vibe/task-logs/vibe-round1/strata-bench/engine.log`
     (my run: 18.6% first request rising to ~57% over 28 requests),
   - `/opt/llm-gateway/data/logs/iq3_s.log` (the deployed long-run log: 38% after the first
     request, 67% by the fourth),
   - Strata's source: `/opt/llm-gateway/data/runtimes/strata/0.1.41-local-1cb045cb/src/core/
     expert_cache.cpp` (their slot storage and residency table; read the whole file) and
     `src/core/peer_experts.cpp`, `remote_experts.cpp`.
2. Write down their policy as pseudocode: when do they admit, what do they evict, what do they
   pin (their static profile `expert-profile.bin` + the adaptive tier), what survives between
   requests, do they decay, how do they handle the prompt burst.

### Phase 2 - simulate (CPU-only, 3-4 h)

Extend `bound/sim_policy.py` (it replays the bench-8 trace through our exact LFU; keep the
old results as the control). Add candidate policies, parameter-swept against the round-1
numbers:

- ours (decay 128, admit 32, prefill 1024): 48.6% cumulative at 1779 slots - the control;
- LRU instead of LFU (their log suggests recency matters across requests);
- LFU without decay + LRU tiebreak, or an aging window (keep counts of the last N tokens);
- a static pinned base set (the round-4 profile scored only 10% alone at 1805 slots, but
  PINNING it and letting the adaptive tier own the rest may be different: simulate
  pinned+adaptive);
- admission by router probability if the trace carries it (check the trace format in
  `bound/sim_policy.py::load()` - the round-8 traces have per-token ids only; the round-4
  routes at `/opt/llm-gateway/data/build/llama-codex2/task-logs/round4-hitmiss/` may carry
  weights);
- per-layer fairness: cap slots per layer (the code has `layer_soft_limit`).

Also simulate over LONGER histories: concatenate the bench-8 trace 10x (the deployed log's
67% is a many-request effect; the sim must reproduce the rise before a policy change is
believed).

Acceptance for phase 2: a policy with >= +5 cumulative hit points over ours at the same 1779
slots, reproduced over 3x and 10x trace repetitions, with the rise-over-history shape matching
Strata's log better than ours does.

### Phase 3 - implement (4-6 h, CPU build + quick GPU checks)

In `src/llama-moe-cache.cpp` behind `GGML_MOE_HYBRID_POLICY=<name>` (default: the current
LFU; switch off = bit-for-bit). The class is self-contained (counts, slot_of, candidates);
a second policy beside it is a contained change. If the winning policy needs the router
probabilities, thread them through the observe() call - check what the CUDA-side collector
passes (grep `observe(` in ggml/src/ggml-cuda and src/).

Correctness: the cache policy changes WHICH experts are resident - it cannot change outputs
beyond the GPU/CPU rounding that the cache already causes (the round-11 screens documented
that admission changes shift outputs at near-ties). Prove: 12-chunk test-de perplexity with
the new policy within +0.5% of cache-off (3.8093 is the reference), and greedy outputs
compared between the old and new policy with every difference explained as a cache-state
effect (draft counts compared per prompt).

### Phase 4 - measure (1 h quiet window)

stage_b: control `q-vibe-async` (the round-1 final stack) vs `q-vibe-policy` (same stack +
the new policy env). Three interleaved rounds, quiet gate, n=12 per config and category.
Also run the hit-rate trajectory out of the server logs (the `adaptive:` statistics lines)
to confirm the sim's prediction arrived.

**Gate: +3% on both code and english** (the round-1 gate; a policy change is cheap to keep,
but it must clear the same bar). If the sim predicts +5 hit points, expect roughly +6-8%
end to end - report the prediction error honestly.

### Measured during round 1 (already done - do not repeat)

The per-ubatch sim (`bound/sim_policy.py`, modeled with admission at ubatch boundaries) reproduces
the server at admit 32 (38.3% sim vs 40.6% server - the earlier 8-point "implementation gap" was
the sim modeling admission per-token). It predicts +8-9 hit points at admit 128-256. The REAL
server screen (`vibe-adm-screen` in the moe-quant folder) says otherwise:

| config | code | english | target hit rate |
|---|---:|---:|---:|
| admit 32 (current) | 51.0 | 42.7 | 40.6% |
| admit 128 | 48.1 (-5.7%) | 40.0 (-6.3%) | 42.6% (+2 points only) |
| admit 256 | 49.4 (-3.1%) | 40.7 (-4.7%) | - |

The upload/eviction cost (up to 280 MB/round of HtoD at 128, plus reader-fence waits in the async
path) eats far more than the small realized hit gain. Admission volume is a dead knob: 32 is the
measured optimum on this stack too. The sim must be extended to model upload costs before its
policy predictions are trusted (admissions x 2.18 MB at 12.9 GB/s, plus a fence-wait factor).

**Strata's cache policy, read from their source (`src/core/expert_cache.cpp:653`): FIRST
TOUCH, NO EVICTION, EVER.** `if (next_free_ >= slots_) return kNotResident; // full: no
eviction, deliberately` - the first `slots` distinct (layer, expert) pairs to be looked up
are admitted in arrival order, and once the cache is full it is FROZEN for the lifetime of
the process. No scoring, no decay, no LFU. The per-layer variant is the same with a per-layer
quota. Consequences:

- Their deployed-log 67% is a REPEAT-MEMORY effect: recurring conversations hit a frozen set
  at 100%. Simulated on our bench trace (`bound/`, 3 rounds): first-touch-freeze scores only
  **10.3% cumulative** - BUT 100% on the second occurrence of the same prompt (our LFU gets
  2% there). It is a memory, not a cache: perfect for repeats, poor for novel requests.
- Our LFU (decay 128) is the opposite: 38.3% on novel requests, near-zero cross-round memory.
- **The hybrid is the round-12 candidate: pin the all-time top-K (a profile learned from the
  deployed workload - the `GGML_MOE_HYBRID_TRACE` + `PROFILE` mechanism already exists,
  round 7) and let the LFU own the remaining slots.** The pinned set must be evict-protected
  (push them to +inf in the eviction heap). K to sweep: 0 / 300 / 600 / 900.
  The first sim attempts are in the round-1 history but buggy (their pin-0 reference must
  reproduce 43.7% per-token / 38.3% per-ubatch - fix the eviction logic until it does, then
  trust the pin sweep).
- Caution: the stage_b bench repeats its 8 prompts 3 times, so a pin-happy policy games the
  benchmark. Report the round-1 (novel) average separately from the steady state, and judge
  the policy on BOTH: it must not lose novel-request performance (round-1 >= 38%).

### Stop rules

- The realistic ceiling is +7 hit points (55.5% oracle), but part of it is already in the 40.6%
  realized number - the honest headroom statement is: the oracle at CURRENT admission cost is
  ~47-48% (the unlimited-admission sim), so a cost-neutral policy change is worth at most
  ~+6-7 real hit points, of which the sim's predictive power is now known to be weak at the
  margins (it predicted +8 and reality gave +2). Calibrate expectations accordingly.
- Stop and report if Strata's source turns out to implement exactly our LFU (then the 67% is
  a workload artifact of the deployed log's longer conversations, not a policy gap - quantify
  the conversation length effect in the sim and document it).
- The 10x-repetition test exists to catch exactly this: if the rise to 67% is a
  conversation-history effect that our 8 x 256-token benchmark cannot show, the whole premise
  dies - report it, do not chase it.

## Deliverables

`NOTES-round12.md` (English, every number with its source, negative results included), the
patch against vibe-parity, the sim extensions in `bound/`, no push (the owner pushes), no
live settings change. Append German one-liners to `/opt/llm-gateway/data/moe-quant/NOTE.log`
per phase.
