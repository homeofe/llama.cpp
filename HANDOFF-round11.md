# Handoff round 11: hide the cache uploads, Ling at 32K context, a gateway candidate (worktree `llama-l5`)

The owner starts you directly with full access; Claude keeps off the GPU while you run. Read first:
`NOTES-round10.md` (your last round), `/opt/llm-gateway/patches/llama.cpp/moe-cache/NOTES.md` (sections 7-12),
`AGENTS.md`, `CONTRIBUTING.md`, `skills/code-review/SKILL.md`. Wall-clock budget: **4 hours**.

## Rules (as in round 10)

- **Write only** in this worktree (branch `l5-cache`), in `/opt/llm-gateway/data/moe-quant/` and, for the panel in
  task 2, in `/opt/llm-gateway/data/bench-home-keep75ud/`. Do not touch other worktrees, `/opt/agent-tools`, Strata, the
  gateway's code, the live `data/settings.json`, the live runtimes under `data/runtimes/` or the live bench history.
  Commit WIP to `l5-cache` at each stage (no `Co-Authored-By`). No push, no PR.
- **No system changes:** no sysctl, kernel modules, governor. Profiling under sudo as in round 10 is fine; say so.
- **GPU:** free and hold at the gateway before each GPU step (`{"reason":"codex: round 11","ttlS":900}`, renew, release
  right after; key in `/opt/llm-gateway/data/api-key`, never print it). Never hold it while building. Do not start if
  another process uses more than 300 MiB of VRAM.
- **Quiet machine:** the default foreign-process gate of `qwen38/stage_b.py`; repeat contaminated passes.
- **Processes:** stop only your own, by PID. **Progress:** one German line per stage to
  `/opt/llm-gateway/data/moe-quant/NOTE.log`.
- **Quality:** cache off bit-exact; perplexity on test-de (12 chunks, ctx 512, `-ub 4`) at most +0.5 % against
  cache-off; output hashes reported.
- **Adoption gate:** a code change counts only with at least +3 % end to end over three interleaved rounds.

## Where we are

- Round 10 profile of a Qwen3.8 server round (MTP 3, round-7 cache): 81.7 / 76.1 ms (code / English), Strata 51 / 41.
  Drafting about 7.5 ms; target verification 70.8 / 65.5 ms. In the graph-off trace the verification contains **copies
  of 8.9 ms per round**: host-to-device 6.3 / 8.7 GB per request, about 107 MB per round at 12.4 GB/s, i.e. mostly the
  adaptive cache's admissions (up to 32 experts of ~2 MB per ubatch) enqueued on the **compute stream** after each
  graph, so the next graph waits for them. The experts are in pinned memory (`-lm none`), so the 2080 Ti's copy engine
  could move them while the SMs compute.
- Ling-3.0-flash DSpark n3 (p_min 0, target cache 4608 MiB, draft ubatch 32, ctx 8192): 66.5 / 51.9 tok/s against
  MTP 50.9 / 47.5; panels hard 39/42 (all three misses: `long-combine`, ~20k tokens rejected by the 8192 context),
  agent 36/36.

## Tasks, in order

1. **Bound the upload cost first (cheap):** same server and prompts, adaptive cache with `GGML_MOE_HYBRID_ADMIT` 32
   against 8 and against 0 after the prompt warm-up (prefill burst still on), one round each, Qwen3.8 and Ling. Report
   tok/s, hit rate and uploaded MiB per round. The difference between 32 and 0 at equal hit rate is roughly what a
   hidden upload could win; if it is below 3 %, stop task 1 and say so.
2. **Uploads on a separate copy stream:** admissions are copied on their own CUDA stream (pinned source, the copy
   engine) into slots that no in-flight graph reads; the slot map entry is published only after the copy's event has
   completed and only for graphs submitted after that (record an event per graph; a slot freed by eviction is reused
   only after every graph that could read it has finished). No host-side synchronization on the critical path. Keep the
   current behaviour behind the old path when the switch is off (`GGML_MOE_HYBRID_ASYNC_UPLOAD=1`, default 0). Extend the
   host ordering test to the two-stream protocol (old readers see old weights, new readers see new weights, no early
   publish; mutation cuts: publishing before the event, reusing a slot early). Then measure end to end, three
   interleaved rounds, Qwen3.8 (`q-warm-draft`) and Ling (DSpark n3), with output hashes and perplexity.
3. **Ling with a 32K context:** the DSpark n3 profile at `-c 32768`: measure VRAM (reduce the target cache budget only as
   far as needed to fit, report the slots lost), speed on the 8 prompts (one round, against the 8192 profile), and rerun
   the hard panel through the second gateway (8791) so that `long-combine` is answered. Report hard/agent scores.
4. **A gateway candidate, prepared but not applied:** write `task-logs/round11/gateway-candidate.json` with the exact
   model entry the live gateway would need for Ling-3.0-flash with DSpark (binary path to be decided by Claude, args,
   env, context, sampling from the model card, label) and, if task 2 is adopted, the same for Qwen3.8 on llama.cpp.
   Claude installs a runtime and edits the live settings only after the owner agrees.

## Deliverable

`NOTES-round11.md`: the upload bound, the two-stream design and its ordering proof, the end-to-end table (both models,
against round 10), perplexity, the 32K results and panel scores, the gateway candidate, open risks.
