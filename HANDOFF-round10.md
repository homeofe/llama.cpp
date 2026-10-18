# Handoff round 10: confirm Ling DSpark, panels for Ling, profile a whole Qwen3.8 server round (worktree `llama-l5`)

The owner starts you directly with full access; Claude keeps off the GPU while you run. Read first:
`NOTES-round9.md` (your last round), `/opt/llm-gateway/patches/llama.cpp/moe-cache/NOTES.md` (sections 7-11),
`AGENTS.md`, `CONTRIBUTING.md`, `skills/code-review/SKILL.md`. Wall-clock budget: **4 hours**.

## Rules (unchanged from round 9, plus two)

- **Write only** in this worktree (branch `l5-cache`), in `/opt/llm-gateway/data/moe-quant/` and, for task 2 only, in
  the second gateway's home `/opt/llm-gateway/data/bench-home-keep75ud/`. Do not touch other worktrees,
  `/opt/agent-tools`, Strata, the gateway's code, the live `data/settings.json`, the live runtimes or the live bench
  history `data/bench/results.jsonl`. Commit WIP to `l5-cache` at each stage (no `Co-Authored-By`). No push, no PR.
- **No system changes this round:** no sysctl, no kernel modules, no governor, nothing via sudo. If a measurement would
  need one, write down why and skip it.
- **GPU:** free and hold at the gateway before each GPU step (`/api/v1/gpu/free`, `/api/v1/gpu/hold` with
  `{"reason":"codex: round 10","ttlS":900}`, renew for longer steps, `DELETE /api/v1/gpu/hold` right after; key in
  `/opt/llm-gateway/data/api-key`, never print it). Never hold it while building. Do not start if another process uses
  more than 300 MiB of VRAM.
- **Quiet machine:** gate on foreign processes (the default gate of `qwen38/stage_b.py`), not on the load average.
- **Processes:** stop only your own, by PID; never `pkill -f` a pattern contained in your own command line.
- **Progress:** one German line per stage to `/opt/llm-gateway/data/moe-quant/NOTE.log`.
- **Quality:** cache off bit-exact; with cache and drafter, perplexity on test-de (12 chunks, ctx 512, `-ub 4`) at most
  +0.5 % against cache-off; output hashes reported.

## Tasks, in order

1. **Confirm Ling DSpark and find a setting that does not lose on English** (`stage_b.py`, 3 interleaved rounds,
   12 requests per category per config). Control `ling-cache-mtp3`. Candidates: your round-9 DSpark7 + 4608 MiB cache;
   shorter draft windows (`--spec-draft-n-max` 3, 4, 5) and confidence cut-offs (`--spec-draft-p-min` 0.3, 0.5, 0.7)
   with the same cache. Screen with one round first, then run the best two plus the control for three rounds. Report
   tok/s, acceptance, mean accepted length, output hashes per prompt. The goal is one setting that beats MTP on code and
   at least ties it on English; if none exists, say so and give the code-only setting.
2. **Our own panels for Ling-3.0-flash** with the best setting from task 1, through the second gateway (8791):
   pattern `/opt/llm-gateway/data/bench-home-keep75ud/run-panels-mmq.sh` (`npm run bench -- --card --hard --agent
   --runs 3 --no-speed <model>`). Add a Ling entry to that home's settings only (binary `llama-l5/build-cuda/bin/
   llama-server`, the args and environment of the chosen config; note from earlier work: a second model entry for the
   same GGUF needs a hard link because the scan deduplicates by realpath). Compare with the existing panel results of
   Qwen3.8 (`iq3_s`, 56/56 hard, 69/72 agent) and Bolt (41/42, 36/36) in the live history (read only).
3. **Profile one whole Qwen3.8 server round** (MTP 3 + the round-7 cache, `q-warm-draft`, decode phase of one code and
   one English prompt): Nsight Systems on the running llama-server (CUDA graphs off for visibility, a second run with
   graphs on for wall time), plus the scheduler's own trace if useful. Break a round into: the three MTP draft steps
   (GPU, CPU experts, waits), the 4-token verify window (GPU dense, GPU hits, CPU misses, copies, waits), sampling and
   host overhead. One table in ms per round, then the three largest levers with an upper bound each, and compare with
   Strata's figures in `/opt/llm-gateway/patches/llama.cpp/strata-parity/PLAN-DE.md` (41-51 ms per round). Do not build
   anything for it in this round.
4. **Only if time remains:** the missing persistent threadpool of the MTP draft context
   (`common_speculative_init_result`, see your round-9 notes): implement it behind the existing paths, test placement
   and teardown, verify output hashes and perplexity, measure end to end (3 rounds); adopt only with at least +3 %.

## Deliverable

`NOTES-round10.md`: the confirmed Ling table and the recommended setting (code-only or general), the panel results
next to Qwen3.8 and Bolt, the round profile table with the three levers and their bounds, task 4 if done, exact
configurations, quality checks, open risks.
