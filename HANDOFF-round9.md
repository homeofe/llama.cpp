# Handoff round 9: GPU-hit / CPU-miss expert cache, now with direct GPU access (worktree `llama-l5`)

The owner starts you directly (not through Claude Code), with full access, so you can run on the RTX 2080 Ti yourself.
Claude does not touch the GPU while you run. Read first: `NOTES-round5.md` to `NOTES-round8.md` in this worktree,
`/opt/llm-gateway/patches/llama.cpp/moe-cache/NOTES.md` (all measurements so far), `AGENTS.md`, `CONTRIBUTING.md`,
`skills/code-review/SKILL.md`. Wall-clock budget: **4 hours**.

## Rules (the machine is shared; these are not optional)

- **Where you may write:** this worktree (branch `l5-cache`) and `/opt/llm-gateway/data/moe-quant/`. Do not touch other
  worktrees, `/opt/agent-tools`, Strata (`~/.openclaw/...`), the gateway's code or `data/settings.json`, or the live
  runtimes under `/opt/llm-gateway/data/runtimes/`. Commit WIP to `l5-cache` at each stage (no `Co-Authored-By` line).
  No push, no PR.
- **GPU:** before every GPU step free and hold it at the gateway, release it right after:
  `curl -s -X POST -H "Authorization: Bearer $(cat /opt/llm-gateway/data/api-key)" http://127.0.0.1:8790/api/v1/gpu/free`,
  then `.../gpu/hold` with `-H 'Content-Type: application/json' -d '{"reason":"codex: round 9","ttlS":900}'`, and
  `curl -s -X DELETE ... /api/v1/gpu/hold`. Never print the key. Never hold the GPU while building or reading code. Do
  not start if another process uses more than 300 MiB of VRAM (`nvidia-smi --query-compute-apps=pid,used_memory
  --format=csv,noheader`).
- **Quiet machine:** other agents use this box. Gate on foreign processes, not on load average: our own llama-server
  alone pushes the 1-minute load above 3, so a load-average gate blocks forever (it did on 2026-10-17). The harness
  `/opt/llm-gateway/data/moe-quant/qwen38/stage_b.py` already does it right without `--load-gate`: under one busy foreign
  core before a request, foreign CPU under one core during it, otherwise the request is repeated. Use it as is.
- **Processes:** stop only your own, by PID. Never `pkill -f <pattern>` from a shell whose command line contains the
  pattern (it kills that shell).
- **Progress for the owner:** after each stage append one German line to `/opt/llm-gateway/data/moe-quant/NOTE.log`
  (`echo "$(date +%H:%M:%S) ..." >> /opt/llm-gateway/data/moe-quant/NOTE.log`); the owner follows `ACTIVITY.log`.
- **Quality rule (owner):** cache-off must stay bit-exact; with the cache, perplexity on
  `/opt/llm-gateway/data/build/texts/test-de.txt` (12 chunks, ctx 512, `-ub 4`) at most +0.5 % against cache-off.

## Where we are (all measured on this machine)

| tok/s, MTP 3, 8 prompts, median of 12 | code | English | notes |
|---|---|---|---|
| Qwen3.8-Flash-Next, no cache (`-ub 256`) | 40.1 | 32.8 | |
| Qwen3.8, round 7 (`q-warm-draft`): adaptive LFU, prompt warming, decay 128, 2.5 GB target + 0.5 GB draft cache | 47.0 | 39.8 | ~25 % hits in the server; 60 % on long text |
| Ling-3.0-flash IQ4_XS, round 7 (`ling-cache-mtp3`): 5 GB adaptive cache, prompt warming | 50.8 | 47.5 | drafts only 68 / 53 % |
| Strata (a separate engine for Qwen3.8) | 70.8 | 61.0 | 53 % GPU hits + 15 % read over PCIe |

Round 8 (you, offline) added byte-value admission, soft per-layer budgets, speculative verdicts in the trace, the PCIe
harness and the draft-ubatch configs. Claude ran on the GPU:
- **PCIe zero-copy reads lose:** misses only, one 4-token window: CPU alone 31.7 ms; GPU reads 10 / 20 / 30 % of the
  miss bytes: 44.0 / 53.5 / 70.2 ms (`task-logs/round8/pcie-run/claude-gpu.jsonl`). Dropped; do not build the graph
  path. (Strata's "15 % over PCIe" must be bulk copies, not MMVQ over mapped memory.)
- Real route traces of the benchmark: `/opt/llm-gateway/data/moe-quant/traces/` (`qwen38-bench8.jsonl`,
  `ling-bench8.jsonl`, `qwen38-long-de.jsonl`; `partial/` is an aborted attempt, ignore it).
- A quick round of the round-8 configs (`q-r8-*`, `ling-r8-*`): `/opt/llm-gateway/data/moe-quant/qwen38/smoke-r8/`
  (`summary.py smoke-r8/runs.jsonl`). Claude reports these to the owner; read them before deciding.

## Tasks, in order (measure each before the next; stop a direction when its bound says no)

1. **Policy defaults from the real traces:** replay the three traces with the production policy (decay, prefill cap,
   per-ubatch cap, byte value, soft budgets, accepted-only learning). Pick the configuration that wins on both models
   without losing on the long trace; then the end-to-end comparison against `q-warm-draft` and `ling-cache-mtp3`,
   three interleaved rounds (`stage_b.py --rounds 3 --only ...`), plus the perplexity check.
2. **More slots:** `--spec-draft-n-ubatch 32` with a larger target budget (round 8 computed +190 slots for Qwen, +83 for
   Ling). Measure what actually fits (VRAM must stay under about 10.7 GB) and what it gains.
3. **Ling's drafter:** its MTP head is accepted only 68 % (code) / 53 % (English), against Qwen's 92 / 69 %. A DSpark
   drafter exists (`inclusionAI/Ling-3.0-flash-dspark`, 2.7 GB safetensors; our fork has `--spec-type draft-dspark`
   and a `--dspark` export path for DeepSeek-V4 in `convert_hf_to_gguf.py`). Check whether conversion works for Ling
   (download into `/opt/llm-gateway/models/Ling-3.0-flash-GGUF/dspark/`), then measure acceptance and tok/s with and
   without the cache. Bound first: tokens per round at the measured acceptance against the vendor's claimed accept
   length 6.3-6.4 (math).
4. **CPU miss path:** from the traces, the histogram of how many tokens of a 4-token window hit the same *missed* expert
   (per quant type). Only if the mean multiplicity of missed experts is at least about 1.6 is a multi-token CPU kernel
   for Qwen3.8's IQ types (IQ4_NL, IQ3_XXS, IQ3_S, IQ2_S) worth building; then build and measure it with the existing
   checkers' style (exact mode, mutation cuts).
5. **CPU bandwidth spread** (35-67 GB/s): a short affinity / thread-count sweep of the CPU miss work (same model,
   `OMP_PLACES`, `-t`, `--cpu-mask`), only if task 4 says the CPU side still dominates.

## Deliverable

`NOTES-round9.md`: per task what you measured, the chosen defaults and the end-to-end table (Qwen3.8 and Ling against
round 7 and Strata), perplexity, what did not work, and the exact settings for the gateway. Commit at each stage.
