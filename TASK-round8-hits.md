# Round 8: more expert-cache hits and a second bandwidth (worktree `llama-l5`)

You continue rounds 5-7 in `/opt/llm-gateway/data/build/llama-l5` (branch `l5-cache`). Read `NOTES-round7.md`,
`AGENTS.md`, `CONTRIBUTING.md`, `skills/code-review/SKILL.md`. Wall-clock budget: **3 hours**.

**This round you measure on the GPU yourself (the owner asked for it).** Your sandbox now allows local network (the
gateway on 127.0.0.1:8790) and writes to `/opt/llm-gateway/data/moe-quant/` and this worktree's git metadata. If the
GPU device itself is not reachable from the sandbox, say so in the notes and do the offline parts. Rules:
- Work only in this worktree and in `/opt/llm-gateway/data/moe-quant/` (traces, harness output). Do not touch other
  worktrees, `/opt/agent-tools`, Strata, the gateway's code or configuration, or anything under `/home`. No push, no PR.
  Commit WIP to `l5-cache` at each stage, without a `Co-Authored-By` line.
- GPU: free and hold it at the gateway before each GPU step and release it right after (`curl -X POST
  http://127.0.0.1:8790/api/v1/gpu/free -H "Authorization: Bearer $(cat /opt/llm-gateway/data/api-key)"`, then
  `.../gpu/hold` with `{"reason":"codex: round 8","ttlS":900}`, `DELETE .../gpu/hold`; never print the key). Never hold
  it while building or reading code. Do not run anything while another process uses more than 300 MiB of VRAM.
- Measure only on a quiet machine: you cannot see other processes from the sandbox, so gate on the 1-minute load
  (below 3 at the start and at the end of each step; repeat a step otherwise). Interleave control and candidate.
  Claude repeats the final end-to-end comparison with the full foreign-process gate.
- End-to-end numbers: `/opt/llm-gateway/data/moe-quant/qwen38/stage_b.py` (configs at the top, `--only`, `--rounds`;
  `summary.py` prints medians). Add your configs there; keep the existing ones. Correctness: perplexity on
  `/opt/llm-gateway/data/build/texts/test-de.txt` (12 chunks, ctx 512, `-ub 4`), the cache off must stay bit-exact.
- Append a German one-line milestone to `/opt/llm-gateway/data/moe-quant/NOTE.log` after each stage (the owner follows
  `ACTIVITY.log`, which collects it): `echo "$(date +%H:%M:%S) ..." >> /opt/llm-gateway/data/moe-quant/NOTE.log`.
- Stop only your own processes. Save your final diff as `task-logs/round8/round8.patch`.

## Where we are (measured by Claude on the GPU)

- Round 7 works on the GPU. Server, Qwen3.8 with MTP 3, `-ub 256`, 8 short prompts of changing topic (one quick round):
  no cache 40.6 / 33.3 tok/s (code / English), round 6 adaptive 43.9 / 37.8, **round 7 prompt warming + decay 128
  47.1 / 38.7**, plus a 512 MiB draft cache (target 2560 MiB) 47.6 / 39.8. Strata 68.8 / 61.7.
- Ling-3.0-flash (bailingmoe3, ~3 MB experts, MTP head in the same file): MTP 3 without cache 42.6 / 38.1, **with a
  5 GB adaptive cache and prompt warming 50.9 / 47.4**. Its drafts are accepted only 68 / 53 %.
- Prompt bursts: 1024 uploads, 2.0 GiB (Qwen) / 3.0 GiB (Ling), enqueue 120-180 ms.
- Real server traces: record them first (`bash /opt/llm-gateway/data/moe-quant/record-traces.sh`, about 15 minutes on
  the GPU: the 8-prompt benchmark for Qwen3.8 and Ling, plus a long German session) into
  `/opt/llm-gateway/data/moe-quant/traces/`. Use them instead of the synthetic workloads; keep the old ones as a
  regression check.

## This round

1. **Policy tuning on the real traces** (host replay, production policy object): decay, prefill cap, per-ubatch cap,
   **byte-weighted value** (Qwen3.8's experts differ in size by quant type: IQ4_NL roughly twice IQ2_S; a hit on a big
   expert saves the CPU more; rank admission by bytes saved per byte uploaded), soft per-layer budgets, and admission
   that ignores routes of rejected draft tokens if the trace marks them (if it does not, add that to the trace). Report
   for each: decode hits weighted by bytes, uploads MiB per token, first-32-token hits. Pick a default that wins on
   both models' traces without losing on the old long-text trace.
2. **GPU reads of misses over PCIe, prototype for a bound:** the routed experts are in pinned host memory (`-lm none`);
   a CUDA masked `MUL_MAT_ID` path that reads a chosen subset of the *missed* experts directly from mapped host memory
   (zero copy) while the CPU computes the rest. First a standalone harness (`tests/` or `task-logs/round8/`) that
   times, for Qwen3.8's real expert shapes and types and a 4-token window, CPU-only misses against a split of 10 / 20 /
   30 % of the miss bytes read by the GPU over PCIe concurrently. Run it. Build the graph path (switch off by default)
   only if the harness shows the split saves more than 3 ms per 4-token window, and then measure it end to end.
3. **VRAM for more slots:** measure (CPU side, from the allocation logs Claude can produce, or by reading the code) what
   the draft context's 1 GB compute buffer at `-ub 512` (0.5 GB at 256) is sized for; if a draft never evaluates more
   than a few tokens per ubatch after its prompt pass, propose a safe smaller reservation (draft-specific ubatch, or
   prompt pass in chunks), implement it behind a switch, and measure how many more slots fit.

Everything off by default; with the new switches off, round 7 behaviour unchanged bit for bit.

## Deliverable

`NOTES-round8.md`: replay tables on the real traces (and the old ones), the chosen default and why, the PCIe harness
results, the VRAM change, the end-to-end numbers (Qwen3.8 and Ling, best configuration against round 7's
`q-warm-draft` / `ling-cache-mtp3`, three interleaved rounds), perplexity, tests, open risks.
