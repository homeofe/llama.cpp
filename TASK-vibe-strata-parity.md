# Vibe round 1: where Strata's round is shorter, and closing the largest part in llama.cpp (worktree `llama-vibe`)

You work in `/opt/llm-gateway/data/build/llama-vibe` (branch `vibe-parity`, a copy of our fork's branch `l5-cache` at
e3b15507c: llama.cpp with a GPU-hit / CPU-miss expert cache, rounds 5 to 11 by Codex). Read first: this file,
`AGENTS.md`, `CONTRIBUTING.md`, `NOTES-round10.md`, `NOTES-round11.md`, and the measured history in
`/opt/llm-gateway/patches/llama.cpp/moe-cache/NOTES.md` (sections 3 to 13).
**Wall-clock budget: 3 hours.** Stage 1 alone must not take more than 75 minutes.

## Goal

The owner believes llama.cpp can reach Strata's speed on the same model and machine. Strata is another engine (MIT,
C++/CUDA) that runs the same two GGUF files. Your job is to **find out with measurements, not estimates**:
1. where, millisecond by millisecond, a Strata decode round is shorter than ours;
2. close the largest part in llama.cpp, or show with numbers why it cannot be closed here.

A negative result with clean numbers is a good result. Do not tune knobs we already measured (list below).

## Machine

- RTX 2080 Ti 11 GB (sm_75, CUDA 12.0, driver 580), PCIe 3.0 x16 (13.1 GB/s pinned).
- Threadripper 3960X, 24 cores / 48 threads, AVX2, no AVX-512; 125 GiB DDR4-3200 in 4 channels.
- The machine is shared. The owner's other programs (agent-tools JavaScript tests: `node` plus several
  `chrome-headless`) can start at any time and push the load above 70. **Measure only when it is quiet; otherwise wait.**

## Rules (you run without a sandbox: these rules are the only fence, keep them exactly)

- Write only in this worktree and in `/opt/llm-gateway/data/moe-quant/` (harness output, traces).
- Read anything you need. Do **not** change:
  - other worktrees under `/opt/llm-gateway/data/build/`;
  - the gateway's code, settings or runtimes (`/opt/llm-gateway/packages`, `apps`, `data/settings.json`,
    `data/runtimes`);
  - `/opt/agent-tools`, anything under `/home`, system settings.
- No `sudo`, no `pip install`, no downloads, no network beyond 127.0.0.1. No `git push`, no PR. Commit WIP on `vibe-parity`
  at each stage, without a `Co-Authored-By` line.
- **GPU:** free and hold it at the gateway before each GPU step, and release it right after:
  - free: `curl -s -X POST http://127.0.0.1:8790/api/v1/gpu/free -H "Authorization: Bearer $(cat /opt/llm-gateway/data/api-key)"`;
  - hold: `POST .../api/v1/gpu/hold` with `{"reason":"vibe: strata parity","ttlS":900}`, renewed at least every 10 minutes;
  - release: `DELETE .../api/v1/gpu/hold`, also in error paths.

  Never print the key, never write it into a file or log. Never hold the GPU while building or reading code. Do not run
  anything while another process uses more than 300 MiB of VRAM.
- **Quiet gate** for every timed run:
  - 1-minute load below 3 at the start and at the end;
  - no foreign process above one core during the run (`ps -eo pcpu,pid,comm --sort=-pcpu`; your own PIDs excepted).

  Repeat a run that fails the gate. Interleave control and candidate (A B B A), three rounds for any number you report.
- Builds: `nice -n 10 cmake --build build-cuda -j 24` (configure like `../llama-l5/build-cuda`, `CMAKE_CUDA_ARCHITECTURES=75`).
  Run model tools with `< /dev/null`. Stop only your own PIDs, never `pkill -f` or wildcards.
- Strata's config JSON contains a key: never print it whole; read single fields.
- After each stage append one German line to `/opt/llm-gateway/data/moe-quant/NOTE.log` (the owner follows it):
  `echo "$(date +%H:%M:%S) Vibe: <was, Zahl>" >> /opt/llm-gateway/data/moe-quant/NOTE.log`.

## What is measured (do not re-derive; cite)

Model: Qwen3.8-Flash-Next GSQ-RCO IQ3_S. Files:
- `/opt/llm-gateway/models/Qwen3.8-Flash-Next/IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-0000{1,2}-of-00002.gguf`
  (`qwen4exp`, 48 layers, 512 experts, 10 used, about 2 MB per expert, 0.98 GB of routed experts per token; the second
  shard is a 28.8 GB per-layer n-gram table read by row).

- **End to end** (`/opt/llm-gateway/data/moe-quant/qwen38/stage_b.py`, 8 prompts: 4 code, 4 English; 256 tokens;
  temperature 0; three interleaved rounds; `summary.py` prints medians):

  | Engine | code tok/s | English tok/s |
  |---|---:|---:|
  | Strata, deployed (config `strata`) | 69.6 to 70.8 | 61.0 to 61.7 |
  | our fork, best (`q-r8-slots`) | 48.4 | 40.1 |
  | our fork, round 7 (`q-warm-draft`) | 47.0 | 39.9 |

  The gap is 1.45 to 1.55 times.
- **Our server round** (MTP 3 plus the round-7 cache, `NOTES-round10.md`): 81.7 ms code / 76.1 ms English, against
  Strata's about 51 / 41 ms.
  - Three draft steps take about 7.5 ms; target verification 70.8 / 65.5 ms.
  - Inside verification (graph-off trace, overlapping): GPU dense 27.6 ms, GPU cache hits 5-6 ms, host-to-device copies
    8.9 ms (cache uploads on the compute stream), host CUDA sync 13.4 ms, CPU expert graphs 35 / 30 ms.
  - CPU samples: libgomp 63-67 % (spin-wait), ggml-cpu 30-33 %.
- **Expert cache hits:** ours 43 % of decode routings. Strata's log of 2026-10-19 (`/opt/llm-gateway/data/logs/iq3_s.log`)
  shows 38 % after the first request, rising to 67 % by the fourth.
  - An older Strata log said 53 % from VRAM plus 15 % read by the GPU over PCIe. The deployed config now has
    `--pcie-frac 0.0`: check what Strata really does today before you build on that.
- **Strata's deployed engine settings** (config `/opt/llm-gateway/data/runtimes/strata/0.1.41-local-1cb045cb/strata-iq3_s.json`,
  field `args`): `--spec 4 --spec-min-p 0.7`, its own MTP head (`--mtp .../strata-data/mtp/rt`), `--expert-cache auto`,
  `--kv int8`, `--prefill auto`, experts from a repacked pack (`--pack .../strata-data/packs/iq3_s`) plus the GGUF.
  - Environment: `STRATA_BF16_TC=1 STRATA_NO_LARGEPAGES=1 STRATA_SELECT_SIMT=1`.
  - It accepts about 2.5 to 3.1 tokens per round (its log prints accepted drafts per request).
  - Source: `/opt/llm-gateway/data/runtimes/strata/0.1.41-local-1cb045cb/src` and `include/`; CPU kernels in
    `src/kernels/cpu/`.
  - Newer upstream (0.2.0-beta, same engine package 0.1.45.3) is in `/opt/llm-gateway/data/build/strata-src`; read it
    if it helps, do not run it.

**Measured and dropped. Do not repeat these unless you find a measured reason the earlier test was wrong:**
- reading misses over PCIe from mapped host memory (CPU 31.7 ms against 44-70 ms on a 4-token window);
- multi-token CPU kernels for missed experts (mean multiplicity 1.24-1.33, needed 1.6);
- cache policy tuning (byte value, decay, per-layer budgets: within noise);
- async upload stream (+1.7 %);
- BF16 to Q8_0/F16 retype of the hyper-connection matrices;
- `--spec-draft-p-min` with MTP (-1 to -8 %);
- draft-context threadpool fix (+1.5 %);
- more threads or SMT (24 OpenMP threads is best).

## Stage 1: Strata's round, measured the same way as ours (at most 75 minutes)

1. Start the deployed Strata server yourself, on a free port, after freeing and holding the GPU:
   - `STRATA_API_KEY=<a random key you make> STRATA_BF16_TC=1 STRATA_NO_LARGEPAGES=1 STRATA_SELECT_SIMT=1
     /opt/llm-gateway/data/venvs/strata/bin/python /opt/llm-gateway/data/runtimes/strata/0.1.41-local-1cb045cb/serve/server.py
     --engine strata --config <the config above> --host 127.0.0.1 --port 18100 --engine-background
     --workspace-dir <a folder in this worktree> --idle-unload 0`;
   - `GET /engine` shows the load, about 65 s.

   Send the same 8 prompts as `stage_b.py` (its `PROMPTS`, `max_tokens` 256, temperature 0) directly to its
   `/v1/chat/completions`, warm-up first.
2. Profile one steady request with Nsight Systems on the engine process (`nsys profile` attached to or following the
   `engine/strata` child; CUDA, OS runtime, CPU sampling), and the same request on our fork's best config. Also read
   Strata's own counters in its log (cache hits, drafts accepted per round).
3. Write **one table, Strata against ours, per round and per token**:
   - tokens per round;
   - round time;
   - GPU dense time;
   - expert work on the GPU (count, ms);
   - expert work on the CPU (count, bytes, ms, achieved GB/s);
   - copies (bytes, ms, which stream);
   - host sync and launch gaps;
   - draft cost;
   - thread wait or spin.

   Name the three largest differences in ms. Mark what is measured and what is inferred.
   **Checkpoint:** commit, NOTE.log line, and write `NOTES-vibe-round1.md` section "Stage 1".
   - Stop and report if no single difference is worth at least 5 % of our round.
   - Also stop and report if the gap is mostly Strata's repacked expert layout, which needs a format change, not a
     runtime change. Then quantify it and stop there.

## Stage 2: the largest difference, in llama.cpp

Implement the fix for the largest measured difference in this worktree:
- behind an environment switch, off by default;
- with the switch off, behaviour bit for bit as before;
- unsupported shapes fall back.

Bound it first with the cheapest experiment you can (switch a part off, count, standalone harness); build the real thing
only if the bound is at least 5 % end to end.

Measure with `stage_b.py`:
- add your configs with the binary `build-cuda/bin/llama-server` of this worktree; keep the existing configs;
- `q-r8-slots` is the control;
- three interleaved rounds, quiet gate.

Correctness:
- perplexity on `/opt/llm-gateway/data/build/texts/test-de.txt` (12 chunks, ctx 512, `-ub 4`), within +0.5 % of the
  control;
- greedy outputs compared with the control, and every difference explained.

The gate for "adopted": at least +3 % on both code and English.

## Deliverable

`NOTES-vibe-round1.md` in this worktree:
- the stage 1 table with the profile files named;
- what you built and why;
- the end-to-end numbers (medians, spread, counts);
- perplexity, outputs;
- what did not work;
- where llama.cpp's ceiling on this machine is in your judgement, with the evidence;
- open questions for the owner.

Also save the final diff as `task-logs/vibe-round1/vibe-round1.patch`. Release the GPU hold, stop the Strata server you
started, and leave no process running.
