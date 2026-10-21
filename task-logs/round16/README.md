# Round 16 reproduction

Run from `/opt/llm-gateway/data/build/llama-vibe`. No packages or models need downloading; the scripts use the existing local model files and Strata Python environment. The gateway checkout and live settings are not modified.

Before rebuilding, preserve the pre-patch `build-cuda/bin` as `task-logs/round16/e2e/original` with symlinks intact. That copy already exists on this machine; do not overwrite it. Its source is ada2198d6; the engine source is unchanged between that commit and the pre-patch 6c8bfda89. Build the changed library only:

```sh
nice -n 10 cmake --build build-cuda --target llama -j 8
python3 task-logs/round16/isolate_runtime.py
```

The isolation script refuses to overwrite its destination. It copies the original runtime, replaces only `libllama.so.0.7.0` and checks every file and symlink. The ignored runtime folders are local artifacts; `isolated-binaries.json` records the exact hashes used for the confirmation. The first exploratory full-server build refreshed UI assets automatically; those files are excluded from the isolated runtime. Use the library-only target to avoid that unrelated build step.

The converter also refuses to overwrite models:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /opt/llm-gateway/data/venvs/strata/bin/python task-logs/round16/make_draft.py q2 q2-identity q2-q8subset q2-q6subset
```

It reads paths declared at its top and writes under `/opt/llm-gateway/data/moe-quant/qwen38/round16-models/`. Q2_0 experts are copied byte-for-byte from Strata; the Q6_K subset is gathered directly from the native target head. All other tensors and tokenizer metadata come from the original external Q8_0 MTP file. Per-tensor SHA256 manifests, metadata checks and gate/up packing controls are saved beside the script.

Standalone checks can be built with the existing compiler and libraries:

```sh
g++ -std=c++17 -O2 -Iggml/include task-logs/round16/check_head.cpp -Lbuild-cuda/bin -Wl,-rpath,/opt/llm-gateway/data/build/llama-vibe/build-cuda/bin -lggml -lggml-cpu -lggml-cuda -lggml-base -o task-logs/round16/check_head
g++ -std=c++17 -O2 -Iinclude -Iggml/include task-logs/round16/check_load.cpp -Lbuild-cuda/bin -Wl,-rpath,/opt/llm-gateway/data/build/llama-vibe/build-cuda/bin -lllama -o task-logs/round16/check_load
CUDA_VISIBLE_DEVICES='' task-logs/round16/check_head
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /opt/llm-gateway/data/venvs/strata/bin/python task-logs/round16/check_mapping.py
```

Use `-fsanitize=address,undefined -fno-omit-frame-pointer` for the sanitizer build of the head checker. `check_head cuda` requires a GPU hold/free/release around the command, just like any other GPU work. Existing results cover CPU, CUDA, sanitizers, swapped IDs, a tiny weight perturbation, invalid/duplicate target IDs, restored controls and the real identity-map model.

The confirmation command (choose a new output name on repetition) is:

```sh
python3 -u task-logs/round16/run_drafts.py --configs base q6gpu --rounds 3 --all-prompts --out confirm-clean --minutes 80
python3 task-logs/round16/analyze.py task-logs/round16/confirm-clean
```

The runner holds/frees/releases the gateway GPU itself; it reads the API key without printing it. The order is A B B A A B, with a fresh model and the same warmup/eight prompts per block. A foreign-load violation discards the entire block and reloads before replay. The analyzer requires all six completed clean blocks, 12 accepted requests per category and configuration, and unique keys. It reports streaming and server rates, per-prompt medians/ranges, paired ratios, output hashes, token and draft counts. `confirm/` is an abandoned earlier attempt and must not enter the result.

After performance confirmation, run the affected quality panels sequentially, with no other GPU work:

```sh
python3 -u task-logs/round16/quality.py base quality-base
python3 -u task-logs/round16/quality.py q6gpu quality-q6gpu
```

These reuse the existing benchmark's five code tests, three simulated tool cases, three German cases and two format cases. Code executes in the benchmark's existing network-disabled, read-only Docker sandbox. Tool calls are simulated locally. The long-context test is excluded because it exceeds the measured 8192-token context. This checks the non-thinking mode used for the speed measurements; it does not establish every language or long-context workload. Exact requests, responses, source hash, individual verdicts and full server logs are archived.
