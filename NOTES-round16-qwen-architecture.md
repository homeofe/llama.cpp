# Qwen architecture audit, 2026-10-21

Sources: [Qwen model card](https://huggingface.co/Qwen/Qwen3.8-Flash-Next), [architecture figure](https://qianwen-res.oss-accelerate.aliyuncs.com/Qwen3.8-Flash-Next/architecture.png), [technical report](https://github.com/QwenLM/Qwen3.8-Flash-Next/blob/main/tech_report.pdf), especially section 2.1.2 and Table 4. The fetched figure and report are local reading artifacts, excluded from Git.

The relevant result is **reuse of QSA top-k indices across speculative prediction steps**. Qwen reports mean accepted lengths 4.06 for full attention and 4.07 for QSA with four-step speculation across five benchmark families. These are Qwen's results, not measured on our GPU, quantization, prompts or engine. They establish a concrete design to examine, not a predicted tok/s gain here.

## Code comparison

| Component | Actual Strata 1cb045cb | This llama.cpp branch | Consequence |
|---|---|---|---|
| MTP attention | Dense attention; default last 32768 cells (`src/core/mtp.cpp`, `src/program/generate.cpp`) | Draft GGUF's final compression ratio is 4; `graph_mtp` invokes QSA on each step/catch-up | Original paths do different work even at short contexts. Experimental `LLAMA_MTP_DENSE=1` disables only draft QSA, requires context <=32768, and uses the existing dense causal-attention path. |
| MTP QSA index reuse | Not applicable to this dense draft | No reuse; `build_qsa_sel` recomputes scores/top-k/mask | A separate long-context experiment should preserve the chosen complete blocks across draft steps while updating causal tails. Merely caching a whole old mask would be wrong. |
| Target QSA with all visible blocks within budget | Top-k kernel returns identity indices; query/scoring work still exists | Builds full selection/mask path | There is a short-context specialization opportunity, but Strata does not skip its entire indexer. Keep indexer K/pool state valid for later long contexts. |
| N-gram embedding (PLE) | Prefetch, with actual gather overlapped with early layers | `llama_prefetch_rows` in the graph input; embedding/PLE expanded in first split | Already implemented in both. Trace the overlap before changing it. Learned PLE embeddings are distinct from suffix/prompt-lookup drafting. |
| Gated residual / four streams | GR read/write, full widened hidden passed to MTP | HC mix/combine, full hidden passed to MTP | This is model architecture already supported, not an unused optional speed switch. Draft hidden normalization differs by engine setting; the experimental global-norm switch is documented separately. |

## Bound and next decision

The previous graphs-off round-14 trace contains 988 lightning-indexer kernel launches totaling approximately 6.29 ms across the entire capture (60 decode rounds plus capture setup). This does not isolate the MTP share or selection-mask/launch overhead. It cannot justify a large speedup claim from indexer arithmetic alone. The earlier graph-node-incomplete Strata trace is not a valid bound.

First run the user-requested Strata-aligned short-context setup: actual draft weights, global hidden norm, full-subset confidence at 0.7, MTP/suffix policy and the same resident target expert IDs. Dense draft attention is part of that reference. After a working measured reference, evaluate Qwen's reuse design separately at contexts where QSA excludes blocks; use acceptance/quality and end-to-end timing, not only kernel speed. Do not reopen closed target quantized-kernel axes on the strength of this paper.
