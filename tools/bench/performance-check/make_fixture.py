#!/usr/bin/env python3
"""Small deterministic Mixtral-shaped fixture for placement/state checks, not performance."""
import pathlib, sys
import numpy as np
ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'gguf-py'))
import gguf
rng = np.random.default_rng(42)
out = ROOT/'build-cuda/audit-moe.gguf'
w = gguf.GGUFWriter(out, 'llama')
vocab = gguf.GGUFReader(ROOT/'models/ggml-vocab-llama-spm.gguf')
for f in vocab.fields.values():
    if not f.name.startswith('tokenizer.'):
        continue
    vt = f.types[0]
    w.add_key_value(f.name, f.contents(), vt, sub_type=f.types[-1] if vt == gguf.GGUFValueType.ARRAY else None)
v = len(vocab.fields['tokenizer.ggml.tokens'].contents())
w.add_name('performance-check-moe')
w.add_context_length(16384)
w.add_embedding_length(128)
w.add_block_count(2)
w.add_feed_forward_length(128)
w.add_head_count(4)
w.add_head_count_kv(4)
w.add_layer_norm_rms_eps(1e-5)
w.add_expert_count(8)
w.add_expert_used_count(2)
w.add_rope_dimension_count(32)
def tensor(name, shape, norm=False, quant=False):
    a = np.ones(shape, np.float32) if norm else (rng.standard_normal(shape)*.02).astype(np.float32)
    if quant:
        q = gguf.GGMLQuantizationType.Q4_0
        w.add_tensor(name, gguf.quants.quantize(a, q), raw_dtype=q)
    else:
        w.add_tensor(name, a)
tensor('token_embd.weight', (v,128))
tensor('output_norm.weight', (128,), norm=True)
for i in range(2):
    for n in ('attn_norm','ffn_norm'):
        tensor(f'blk.{i}.{n}.weight',(128,),norm=True)
    for n in ('attn_q','attn_k','attn_v','attn_output'):
        tensor(f'blk.{i}.{n}.weight',(128,128))
    tensor(f'blk.{i}.ffn_gate_inp.weight',(8,128))
    for n in ('up','gate','down'):
        tensor(f'blk.{i}.ffn_{n}_exps.weight',(8,128,128),quant=True)
w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
print(out, out.stat().st_size)
