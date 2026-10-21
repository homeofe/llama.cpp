#!/usr/bin/env python3
"""Copy Strata's packed Q2_0 experts into the existing external MTP GGUF."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'gguf-py'))
import gguf
import numpy as np

BASE = Path('/opt/llm-gateway/models/Qwen3.8-Flash-Next/MTP/mtp-Qwen3.8-Flash-Next-Q8_0.gguf')
STRATA = Path('/opt/llm-gateway/data/strata-data/mtp/mtp-q2_0.gguf')
VOCAB = Path('/opt/llm-gateway/data/strata-data/mtp/rt/draft_vocab.bin')
TARGET = Path('/opt/llm-gateway/models/Qwen3.8-Flash-Next/IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf')
OUT = Path('/opt/llm-gateway/data/moe-quant/qwen38/round16-models')
RT = Path('/opt/llm-gateway/data/strata-data/mtp/rt')


def sha(data):
    return hashlib.sha256(memoryview(np.ascontiguousarray(data)).cast('B')).hexdigest()


def strata_dense():
    payload = np.memmap(RT/'dense.bin', mode='r', dtype=np.uint8)
    tensors = {}
    types = {'q8_0': gguf.GGMLQuantizationType.Q8_0, 'bf16': gguf.GGMLQuantizationType.BF16, 'f32': gguf.GGMLQuantizationType.F32}
    for line in (RT/'dense.txt').read_text().splitlines():
        name, kind, rows, cols, offset, size = line.split()
        rows, cols, offset, size = map(int, (rows, cols, offset, size))
        raw = payload[offset:offset+size]
        if kind == 'q8_0':
            data = raw.reshape(rows, cols//32*34)
        else:
            data = raw.view('<u2' if kind == 'bf16' else '<f4').reshape(rows, cols)
        tensors[name] = (data, types[kind])
    names = {
        'attn_k': 'self_attn.k_proj', 'attn_k_norm': 'self_attn.k_norm',
        'attn_output': 'self_attn.o_proj', 'attn_q': 'self_attn.q_proj',
        'attn_q_norm': 'self_attn.q_norm', 'attn_v': 'self_attn.v_proj',
        'ffn_gate_inp': 'mlp.gate', 'ffn_gate_inp_shexp': 'mlp.shared_expert_gate',
        'ffn_down_shexp': 'mlp.shared_expert.down_proj',
        'ffn_gate_shexp': 'mlp.shared_expert.gate_proj', 'ffn_up_shexp': 'mlp.shared_expert.up_proj',
        'hc_attn_down': 'attn_hyper_connection.input_mix_weight_down',
        'hc_attn_up': 'attn_hyper_connection.input_mix_weight_up',
        'hc_attn_inject': 'attn_hyper_connection.block_inject_weight',
        'hc_attn_norm': 'attn_hyper_connection.hc_norm',
        'hc_ffn_down': 'mlp_hyper_connection.input_mix_weight_down',
        'hc_ffn_up': 'mlp_hyper_connection.input_mix_weight_up',
        'hc_ffn_inject': 'mlp_hyper_connection.block_inject_weight',
        'hc_ffn_norm': 'mlp_hyper_connection.hc_norm',
        'nextn.hc_head_down': 'hyper_connection_mixer.input_mix_weight_down',
        'nextn.hc_head_up': 'hyper_connection_mixer.input_mix_weight_up',
        'nextn.hc_head_norm': 'hyper_connection_mixer.hc_norm',
        'nextn.enorm': 'pre_fc_norm_embedding', 'nextn.hnorm': 'pre_fc_norm_hidden',
        'indexer.k_norm': 'self_attn.indexer.k_layernorm',
        'indexer.q_norm': 'self_attn.indexer.q_layernorm',
    }
    result = {'blk.48.'+dest+'.weight': tensors[src+'.weight'] for dest,src in names.items()}
    e, kind = tensors['fc_embedding.weight']
    h, hkind = tensors['fc_hidden.weight']
    assert kind == hkind == gguf.GGMLQuantizationType.Q8_0
    result['blk.48.nextn.eh_proj.weight'] = (np.concatenate([e,h],axis=1),kind)
    qk, kind = tensors['self_attn.indexer.index_qk_proj.weight']
    assert qk.shape == (640, 2720) and kind == gguf.GGMLQuantizationType.Q8_0
    result['blk.48.indexer.q_proj.weight'] = (qk[:512],kind)
    result['blk.48.indexer.k_proj.weight'] = (qk[512:],kind)
    assert np.array_equal(np.concatenate([result['blk.48.indexer.q_proj.weight'][0],result['blk.48.indexer.k_proj.weight'][0]]),qk)
    assert np.array_equal(result['blk.48.nextn.eh_proj.weight'][0][:,:2720],e)
    assert np.array_equal(result['blk.48.nextn.eh_proj.weight'][0][:,2720:],h)
    return result


def make(variant, base, strata, native, embedding=None):
    path = OUT / (variant + '.gguf')
    pending = OUT / (variant + '.incomplete.gguf')
    if path.exists() or pending.exists():
        raise FileExistsError(path if path.exists() else pending)
    original = {t.name: t for t in base.tensors}
    st = {t.name: t for t in strata.tensors}
    gu = st['mtp.layers.0.mlp.experts.gate_up_proj']
    dn = st['mtp.layers.0.mlp.experts.down_proj']
    assert gu.tensor_type == dn.tensor_type == gguf.GGMLQuantizationType.Q2_0
    assert gu.data.shape == (512, 1280, 720) and dn.data.shape == (512, 2560, 180)
    replacements = {} if variant.startswith('q8exp-') else {
        'blk.48.ffn_gate_exps.weight': (np.ascontiguousarray(gu.data[:, :640, :]), gu.tensor_type),
        'blk.48.ffn_up_exps.weight': (np.ascontiguousarray(gu.data[:, 640:, :]), gu.tensor_type),
        'blk.48.ffn_down_exps.weight': (dn.data, dn.tensor_type),
    }
    if variant != 'q2':
        ids = (np.arange(248320, dtype=np.int64) if variant == 'q2-identity'
               else np.fromfile(VOCAB, dtype='<i4').astype(np.int64))
        assert len(ids) > 0 and len(set(ids.tolist())) == len(ids)
        assert ids.min() >= 0 and ids.max() < 248320
        head = native if variant.endswith('q6subset') else original['output.weight']
        replacements['output.weight'] = (np.ascontiguousarray(head.data[ids]), head.tensor_type)
        replacements['d2t'] = (ids, gguf.GGMLQuantizationType.I64)
    if variant == 'strata-q6subset':
        replacements.update(strata_dense())
        assert embedding.tensor_type == gguf.GGMLQuantizationType.IQ4_XS
        replacements['token_embd.weight'] = (embedding.data, embedding.tensor_type)
        assert set(replacements) == set(original) | {'d2t'}
        for name, (data, kind) in list(replacements.items()):
            if name in original and kind in (gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.BF16):
                replacements[name] = (data.reshape(tuple(original[name].shape[::-1])), kind)

    writer = gguf.GGUFWriter(str(pending), base.fields['general.architecture'].contents())
    for field in base.fields.values():
        if field.name.startswith('GGUF.') or field.name == gguf.Keys.General.ARCHITECTURE:
            continue
        kind = field.types[0]
        writer.add_key_value(field.name, field.contents(), kind,
                             sub_type=field.types[-1] if kind == gguf.GGUFValueType.ARRAY else None)
    entries = []
    for tensor in base.tensors:
        data, kind = replacements.get(tensor.name, (tensor.data, tensor.tensor_type))
        entries.append((tensor.name, data, kind))
    if 'd2t' in replacements:
        entries.append(('d2t', *replacements['d2t']))
    for name, data, kind in entries:
        writer.add_tensor_info(name, data.shape, data.dtype, data.nbytes, kind)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()
    for _, data, _ in entries:
        writer.write_tensor_data(data)
    writer.close()

    # Independent readback checks every tensor and preserves all tokenizer metadata.
    result = gguf.GGUFReader(str(pending))
    actual = {t.name: t for t in result.tensors}
    checks = []
    for name, data, kind in entries:
        digest = sha(data)
        assert actual[name].tensor_type == kind and sha(actual[name].data) == digest, name
        if name in original and name != 'output.weight':
            assert np.array_equal(actual[name].shape, original[name].shape), name
        checks.append(dict(name=name, type=kind.name, bytes=data.nbytes, sha256=digest,
                           changed=name in replacements))
    for key, field in base.fields.items():
        if not key.startswith('GGUF.'):
            assert field.contents() == result.fields[key].contents(), key
    # Reconstruct the original packed gate/up tensor, including its expert and row ordering.
    q2_experts = not variant.startswith('q8exp-')
    if q2_experts:
        gate = actual['blk.48.ffn_gate_exps.weight'].data
        up = actual['blk.48.ffn_up_exps.weight'].data
        assert np.array_equal(np.concatenate([gate, up], axis=1), gu.data)
        assert not np.array_equal(np.concatenate([up, gate], axis=1), gu.data)
    report = dict(path=str(path), bytes=pending.stat().st_size, variant=variant,
                  sources=dict(base=str(BASE), experts=str(STRATA if q2_experts else BASE), vocab=str(VOCAB),
                               head=str(TARGET if variant.endswith('q6subset') else BASE)),
                  tensor_checks=checks, metadata_equal=True, gate_up_roundtrip=q2_experts,
                  swapped_gate_up_negative_control=q2_experts)
    if variant == 'strata-q6subset':
        report['sources'].update(dense=str(RT/'dense.bin'), dense_index=str(RT/'dense.txt'), embedding=str(TARGET))
    pending.rename(path)
    (Path(__file__).parent / (variant + '-manifest.json')).write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(variant=variant, mib=path.stat().st_size/2**20, checks='passed')), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('variants', nargs='+', choices=['q2', 'q2-identity', 'q2-q8subset', 'q2-q6subset', 'q8exp-q8subset', 'q8exp-q6subset', 'strata-q6subset'])
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True)
    base = gguf.GGUFReader(str(BASE))
    strata = gguf.GGUFReader(str(STRATA))
    target = gguf.GGUFReader(str(TARGET)) if any(v.endswith('q6subset') for v in args.variants) else None
    native = next(t for t in target.tensors if t.name == 'output.weight') if target else None
    embedding = next(t for t in target.tensors if t.name == 'token_embd.weight') if target else None
    if native:
        assert native.tensor_type == gguf.GGMLQuantizationType.Q6_K
    for variant in args.variants:
        make(variant, base, strata, native, embedding)
    (OUT / 'all-experts.profile').write_text('llama-moe-profile-v1\n' + ''.join(f'48 {e} 1\n' for e in range(512)))


if __name__ == '__main__':
    main()
