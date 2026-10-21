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


def sha(data):
    return hashlib.sha256(memoryview(np.ascontiguousarray(data)).cast('B')).hexdigest()


def make(variant, base, strata, native):
    path = OUT / (variant + '.gguf')
    if path.exists():
        raise FileExistsError(path)
    original = {t.name: t for t in base.tensors}
    st = {t.name: t for t in strata.tensors}
    gu = st['mtp.layers.0.mlp.experts.gate_up_proj']
    dn = st['mtp.layers.0.mlp.experts.down_proj']
    assert gu.tensor_type == dn.tensor_type == gguf.GGMLQuantizationType.Q2_0
    assert gu.data.shape == (512, 1280, 720) and dn.data.shape == (512, 2560, 180)
    replacements = {
        'blk.48.ffn_gate_exps.weight': (np.ascontiguousarray(gu.data[:, :640, :]), gu.tensor_type),
        'blk.48.ffn_up_exps.weight': (np.ascontiguousarray(gu.data[:, 640:, :]), gu.tensor_type),
        'blk.48.ffn_down_exps.weight': (dn.data, dn.tensor_type),
    }
    if variant != 'q2':
        ids = (np.arange(248320, dtype=np.int64) if variant == 'q2-identity'
               else np.fromfile(VOCAB, dtype='<i4').astype(np.int64))
        assert len(ids) > 0 and len(set(ids.tolist())) == len(ids)
        assert ids.min() >= 0 and ids.max() < 248320
        head = native if variant == 'q2-q6subset' else original['output.weight']
        replacements['output.weight'] = (np.ascontiguousarray(head.data[ids]), head.tensor_type)
        replacements['d2t'] = (ids, gguf.GGMLQuantizationType.I64)

    writer = gguf.GGUFWriter(str(path), base.fields['general.architecture'].contents())
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
    result = gguf.GGUFReader(str(path))
    actual = {t.name: t for t in result.tensors}
    checks = []
    for name, data, kind in entries:
        digest = sha(data)
        assert actual[name].tensor_type == kind and sha(actual[name].data) == digest, name
        checks.append(dict(name=name, type=kind.name, bytes=data.nbytes, sha256=digest,
                           changed=name in replacements))
    for key, field in base.fields.items():
        if not key.startswith('GGUF.'):
            assert field.contents() == result.fields[key].contents(), key
    # Reconstruct the original packed gate/up tensor, including its expert and row ordering.
    gate = actual['blk.48.ffn_gate_exps.weight'].data
    up = actual['blk.48.ffn_up_exps.weight'].data
    assert np.array_equal(np.concatenate([gate, up], axis=1), gu.data)
    assert not np.array_equal(np.concatenate([up, gate], axis=1), gu.data)
    report = dict(path=str(path), bytes=path.stat().st_size, variant=variant,
                  sources=dict(base=str(BASE), experts=str(STRATA), vocab=str(VOCAB)),
                  tensor_checks=checks, metadata_equal=True, gate_up_roundtrip=True,
                  swapped_gate_up_negative_control=True)
    (Path(__file__).parent / (variant + '-manifest.json')).write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(variant=variant, mib=path.stat().st_size/2**20, checks='passed')), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('variants', nargs='+', choices=['q2', 'q2-identity', 'q2-q8subset', 'q2-q6subset'])
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True)
    base = gguf.GGUFReader(str(BASE))
    strata = gguf.GGUFReader(str(STRATA))
    target = gguf.GGUFReader(str(TARGET)) if 'q2-q6subset' in args.variants else None
    native = next(t for t in target.tensors if t.name == 'output.weight') if target else None
    if native:
        assert native.tensor_type == gguf.GGMLQuantizationType.Q6_K
    for variant in args.variants:
        make(variant, base, strata, native)
    (OUT / 'all-experts.profile').write_text('llama-moe-profile-v1\n' + ''.join(f'48 {e} 1\n' for e in range(512)))


if __name__ == '__main__':
    main()
