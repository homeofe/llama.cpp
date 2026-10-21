#!/usr/bin/env python3
"""Translate Strata's first 1815 resident expert IDs; size llama's separate banks."""
from pathlib import Path
import collections
import hashlib
import json
import struct
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]/'gguf-py'))
import gguf

raw = Path('/opt/llm-gateway/data/strata-data/expert-profile.bin').read_bytes()
assert raw[:4] == b'STRP'
version, layers, experts, slots, ranked = struct.unpack_from('<5I', raw, 4)
assert (version, layers, experts) == (1,48,512) and 1815 <= ranked <= slots
assert len(raw) >= 24+ranked*4
pairs = [struct.unpack_from('<HH',raw,24+4*i) for i in range(1815)]
assert len(set(pairs)) == 1815 and all(l<48 and e<512 for l,e in pairs)
(HERE/'strata-target.profile').write_text('llama-moe-profile-v1\n'+''.join(f'{l} {e} {1815-i}\n' for i,(l,e) in enumerate(pairs)))
model_dir = Path('/opt/llm-gateway/models/Qwen3.8-Flash-Next/IQ3_S')
by_layer = collections.defaultdict(list)
for path in sorted(model_dir.glob('*of-00002.gguf')):
    for tensor in gguf.GGUFReader(path).tensors:
        if tensor.name.startswith('blk.') and tensor.name.endswith(('_gate_exps.weight','_up_exps.weight','_down_exps.weight')):
            by_layer[int(tensor.name.split('.')[1])].append((tensor.name.split('.')[2], int(tensor.tensor_type),
                tuple(map(int,tensor.shape)), tensor.n_bytes//512))
assert len(by_layer) == 48 and all(len(v)==3 for v in by_layer.values())
groups = collections.defaultdict(list)
for layer, tensors in by_layer.items():
    groups[tuple(sorted(tensors))].append(layer)
total = sum(sum(t[3] for t in key)*512*len(ls) for key,ls in groups.items())
summary = []
needed = 0
for key,ls in groups.items():
    count = sum(layer in ls for layer,expert in pairs)
    size = sum(t[3] for t in key)
    fraction = size*512*len(ls)/total
    # CUDA bank/map sizes are multiples of the backend's 256-byte alignment here.
    alloc = (count+1)*size+len(ls)*512*4
    budget = alloc/fraction
    needed = max(needed,budget)
    summary.append(dict(layers=ls, expert_bytes=size, required_slots=count, budget_mib=budget/2**20))
report = dict(profile_source_sha256=hashlib.sha256(raw).hexdigest(), count=1815,
              groups=summary, minimum_budget_mib=needed/2**20)
(HERE/'strata-target-profile.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report))
