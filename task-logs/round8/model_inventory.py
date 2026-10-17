#!/usr/bin/env python3
import collections
import json
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'gguf-py'))
from gguf import GGUFReader

models = {
    'qwen38': '/opt/llm-gateway/models/Qwen3.8-Flash-Next/IQ3_S/*.gguf',
    'qwen38_mtp': '/opt/llm-gateway/models/Qwen3.8-Flash-Next/MTP/mtp-Qwen3.8-Flash-Next-Q8_0.gguf',
    'ling': '/opt/llm-gateway/models/Ling-3.0-flash-GGUF/Ling-3.0-flash-IQ4_XS/*.gguf',
}
import glob
result: dict[str, dict[str, Any]] = {}
for model, pattern in models.items():
    layers = collections.defaultdict(list)
    metadata = {}
    for filename in sorted(glob.glob(pattern)):
        reader = GGUFReader(filename)
        for key, field in reader.fields.items():
            if key.endswith(('expert_used_count', 'expert_count', 'block_count')):
                metadata[key] = field.contents()
        for tensor in reader.tensors:
            if '_exps.' not in tensor.name:
                continue
            shape = list(map(int, tensor.shape))
            if len(shape) != 3:
                continue
            layers[int(tensor.name.split('.')[1])].append(dict(name=tensor.name, type=tensor.tensor_type.name,
                type_id=int(tensor.tensor_type), file=filename, offset=int(tensor.data_offset),
                shape=shape, expert_bytes=int(tensor.n_bytes)//shape[2]))
    result[model] = dict(layers=dict(layers), metadata=metadata)
    groups = collections.defaultdict(list)
    for layer, tensors in layers.items():
        signature = tuple((t['name'].split('.',2)[2], t['type'], tuple(t['shape'])) for t in sorted(tensors,key=lambda t:t['name']))
        groups[signature].append(layer)
    result[model]['groups'] = [dict(layers=ids, expert_bytes=sum(t['expert_bytes'] for t in layers[ids[0]]), tensors=layers[ids[0]]) for ids in groups.values()]
Path('task-logs/round8/model-inventory.json').write_text(json.dumps(result,indent=2)+'\n')
for model, data in result.items():
    print(model)
    for g in data['groups']:
        print('layers', g['layers'], 'bytes/expert', g['expert_bytes'], [(t['type'],t['shape']) for t in g['tensors']])
