#!/usr/bin/env python3
"""The real loader must accept valid maps and reject unsafe target IDs."""
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / 'gguf-py'))
import gguf

source = Path('/opt/llm-gateway/data/moe-quant/qwen38/round16-models/q2-q8subset.gguf')
scratch = source.with_name('invalid-map-check.gguf')
if scratch.exists():
    raise FileExistsError(scratch)
subprocess.run(['cp', '--reflink=auto', str(source), str(scratch)], check=True)
reader = gguf.GGUFReader(str(scratch))
mapping = next(t for t in reader.tensors if t.name == 'd2t')
offset = mapping.data_offset
original = scratch.open('rb')
original.seek(offset)
saved = original.read(16)
original.close()
cases = [('valid', saved, 0), ('negative', struct.pack('<q', -1)+saved[8:], 1),
         ('too-large', struct.pack('<q', 248320)+saved[8:], 1), ('duplicate', saved[:8]*2, 1)]
results = []
try:
    for name, payload, expected in cases:
        with scratch.open('r+b') as f:
            f.seek(offset)
            f.write(payload)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='')
        r = subprocess.run([str(HERE / 'check_load'), str(scratch)], env=env, capture_output=True, text=True, timeout=30)
        (HERE / ('mapping-' + name + '.txt')).write_text(r.stdout+r.stderr)
        assert r.returncode == expected, (name, r.returncode)
        if expected:
            assert 'd2t contains an invalid or duplicate target token ID' in r.stderr
        results.append(dict(case=name, returncode=r.returncode, expected=expected))
    with scratch.open('r+b') as f:
        f.seek(offset)
        f.write(saved)
    r = subprocess.run([str(HERE / 'check_load'), str(scratch)], env=env, capture_output=True, text=True, timeout=30)
    assert r.returncode == 0
    results.append(dict(case='restored', returncode=r.returncode, expected=0))
finally:
    scratch.unlink()
(HERE / 'mapping-checks.json').write_text(json.dumps(results, indent=2)+'\n')
print(json.dumps(results))
